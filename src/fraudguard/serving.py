"""Real-time scoring service (standard library only; no web framework needed).

``OnlineScorer`` keeps each customer's history in memory and calls the SAME
``build_features`` used in training, so training and serving cannot drift apart.
Production note: the in-memory history stands in for a feature store (e.g. Redis); it is
lost on restart unless warmed with ``warm_start``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from fraudguard.config import CHANNELS, MERCHANT_CATEGORIES
from fraudguard.explain import reason_codes
from fraudguard.features import KNOWN_CATEGORIES, build_features

log = logging.getLogger("fraudguard.serving")

MAX_BODY_BYTES = 64 * 1024
_STRING_FIELDS = ("customer_id", "merchant_id", "merchant_category", "channel", "country",
                  "customer_home_country", "device_id")
_REQUIRED_FIELDS = ("timestamp", "amount", *_STRING_FIELDS)
_MAX_STR = 128
_MAX_AMOUNT = 1e9


class RequestError(Exception):
    """A problem caused by the caller; carries the HTTP status to return."""

    def __init__(self, message: str, status: int = HTTPStatus.UNPROCESSABLE_ENTITY):
        super().__init__(message)
        self.status = int(status)


def parse_transaction(payload) -> dict:
    """Validate and normalize one incoming transaction. Raises RequestError (422)."""
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    missing = [f for f in _REQUIRED_FIELDS if f not in payload]
    if missing:
        raise RequestError(f"missing fields: {missing}")
    out = {}
    for f in _STRING_FIELDS:
        v = payload[f]
        if not isinstance(v, str) or not v.strip():
            raise RequestError(f"'{f}' must be a non-empty string")
        if len(v) > _MAX_STR:
            raise RequestError(f"'{f}' is too long (max {_MAX_STR} characters)")
        out[f] = v.strip()
    if out["channel"] not in CHANNELS:
        raise RequestError(f"'channel' must be one of {list(CHANNELS)}")
    amount = payload["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise RequestError("'amount' must be a number")
    if not math.isfinite(amount) or amount <= 0 or amount > _MAX_AMOUNT:
        raise RequestError(f"'amount' must be > 0 and <= {_MAX_AMOUNT:g}")
    out["amount"] = float(amount)
    ts = payload["timestamp"]
    if not isinstance(ts, str):
        raise RequestError("'timestamp' must be an ISO-8601 string")
    try:
        t = pd.Timestamp(ts)
    except (ValueError, TypeError) as exc:
        raise RequestError("'timestamp' is not a valid ISO-8601 time") from exc
    if pd.isna(t):
        raise RequestError("'timestamp' is not a valid ISO-8601 time")
    out["timestamp"] = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t  # naive = UTC
    ext = payload.get("transaction_id")
    if ext is not None:
        if isinstance(ext, bool) or not isinstance(ext, (str, int)) or not str(ext).strip():
            raise RequestError("'transaction_id' must be a non-empty string or integer")
        out["external_id"] = str(ext)
    return out


class OnlineScorer:
    def __init__(self, bundle: dict, max_history_per_customer: int = 10_000):
        if max_history_per_customer < 1:
            raise ValueError("max_history_per_customer must be >= 1")
        self.bundle = bundle
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.reference = bundle["reference"]
        self.max_history = max_history_per_customer
        self._history: dict[str, list[dict]] = {}
        self._seen_ids: dict[str, set] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def info(self) -> dict:
        thr = self.threshold if math.isfinite(self.threshold) else None
        return {"model": self.bundle["model_name"], "threshold": thr,
                "features": list(self.bundle["feature_columns"]),
                "trained_at": self.bundle.get("trained_at"),
                "customers_in_memory": len(self._history)}

    def _append(self, row: dict) -> None:
        hist = self._history.setdefault(row["customer_id"], [])
        hist.append(row)
        if len(hist) > self.max_history:
            dropped = hist.pop(0)
            self._seen_ids.get(row["customer_id"], set()).discard(dropped.get("external_id"))

    def warm_start(self, df: pd.DataFrame) -> int:
        """Load past transactions (e.g. the last weeks) so customers are not 'new'."""
        need = ["timestamp", "amount", "transaction_id", *_STRING_FIELDS]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"warm-start data is missing columns: {missing}")
        with self._lock:
            for rec in df.sort_values(["timestamp", "transaction_id"], kind="stable").to_dict(
                    "records"):
                self._seq += 1
                row = {f: rec[f] for f in _STRING_FIELDS}
                row.update(timestamp=pd.Timestamp(rec["timestamp"]), amount=float(rec["amount"]),
                           transaction_id=self._seq, external_id=str(rec["transaction_id"]))
                self._seen_ids.setdefault(row["customer_id"], set()).add(row["external_id"])
                self._append(row)
        return len(df)

    def score(self, payload) -> dict:
        tx = parse_transaction(payload)
        cust = tx["customer_id"]
        with self._lock:
            ext = tx.get("external_id")
            if ext is not None and ext in self._seen_ids.get(cust, set()):
                raise RequestError(
                    f"transaction_id {ext!r} was already scored", HTTPStatus.CONFLICT)
            self._seq += 1
            row = {k: v for k, v in tx.items() if k != "external_id"}
            row.update(transaction_id=self._seq, external_id=ext)
            frame = pd.DataFrame(self._history.get(cust, []) + [row])
            X = build_features(frame.drop(columns="external_id")).iloc[[-1]]
            score = float(self.model.score(X)[0])
            flagged = bool(score >= self.threshold)
            reasons = reason_codes(self.model, X.iloc[0], self.reference) if flagged else []
            self._append(row)
            if ext is not None:
                self._seen_ids.setdefault(cust, set()).add(ext)
        warnings = []
        if tx["merchant_category"] not in KNOWN_CATEGORIES:
            warnings.append(f"unknown merchant_category {tx['merchant_category']!r}; "
                            f"known: {list(MERCHANT_CATEGORIES)}")
        return {
            "transaction_id": ext if ext is not None else row["transaction_id"],
            "score": score,
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
            "decision": "review" if flagged else "approve",
            "reasons": reasons,
            "warnings": warnings,
            "model": self.bundle["model_name"],
        }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def make_server(scorer: OnlineScorer, host: str = "127.0.0.1", port: int = 8000):
    """Build (but do not start) a threaded HTTP server. Use port 0 for a free port."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "FraudGuard/0.1"

        def log_message(self, fmt, *args):  # route access logs to logging, not stderr
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status, body: dict):
            data = json.dumps(body, default=_json_default, allow_nan=False).encode()
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status, message):
            self._send(status, {"error": message})

        def do_GET(self):
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/model-info":
                self._send(HTTPStatus.OK, scorer.info())
            elif self.path == "/score":
                self._error(HTTPStatus.METHOD_NOT_ALLOWED, "use POST /score")
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown path")

        def do_POST(self):
            if self.path != "/score":
                return self._error(HTTPStatus.NOT_FOUND, "unknown path")
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "use application/json")
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                return self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            try:
                length = int(raw_len)
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length < 0:
                return self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length > MAX_BODY_BYTES:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "body is not valid JSON")
            try:
                self._send(HTTPStatus.OK, scorer.score(payload))
            except RequestError as exc:
                self._error(exc.status, str(exc))
            except Exception:  # never leak internals to the caller
                log.exception("unexpected error while scoring")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

    return ThreadingHTTPServer((host, port), Handler)
