import http.client
import json
import socket
import threading
import unittest
from functools import lru_cache

import numpy as np
import pandas as pd

from fraudguard.data import generate_transactions
from fraudguard.features import FEATURE_COLUMNS, build_features
from fraudguard.pipeline import run_experiment
from fraudguard.serving import (
    MAX_BODY_BYTES,
    OnlineScorer,
    RequestError,
    make_server,
    parse_transaction,
)
from tests.helpers import small_config, stub_bundle


def payload(**over):
    p = dict(timestamp="2025-03-01T10:00:00", customer_id="C1", merchant_id="M1",
             merchant_category="grocery", channel="pos", country="US",
             customer_home_country="US", device_id="D1", amount=25.0)
    p.update(over)
    return p


class TestParseTransaction(unittest.TestCase):
    def test_valid_payload_is_normalized(self):
        tx = parse_transaction(payload(customer_id="  C1  ", amount=7, transaction_id=42))
        self.assertEqual(tx["customer_id"], "C1")
        self.assertEqual(tx["amount"], 7.0)
        self.assertEqual(tx["external_id"], "42")
        self.assertEqual(tx["timestamp"], pd.Timestamp("2025-03-01 10:00:00"))

    def test_timezone_is_converted_to_naive_utc(self):
        self.assertEqual(parse_transaction(payload(timestamp="2025-03-01T10:00:00+02:00"))[
            "timestamp"], pd.Timestamp("2025-03-01 08:00:00"))
        self.assertEqual(parse_transaction(payload(timestamp="2025-03-01T10:00:00Z"))[
            "timestamp"], pd.Timestamp("2025-03-01 10:00:00"))

    def test_bad_payloads(self):
        bad = [None, [], "x", 5, {}]
        bad += [payload(amount=a) for a in (0, -1, "10", True, None, float("nan"), float("inf"),
                                              1e12, [1])]
        bad += [payload(timestamp=t) for t in ("yesterday", "", None, 123, "2025-13-45")]
        bad += [payload(channel="carrier-pigeon"), payload(channel=5)]
        bad += [payload(**{f: v}) for f in ("customer_id", "device_id", "country")
                for v in ("", "   ", None, 5, "x" * 129)]
        bad += [payload(transaction_id=v) for v in ("", True, 1.5, [], {})]
        for p in bad:
            with self.subTest(payload=str(p)[:80]), self.assertRaises(RequestError) as ctx:
                parse_transaction(p)
            self.assertEqual(ctx.exception.status, 422)

    def test_each_required_field_is_required(self):
        for f in payload():
            p = payload()
            del p[f]
            with self.subTest(missing=f), self.assertRaisesRegex(RequestError, "missing fields"):
                parse_transaction(p)

    def test_unknown_extra_fields_are_ignored_and_labels_cannot_be_injected(self):
        tx = parse_transaction(payload(is_fraud=1, whatever="x"))
        self.assertNotIn("is_fraud", tx)


class TestOnlineScorerSemantics(unittest.TestCase):
    def test_history_accumulates_per_customer(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99))
        a = [sc.score(payload(timestamp=f"2025-03-01T1{i}:00:00", customer_id="A"))["score"]
             for i in range(3)]
        b = sc.score(payload(customer_id="B"))["score"]
        self.assertEqual(a, [0.0, 1.0, 2.0])
        self.assertEqual(b, 0.0)

    def test_out_of_order_arrival_never_uses_the_future(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99))
        sc.score(payload(timestamp="2025-03-01T10:00:00"))
        late_arrival = sc.score(payload(timestamp="2025-03-01T09:00:00"))
        self.assertEqual(late_arrival["score"], 0.0)  # the 10:00 txn is in ITS future
        later = sc.score(payload(timestamp="2025-03-01T11:00:00"))
        self.assertEqual(later["score"], 2.0)

    def test_window_features_work_online(self):
        sc = OnlineScorer(stub_bundle("txn_count_1h", threshold=99))
        sc.score(payload(timestamp="2025-03-01T10:00:00"))
        sc.score(payload(timestamp="2025-03-01T10:10:00"))
        self.assertEqual(sc.score(payload(timestamp="2025-03-01T10:20:00"))["score"], 2.0)
        self.assertEqual(sc.score(payload(timestamp="2025-03-01T12:00:00"))["score"], 0.0)

    def test_decision_threshold_and_reasons(self):
        sc = OnlineScorer(stub_bundle("amount", threshold=100.0))
        low = sc.score(payload(amount=10.0))
        self.assertEqual((low["decision"], low["reasons"]), ("approve", []))
        high = sc.score(payload(amount=500.0, timestamp="2025-03-01T11:00:00"))
        self.assertEqual(high["decision"], "review")
        self.assertEqual(high["reasons"][0]["feature"], "amount")
        edge = sc.score(payload(amount=100.0, timestamp="2025-03-01T12:00:00"))
        self.assertEqual(edge["decision"], "review")  # score >= threshold

    def test_infinite_threshold_never_flags_and_is_json_safe(self):
        sc = OnlineScorer(stub_bundle("amount", threshold=float("inf")))
        r = sc.score(payload(amount=1e8))
        self.assertEqual(r["decision"], "approve")
        self.assertIsNone(r["threshold"])
        json.dumps(r, allow_nan=False)

    def test_duplicate_transaction_id_is_a_conflict_and_does_not_change_history(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99))
        sc.score(payload(transaction_id="T1"))
        with self.assertRaises(RequestError) as ctx:
            sc.score(payload(transaction_id="T1", timestamp="2025-03-01T10:05:00"))
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(sc.score(payload(timestamp="2025-03-01T10:10:00"))["score"], 1.0)
        # the same external id for ANOTHER customer is fine
        sc.score(payload(customer_id="OTHER", transaction_id="T1"))

    def test_invalid_request_does_not_touch_state(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99))
        with self.assertRaises(RequestError):
            sc.score(payload(amount=-5))
        self.assertEqual(sc.score(payload())["score"], 0.0)

    def test_unknown_category_gives_a_warning_not_an_error(self):
        sc = OnlineScorer(stub_bundle("amount", threshold=1e9))
        r = sc.score(payload(merchant_category="space_tourism"))
        self.assertEqual(len(r["warnings"]), 1)
        self.assertEqual(sc.score(payload(timestamp="2025-03-01T10:01:00"))["warnings"], [])

    def test_history_cap_bounds_memory(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99),
                          max_history_per_customer=3)
        for i in range(6):
            r = sc.score(payload(timestamp=f"2025-03-01T1{i}:00:00", transaction_id=f"T{i}"))
        self.assertEqual(len(sc._history["C1"]), 3)
        self.assertEqual(len(sc._seen_ids["C1"]), 3)
        self.assertEqual(r["score"], 3.0)  # documented limitation: capped history
        with self.assertRaises(ValueError):
            OnlineScorer(stub_bundle(), max_history_per_customer=0)

    def test_warm_start_loads_history(self):
        df = generate_transactions(2000, 0.05)
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=1e9))
        self.assertEqual(sc.warm_start(df), 2000)
        cust = df["customer_id"].iloc[0]
        n_prior = int((df["customer_id"] == cust).sum())
        later = (df["timestamp"].max() + pd.Timedelta(days=1)).isoformat()
        r = sc.score(payload(customer_id=cust, timestamp=later))
        self.assertEqual(r["score"], float(n_prior))
        known_id = int(df.loc[df["customer_id"] == cust, "transaction_id"].iloc[0])
        with self.assertRaises(RequestError):  # warm-start ids count as already scored
            sc.score(payload(customer_id=cust, timestamp=later, transaction_id=known_id))
        with self.assertRaises(ValueError):
            sc.warm_start(df.drop(columns="amount"))

    def test_thread_safety(self):
        sc = OnlineScorer(stub_bundle("cust_prior_txn_count", threshold=99))
        errors = []

        def work(k):
            try:
                for i in range(25):
                    sc.score(payload(customer_id=f"T{k}", timestamp=f"2025-03-01T10:{i:02d}:00"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(sc._seq, 200)
        self.assertTrue(all(len(h) == 25 for h in sc._history.values()))

    def test_info(self):
        info = OnlineScorer(stub_bundle(threshold=5.0)).info()
        self.assertEqual(info["threshold"], 5.0)
        self.assertEqual(info["features"], FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def real_bundle_and_data():
    df = generate_transactions(6000, 0.05)
    return run_experiment(df, small_config()).bundle, df


class TestTrainingServingParity(unittest.TestCase):
    def test_online_scores_equal_batch_scores(self):
        """The most important production test: streaming == batch, row by row."""
        bundle, df = real_bundle_and_data()
        n = 500
        batch = bundle["model"].score(build_features(df).head(n))
        sc = OnlineScorer(bundle)
        online = []
        for rec in df.head(n).to_dict("records"):
            p = {k: rec[k] for k in ("customer_id", "merchant_id", "merchant_category",
                                     "channel", "country", "customer_home_country",
                                     "device_id", "amount")}
            p["timestamp"] = rec["timestamp"].isoformat()
            p["transaction_id"] = str(rec["transaction_id"])
            online.append(sc.score(p)["score"])
        np.testing.assert_allclose(online, batch, rtol=0, atol=1e-9)


class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(OnlineScorer(stub_bundle("amount", threshold=100.0)), port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        out = json.loads(resp.read().decode())
        conn.close()
        return resp.status, out

    def raw(self, text: bytes):
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as s:
            s.sendall(text)
            return int(s.recv(4096).split(b" ")[1])

    def test_health_and_model_info(self):
        self.assertEqual(self.call("GET", "/health"), (200, {"status": "ok"}))
        status, info = self.call("GET", "/model-info")
        self.assertEqual((status, info["threshold"]), (200, 100.0))

    def test_score_ok_and_flagged(self):
        status, out = self.call("POST", "/score", payload(customer_id="H1", amount=900.0))
        self.assertEqual(status, 200)
        self.assertEqual(out["decision"], "review")
        self.assertEqual(out["reasons"][0]["feature"], "amount")
        status, out = self.call("POST", "/score", payload(customer_id="H1", amount=5.0,
                                                         timestamp="2025-03-01T10:30:00"))
        self.assertEqual((status, out["decision"]), (200, "approve"))

    def test_error_codes(self):
        cases = [
            ("POST", "/score", b"{not json", None, 400),
            ("POST", "/score", b"", None, 400),
            ("POST", "/score", b"\xff\xfe", None, 400),
            ("POST", "/score", b"[1, 2]", None, 422),
            ("POST", "/score", b'{"amount": NaN}', None, 422),
            ("POST", "/score", json.dumps(payload(amount=-1)).encode(), None, 422),
            ("POST", "/score", json.dumps(payload()).encode(), {"Content-Type": "text/plain"}, 415),
            ("POST", "/nope", b"{}", None, 404),
            ("GET", "/nope", None, None, 404),
            ("GET", "/score", None, None, 405),
        ]
        for method, path, body, headers, expected in cases:
            with self.subTest(method=method, path=path, body=str(body)[:30]):
                status, out = self.call(method, path, body, headers)
                self.assertEqual(status, expected)
                self.assertIn("error", out)

    def test_duplicate_returns_409(self):
        p = payload(customer_id="H2", transaction_id="dup-1")
        self.assertEqual(self.call("POST", "/score", p)[0], 200)
        self.assertEqual(self.call("POST", "/score", p)[0], 409)

    def test_missing_and_invalid_content_length(self):
        base = b"POST /score HTTP/1.0\r\nContent-Type: application/json\r\n"
        self.assertEqual(self.raw(base + b"\r\n"), 411)
        self.assertEqual(self.raw(base + b"Content-Length: abc\r\n\r\n"), 400)
        self.assertEqual(self.raw(base + b"Content-Length: -5\r\n\r\n"), 400)
        too_big = f"Content-Length: {MAX_BODY_BYTES + 1}\r\n\r\n".encode()
        self.assertEqual(self.raw(base + too_big), 413)

    def test_unexpected_error_returns_500_without_leaking_details(self):
        class Boom(OnlineScorer):
            def score(self, payload):
                raise RuntimeError("secret internal detail")

        server = make_server(Boom(stub_bundle()), port=0)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
            with self.assertLogs("fraudguard.serving", level="ERROR"):
                conn.request("POST", "/score", json.dumps(payload()).encode(),
                             {"Content-Type": "application/json"})
                resp = conn.getresponse()
                body = resp.read().decode()
            self.assertEqual(resp.status, 500)
            self.assertNotIn("secret", body)
        finally:
            server.shutdown()
            server.server_close()

    def test_concurrent_requests(self):
        results = []

        def hit(k):
            results.append(self.call("POST", "/score", payload(customer_id=f"CC{k}"))[0])

        threads = [threading.Thread(target=hit, args=(k,)) for k in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(results, [200] * 20)


if __name__ == "__main__":
    unittest.main()
