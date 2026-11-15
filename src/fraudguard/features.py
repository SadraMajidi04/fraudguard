"""Leak-free feature engineering.

GOLDEN RULE: the features of a transaction may only use information that was available
BEFORE it happened (the transaction itself + that customer's strictly earlier history).
No labels are used. This is verified in tests/test_features.py by checking that deleting
all future rows never changes a row's features.

The same function is used for training and for online scoring, which prevents
training/serving skew (a classic production bug).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import CASH_CATEGORY, HIGH_RISK_CATEGORIES, MERCHANT_CATEGORIES

FEATURE_COLUMNS = [
    "amount", "log_amount", "hour", "is_night", "day_of_week",
    "is_online", "is_atm", "category_code", "is_high_risk_category",
    "is_foreign", "is_new_device", "is_new_country",
    "cust_prior_txn_count", "cust_prior_mean_amount", "amount_to_prior_mean",
    "amount_to_prior_max", "secs_since_prev_txn",
    "txn_count_1h", "txn_count_24h", "amount_sum_24h",
]

FEATURE_DESCRIPTIONS = {
    "amount": "transaction amount",
    "log_amount": "transaction amount (log scale)",
    "hour": "hour of day",
    "is_night": "transaction made at night (00:00-05:59)",
    "day_of_week": "day of week",
    "is_online": "online (card-not-present) transaction",
    "is_atm": "ATM withdrawal",
    "category_code": "merchant category",
    "is_high_risk_category": "high-risk merchant category",
    "is_foreign": "made outside the customer's home country",
    "is_new_device": "device never seen before for this customer",
    "is_new_country": "first time this customer transacts in this country",
    "cust_prior_txn_count": "number of earlier transactions by this customer",
    "cust_prior_mean_amount": "customer's average earlier amount",
    "amount_to_prior_mean": "amount compared with the customer's usual amount",
    "amount_to_prior_max": "amount compared with the customer's largest earlier amount",
    "secs_since_prev_txn": "seconds since the customer's previous transaction",
    "txn_count_1h": "number of transactions by this customer in the last hour",
    "txn_count_24h": "number of transactions by this customer in the last 24 hours",
    "amount_sum_24h": "total amount spent by this customer in the last 24 hours",
}

REQUIRED_INPUT_COLUMNS = (
    "transaction_id", "timestamp", "customer_id", "merchant_category", "channel",
    "country", "customer_home_country", "device_id", "amount",
)

_CATEGORY_CODE = {c: i for i, c in enumerate((*MERCHANT_CATEGORIES, CASH_CATEGORY))}
KNOWN_CATEGORIES = frozenset(_CATEGORY_CODE)
CATEGORY_NAMES = tuple(_CATEGORY_CODE)  # index == category_code
_HOUR, _DAY = 3600, 86_400
_BAND = 10**10  # separates customers inside one sorted int64 key (secs < 1e10)


def _check_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame")
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns for feature building: {missing}")
    if df.columns.duplicated().any():
        raise ValueError("duplicate column names")
    nulls = [c for c in REQUIRED_INPUT_COLUMNS if df[c].isna().any()]
    if nulls:
        raise ValueError(f"null values in columns: {nulls}")
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        raise ValueError("timestamps must be timezone-naive (convert to UTC first)")
    if not pd.api.types.is_numeric_dtype(df["amount"]) or not np.isfinite(
        df["amount"].to_numpy(dtype=float)
    ).all():
        raise ValueError("'amount' must be finite and numeric")
    if (df["amount"] <= 0).any():
        raise ValueError("'amount' must be > 0")
    if df["transaction_id"].duplicated().any():
        raise ValueError("duplicate transaction_id")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of features per input row (same index, same order).

    Input row order does not matter; results are identical for any permutation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame")
    if len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in FEATURE_COLUMNS}, index=df.index)
    _check_input(df)

    n = len(df)
    ts_ns = df["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
    secs = ts_ns // 10**9
    cust_code = pd.factorize(df["customer_id"])[0].astype("int64")
    order = np.lexsort((df["transaction_id"].to_numpy(), ts_ns, cust_code))  # cust, time, id

    d = df.iloc[order]
    cust, t = cust_code[order], secs[order]
    amt = d["amount"].to_numpy(dtype=float)
    channel = d["channel"].to_numpy()
    country, home = d["country"].to_numpy(), d["customer_home_country"].to_numpy()

    new_grp = np.r_[True, cust[1:] != cust[:-1]]
    starts = np.flatnonzero(new_grp)
    gid = np.cumsum(new_grp) - 1
    pos = np.arange(n) - starts[gid]  # number of earlier transactions of this customer

    csum = np.r_[0.0, np.cumsum(amt)]  # csum[i] = sum(amt[:i])
    prior_sum = csum[:-1] - csum[starts][gid]
    has_hist = pos > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_mean = np.where(has_hist, prior_sum / np.maximum(pos, 1), np.nan)
    run_max = pd.Series(amt).groupby(gid).cummax()
    prior_max = run_max.groupby(gid).shift(1).to_numpy()

    prev_t = np.r_[0, t[:-1]]
    since_prev = np.where(has_hist, t - prev_t, np.nan)

    key = cust * _BAND + t  # non-decreasing: sorted by customer, then time
    hi = np.searchsorted(key, key, side="left")  # rows strictly earlier in time
    lo_1h = np.searchsorted(key, key - _HOUR, side="left")
    lo_24h = np.searchsorted(key, key - _DAY, side="left")

    is_atm = channel == "atm"
    dev_seen = pd.DataFrame({"c": cust, "v": d["device_id"].to_numpy()}).duplicated().to_numpy()
    ctry_seen = pd.DataFrame({"c": cust, "v": country}).duplicated().to_numpy()
    foreign = country != home
    hours = d["timestamp"].dt.hour.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            "amount": amt,
            "log_amount": np.log1p(amt),
            "hour": hours.astype(float),
            "is_night": (hours < 6).astype(float),
            "day_of_week": d["timestamp"].dt.day_of_week.to_numpy().astype(float),
            "is_online": (channel == "online").astype(float),
            "is_atm": is_atm.astype(float),
            "category_code": d["merchant_category"].map(_CATEGORY_CODE).to_numpy(dtype=float),
            "is_high_risk_category": d["merchant_category"].isin(HIGH_RISK_CATEGORIES)
            .to_numpy().astype(float),
            "is_foreign": foreign.astype(float),
            "is_new_device": ((~dev_seen) & (~is_atm)).astype(float),
            "is_new_country": ((~ctry_seen) & foreign).astype(float),
            "cust_prior_txn_count": pos.astype(float),
            "cust_prior_mean_amount": prior_mean,
            "amount_to_prior_mean": amt / prior_mean,
            "amount_to_prior_max": amt / prior_max,
            "secs_since_prev_txn": since_prev.astype(float),
            "txn_count_1h": (hi - lo_1h).astype(float),
            "txn_count_24h": (hi - lo_24h).astype(float),
            "amount_sum_24h": csum[hi] - csum[lo_24h],
        }
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)  # undo the sort
    out = pd.DataFrame({k: np.asarray(feats[k], dtype=float)[inv] for k in FEATURE_COLUMNS})
    out.index = df.index
    return out
