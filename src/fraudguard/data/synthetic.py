"""Synthetic card-transaction generator with realistic fraud patterns.

Why synthetic data? Real transaction data is confidential. This generator lets the
whole pipeline (features, training, API, tests) be developed and verified end to end,
with a known ground truth. Final model claims should ALSO be checked on a public
real dataset (see docs/PROBLEM.md) because synthetic data can only be as hard as
the patterns we wrote into it.

Fraud patterns modelled:
  * account_takeover - large online purchases, often abroad, at night, new device
  * card_testing     - bursts of tiny online purchases within minutes
  * atm_cashout      - large ATM withdrawals, usually abroad
  * stealth_takeover - fraud that mimics the victim's normal behaviour (home country,
                       daytime, own device, only moderately larger amounts). Hard to catch
                       on purpose, so model metrics are not unrealistically perfect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import (
    CASH_CATEGORY,
    COUNTRIES,
    HIGH_RISK_CATEGORIES,
    MERCHANT_CATEGORIES,
    PATTERN_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
)

FRAUD_PATTERNS = ("account_takeover", "card_testing", "atm_cashout", "stealth_takeover")

_SECONDS_PER_DAY = 86_400
_N_ATMS = 50
_PATTERN_WEIGHTS = (0.35, 0.30, 0.20, 0.15)  # share of fraud rows, order of FRAUD_PATTERNS

_HOME_COUNTRY_P = np.array([0.35, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05, 0.06, 0.06, 0.03])
_CATEGORY_P = np.array([0.25, 0.20, 0.10, 0.20, 0.05, 0.06, 0.06, 0.03, 0.05])
_LEGIT_CHANNEL_P = np.array([0.55, 0.35, 0.10])  # pos, online, atm
_LEGIT_ATM_AMOUNTS = np.array([20, 40, 60, 80, 100, 200, 300])
_LEGIT_ATM_P = np.array([0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])
_CASHOUT_AMOUNTS = np.array([200, 300, 400, 500, 600, 800, 1000])

_COUNTRIES = np.array(COUNTRIES)
_CATEGORIES = np.array(MERCHANT_CATEGORIES)


# ----------------------------------------------------------------------------- helpers
def _ids(prefix: str, n: int, width: int) -> np.ndarray:
    return np.array([f"{prefix}{i:0{width}d}" for i in range(n)])


def _new_devices(rng: np.random.Generator, n: int) -> np.ndarray:
    """Device ids never seen in a customer's normal history."""
    return np.char.add("X", np.char.zfill(rng.integers(0, 10**8, n).astype(str), 8))


def _country_idx(
    rng: np.random.Generator, home_idx: np.ndarray, p_foreign: float
) -> np.ndarray:
    """Home country, or (with prob p_foreign) a country guaranteed to differ from home."""
    n_c = len(COUNTRIES)
    foreign = (home_idx + rng.integers(1, n_c, home_idx.size)) % n_c
    return np.where(rng.random(home_idx.size) < p_foreign, foreign, home_idx)


def _timestamps(start: pd.Timestamp, seconds: np.ndarray) -> pd.Series:
    td = pd.to_timedelta(np.asarray(seconds, dtype="int64"), unit="s")
    return pd.Series(start + td)


def _night_hours(rng: np.random.Generator, n: int, p_night: float) -> np.ndarray:
    return np.where(rng.random(n) < p_night, rng.integers(0, 6, n), rng.integers(0, 24, n))


def _make_customers(rng: np.random.Generator, n: int) -> dict:
    activity = rng.gamma(shape=1.5, scale=1.0, size=n) + 0.05  # a few very active customers
    return {
        "n": n,
        "id": _ids("C", n, 6),
        "home": rng.choice(len(COUNTRIES), size=n, p=_HOME_COUNTRY_P),
        "mu": rng.normal(3.5, 0.5, n),  # median spend ~ $33, varies per customer
        "sigma": rng.uniform(0.4, 0.8, n),
        "peak_hour": rng.integers(10, 21, n),
        "weight": activity / activity.sum(),
    }


def _make_merchants(rng: np.random.Generator, n: int) -> dict:
    cat = rng.choice(len(MERCHANT_CATEGORIES), size=n, p=_CATEGORY_P)
    k = min(n, len(MERCHANT_CATEGORIES))
    cat[:k] = np.arange(k)  # every category exists whenever n is large enough
    risky = np.flatnonzero(np.isin(_CATEGORIES[cat], HIGH_RISK_CATEGORIES))
    return {
        "n": n,
        "id": _ids("M", n, 5),
        "cat": cat,
        "risky": risky if risky.size else np.arange(n),
    }


def _frame(
    *, ts, cust, cidx, merchant_id, category, channel, country_idx, amount, device,
    is_fraud: int, pattern: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ts.to_numpy(),
            "customer_id": cust["id"][cidx],
            "merchant_id": merchant_id,
            "merchant_category": category,
            "channel": channel,
            "country": _COUNTRIES[country_idx],
            "customer_home_country": _COUNTRIES[cust["home"][cidx]],
            "device_id": device,
            "amount": np.round(amount, 2),
            TARGET_COLUMN: np.full(len(cidx), is_fraud, dtype=np.int64),
            PATTERN_COLUMN: pattern,
        }
    )


# ----------------------------------------------------------------------------- legit
def _make_legit(rng, n, cust, merch, start, n_days) -> pd.DataFrame:
    cidx = rng.choice(cust["n"], size=n, p=cust["weight"])
    is_atm = rng.choice(3, size=n, p=_LEGIT_CHANNEL_P) == 2
    channel = np.where(
        is_atm, "atm", np.where(rng.random(n) < _LEGIT_CHANNEL_P[1] / (1 - _LEGIT_CHANNEL_P[2]),
                                "online", "pos")
    )
    midx = rng.integers(0, merch["n"], n)
    atm_ids = _ids("A", _N_ATMS, 3)[rng.integers(0, _N_ATMS, n)]
    merchant_id = np.where(is_atm, atm_ids, merch["id"][midx])
    category = np.where(is_atm, CASH_CATEGORY, _CATEGORIES[merch["cat"][midx]])

    spend = np.exp(rng.normal(cust["mu"][cidx], cust["sigma"][cidx]))
    atm_amt = rng.choice(_LEGIT_ATM_AMOUNTS, size=n, p=_LEGIT_ATM_P)
    amount = np.clip(np.where(is_atm, atm_amt, spend), 0.5, 20_000)

    hour = np.rint(rng.normal(cust["peak_hour"][cidx], 3.5)).astype(int) % 24
    seconds = (
        rng.integers(0, n_days, n) * _SECONDS_PER_DAY + hour * 3600 + rng.integers(0, 3600, n)
    )
    secondary = rng.random(n) < 0.10  # people own more than one phone/browser
    device = np.where(
        is_atm, merchant_id, np.char.add(cust["id"][cidx], np.where(secondary, "-1", "-0"))
    )
    return _frame(
        ts=_timestamps(start, seconds), cust=cust, cidx=cidx, merchant_id=merchant_id,
        category=category, channel=channel,
        country_idx=_country_idx(rng, cust["home"][cidx], 0.05),
        amount=amount, device=device, is_fraud=0, pattern="legit",
    )


# ----------------------------------------------------------------------------- fraud
def _account_takeover(rng, n, cust, merch, start, n_days) -> pd.DataFrame:
    cidx = rng.choice(cust["n"], size=n, p=cust["weight"])
    midx = rng.choice(merch["risky"], size=n)
    seconds = (
        rng.integers(0, n_days, n) * _SECONDS_PER_DAY
        + _night_hours(rng, n, 0.70) * 3600 + rng.integers(0, 3600, n)
    )
    own_device = np.char.add(cust["id"][cidx], "-0")  # session hijack keeps the victim's device
    device = np.where(rng.random(n) < 0.75, _new_devices(rng, n), own_device)
    return _frame(
        ts=_timestamps(start, seconds), cust=cust, cidx=cidx, merchant_id=merch["id"][midx],
        category=_CATEGORIES[merch["cat"][midx]], channel=np.full(n, "online"),
        country_idx=_country_idx(rng, cust["home"][cidx], 0.80),
        amount=np.clip(np.exp(rng.normal(6.0, 0.6, n)), 50, 20_000),
        device=device, is_fraud=1, pattern="account_takeover",
    )


def _card_testing(rng, n, cust, merch, start, n_days) -> pd.DataFrame:
    sizes, total = [], 0
    while total < n:  # bursts of 3-8 tiny purchases; last burst is truncated to hit n exactly
        size = min(int(rng.integers(3, 9)), n - total)
        sizes.append(size)
        total += size
    sizes = np.array(sizes)
    n_ev = sizes.size
    ev = np.repeat(np.arange(n_ev), sizes)

    ev_cust = rng.choice(cust["n"], size=n_ev, p=cust["weight"])
    ev_start = rng.integers(0, n_days * _SECONDS_PER_DAY - 3600, n_ev)  # burst fits in range
    ev_country = _country_idx(rng, cust["home"][ev_cust], 0.50)
    ev_device = _new_devices(rng, n_ev)

    gaps = rng.integers(5, 120, n)
    first = np.cumsum(sizes) - sizes
    gaps[first] = 0
    csum = np.cumsum(gaps)
    offset = csum - np.repeat(csum[first], sizes)  # seconds since the burst started

    midx = rng.choice(merch["risky"], size=n)
    return _frame(
        ts=_timestamps(start, ev_start[ev] + offset), cust=cust, cidx=ev_cust[ev],
        merchant_id=merch["id"][midx], category=_CATEGORIES[merch["cat"][midx]],
        channel=np.full(n, "online"), country_idx=ev_country[ev],
        amount=rng.uniform(0.5, 5.0, n), device=ev_device[ev],
        is_fraud=1, pattern="card_testing",
    )


def _stealth_takeover(rng, n, cust, merch, start, n_days) -> pd.DataFrame:
    cidx = rng.choice(cust["n"], size=n, p=cust["weight"])
    midx = rng.integers(0, merch["n"], n)
    hour = np.rint(rng.normal(cust["peak_hour"][cidx], 3.5)).astype(int) % 24
    seconds = (
        rng.integers(0, n_days, n) * _SECONDS_PER_DAY + hour * 3600 + rng.integers(0, 3600, n)
    )
    amount = np.clip(np.exp(rng.normal(cust["mu"][cidx] + 1.0, cust["sigma"][cidx])), 1, 20_000)
    return _frame(
        ts=_timestamps(start, seconds), cust=cust, cidx=cidx, merchant_id=merch["id"][midx],
        category=_CATEGORIES[merch["cat"][midx]], channel=np.full(n, "online"),
        country_idx=_country_idx(rng, cust["home"][cidx], 0.05), amount=amount,
        device=np.char.add(cust["id"][cidx], "-0"), is_fraud=1, pattern="stealth_takeover",
    )


def _atm_cashout(rng, n, cust, merch, start, n_days) -> pd.DataFrame:
    cidx = rng.choice(cust["n"], size=n, p=cust["weight"])
    atm = _ids("A", _N_ATMS, 3)[rng.integers(0, _N_ATMS, n)]
    seconds = (
        rng.integers(0, n_days, n) * _SECONDS_PER_DAY
        + _night_hours(rng, n, 0.60) * 3600 + rng.integers(0, 3600, n)
    )
    return _frame(
        ts=_timestamps(start, seconds), cust=cust, cidx=cidx, merchant_id=atm,
        category=np.full(n, CASH_CATEGORY), channel=np.full(n, "atm"),
        country_idx=_country_idx(rng, cust["home"][cidx], 0.90),
        amount=rng.choice(_CASHOUT_AMOUNTS, size=n), device=atm,
        is_fraud=1, pattern="atm_cashout",
    )


# ----------------------------------------------------------------------------- public
def generate_transactions(
    n_transactions: int = 50_000,
    fraud_rate: float = 0.01,
    n_customers: int = 2_000,
    n_merchants: int = 500,
    n_days: int = 90,
    start_date: str = "2025-01-01",
    seed: int = RANDOM_SEED,
    include_pattern: bool = False,
) -> pd.DataFrame:
    """Generate a time-sorted transaction table with EXACTLY round(n*fraud_rate) frauds.

    Deterministic for a given ``seed``. ``include_pattern=True`` adds the ``fraud_pattern``
    column for analysis; never use it as a model feature (target leakage).
    """
    if not isinstance(n_transactions, (int, np.integer)) or n_transactions < 2:
        raise ValueError("n_transactions must be an integer >= 2")
    if not (0 < fraud_rate < 1):
        raise ValueError("fraud_rate must be strictly between 0 and 1")
    for name, val in (("n_customers", n_customers), ("n_merchants", n_merchants),
                      ("n_days", n_days)):
        if not isinstance(val, (int, np.integer)) or val < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    n_fraud = int(round(n_transactions * fraud_rate))
    if n_fraud < 1:
        raise ValueError(
            f"fraud_rate={fraud_rate} with n_transactions={n_transactions} yields 0 frauds; "
            "increase either value"
        )
    if n_fraud >= n_transactions:
        raise ValueError("fraud_rate too high: no legitimate transactions would remain")
    try:
        start = pd.Timestamp(start_date)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid start_date: {start_date!r}") from exc
    if pd.isna(start):
        raise ValueError(f"invalid start_date: {start_date!r}")

    rng = np.random.default_rng(seed)
    cust = _make_customers(rng, int(n_customers))
    merch = _make_merchants(rng, int(n_merchants))

    raw = np.array(_PATTERN_WEIGHTS) * n_fraud  # largest-remainder allocation: exact total
    counts = np.floor(raw).astype(int)
    extra = n_fraud - int(counts.sum())
    counts[np.argsort(-(raw - counts), kind="stable")[:extra]] += 1

    parts = [_make_legit(rng, n_transactions - n_fraud, cust, merch, start, int(n_days))]
    for fn, count in zip(
        (_account_takeover, _card_testing, _atm_cashout, _stealth_takeover), counts, strict=True
    ):
        if count > 0:
            parts.append(fn(rng, int(count), cust, merch, start, int(n_days)))

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1, dtype=np.int64))
    cols = [c for c in df.columns if c not in (TARGET_COLUMN, PATTERN_COLUMN)] + [TARGET_COLUMN]
    if include_pattern:
        cols.append(PATTERN_COLUMN)
    return df[cols]
