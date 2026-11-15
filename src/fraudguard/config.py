"""Central constants. Keep magic values here so tests and code agree."""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CHANNELS = ("pos", "online", "atm")
COUNTRIES = ("US", "GB", "DE", "FR", "CA", "TR", "AE", "BR", "IN", "NG")
MERCHANT_CATEGORIES = (
    "grocery", "restaurants", "fuel", "retail", "utilities",
    "travel", "electronics", "gift_cards", "jewelry",
)
CASH_CATEGORY = "cash_withdrawal"
HIGH_RISK_CATEGORIES = ("electronics", "gift_cards", "travel", "jewelry")

TARGET_COLUMN = "is_fraud"
PATTERN_COLUMN = "fraud_pattern"  # only for analysis, NEVER a model feature (leakage)

# Canonical internal schema. Real-world datasets get an adapter that maps to this.
REQUIRED_COLUMNS = (
    "transaction_id",
    "timestamp",
    "customer_id",
    "merchant_id",
    "merchant_category",
    "channel",
    "country",
    "customer_home_country",
    "device_id",
    "amount",
    TARGET_COLUMN,
)
STRING_COLUMNS = (
    "customer_id", "merchant_id", "merchant_category",
    "channel", "country", "customer_home_country", "device_id",
)
