"""Schema and sanity validation. Bad data must fail loudly, before it reaches a model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from fraudguard.config import CHANNELS, REQUIRED_COLUMNS, STRING_COLUMNS, TARGET_COLUMN


class DataValidationError(ValueError):
    """Raised when a transaction table violates the schema."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


def _preview(values, limit: int = 5) -> str:
    vals = list(values)
    shown = ", ".join(repr(v) for v in vals[:limit])
    return shown + (f", ... (+{len(vals) - limit} more)" if len(vals) > limit else "")


def validate_transactions(df: pd.DataFrame, *, raise_on_error: bool = True) -> ValidationReport:
    """Validate ``df`` against the canonical schema. Never mutates ``df``.

    All problems are collected (not just the first) so they can be fixed in one pass.
    Extra columns are allowed. Suspicious-but-legal situations become warnings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")

    rep = ValidationReport()
    err, warn = rep.errors, rep.warnings

    if df.columns.duplicated().any():
        err.append(f"duplicate column names: {_preview(df.columns[df.columns.duplicated()])}")
    elif df.empty:
        err.append("table has 0 rows")
    else:
        _check_columns(df, err, warn)

    if raise_on_error:
        rep.raise_if_failed()
    return rep


def _check_columns(df: pd.DataFrame, err: list[str], warn: list[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        err.append(f"missing required columns: {missing}")
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]

    for col in present:
        n_null = int(df[col].isna().sum())
        if n_null:
            err.append(f"column '{col}' has {n_null} null value(s)")

    if "transaction_id" in present:
        s = df["transaction_id"].dropna()
        if not pdt.is_integer_dtype(s):
            err.append("'transaction_id' must be an integer column")
        elif s.duplicated().any():
            err.append(f"'transaction_id' has {int(s.duplicated().sum())} duplicate value(s)")

    if "timestamp" in present and not pdt.is_datetime64_any_dtype(df["timestamp"]):
        err.append("'timestamp' must be a datetime column (parse it before validating)")

    if "amount" in present:
        s = df["amount"].dropna()
        if not pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s):
            err.append("'amount' must be numeric")
        else:
            arr = s.to_numpy(dtype=float)
            n_bad = int((~np.isfinite(arr)).sum())
            if n_bad:
                err.append(f"'amount' has {n_bad} non-finite value(s) (inf)")
            n_nonpos = int((arr[np.isfinite(arr)] <= 0).sum())
            if n_nonpos:
                err.append(f"'amount' has {n_nonpos} value(s) <= 0")

    if TARGET_COLUMN in present:
        s = df[TARGET_COLUMN].dropna()
        if not (pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s)):
            err.append(f"'{TARGET_COLUMN}' must be numeric 0/1")
        elif not s.isin([0, 1]).all():
            bad = sorted(set(s[~s.isin([0, 1])].tolist()))
            err.append(f"'{TARGET_COLUMN}' must contain only 0/1, found {_preview(bad)}")
        elif len(s) == len(df):
            n_fraud = int((s == 1).sum())
            if n_fraud == 0:
                warn.append("no fraud rows: a model cannot be trained or evaluated")
            elif n_fraud == len(df):
                warn.append("every row is fraud: a model cannot be trained or evaluated")
            elif n_fraud / len(df) > 0.5:
                warn.append(f"fraud rate is {n_fraud / len(df):.1%}; unusually high, check labels")

    for col in STRING_COLUMNS:
        if col not in present:
            continue
        s = df[col].dropna()
        if not pdt.is_string_dtype(s):
            err.append(f"'{col}' must be a string column")
        elif s.astype(str).str.strip().eq("").any():
            err.append(f"'{col}' contains blank strings")
        elif col == "channel":
            bad = sorted(set(s) - set(CHANNELS))
            if bad:
                err.append(f"'channel' has unknown values {_preview(bad)}; allowed: {CHANNELS}")
