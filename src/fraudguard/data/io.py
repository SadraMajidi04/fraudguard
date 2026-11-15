"""Safe load/save. Every file that enters or leaves the pipeline is validated."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudguard.data.validation import validate_transactions

_ALLOWED_SUFFIXES = (".csv", ".csv.gz")


def _check_suffix(path: Path) -> None:
    if not str(path).lower().endswith(_ALLOWED_SUFFIXES):
        raise ValueError(f"unsupported file type for {path.name!r}; use one of {_ALLOWED_SUFFIXES}")


def save_transactions(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate, then write ``df`` as CSV (or gzipped CSV). Creates parent folders."""
    path = Path(path)
    _check_suffix(path)
    validate_transactions(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by :func:`save_transactions` and validate it."""
    path = Path(path)
    _check_suffix(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    validate_transactions(df)
    return df
