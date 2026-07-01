"""Phase 2: Data cleaning.

Take the raw parsed transactions and enforce the data-quality guarantees
required by SPEC.md:

* remove duplicate product lines,
* drop rows with unparseable datetimes or missing keys,
* validate data types,
* validate quantities (must be positive),
* validate prices (no negative unit prices or line totals).

The cleaning is deterministic and returns a small report describing what
was removed, which keeps the pipeline auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src import config
from src.schema import TxnCol

# Columns whose presence/values are mandatory for a usable transaction line.
_REQUIRED_COLUMNS = [
    TxnCol.TRANSACTION_UID,
    TxnCol.TRANSACTION_DATETIME,
    TxnCol.PRODUCT_CODE,
    TxnCol.PRODUCT_NAME,
    TxnCol.QUANTITY,
    TxnCol.UNIT_PRICE,
    TxnCol.LINE_TOTAL,
]


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a boolean-like column to real booleans.

    Handles both native booleans (the in-memory path) and the string values
    (``"True"`` / ``"False"``) produced by a CSV round-trip, where a naive
    ``astype(bool)`` would wrongly treat every non-empty string as ``True``.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype("string").str.strip().str.lower().isin(("true", "1", "yes"))


@dataclass
class CleaningReport:
    """Counts of rows removed at each cleaning step, for auditing."""

    rows_in: int = 0
    duplicates_removed: int = 0
    missing_removed: int = 0
    invalid_quantity_removed: int = 0
    invalid_price_removed: int = 0
    out_of_scope_year_removed: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Cleaning: {self.rows_in:,} -> {self.rows_out:,} rows "
            f"(dups -{self.duplicates_removed:,}, "
            f"missing -{self.missing_removed:,}, "
            f"bad qty -{self.invalid_quantity_removed:,}, "
            f"bad price -{self.invalid_price_removed:,}, "
            f"out-of-scope year -{self.out_of_scope_year_removed:,})"
        )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw transactions frame and report what was removed."""
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # 1) Validate data types up front so later filters behave predictably.
    df[TxnCol.PRODUCT_CODE] = df[TxnCol.PRODUCT_CODE].astype("string")
    df[TxnCol.PRODUCT_NAME] = df[TxnCol.PRODUCT_NAME].astype("string").str.strip()
    df[TxnCol.TRANSACTION_DATETIME] = pd.to_datetime(
        df[TxnCol.TRANSACTION_DATETIME], errors="coerce"
    )
    for col in (
        TxnCol.QUANTITY,
        TxnCol.UNIT_PRICE,
        TxnCol.LINE_TOTAL,
        TxnCol.LINE_DISCOUNT,
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) Remove exact duplicate product lines.
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # 3) Drop rows with missing required fields (incl. unparseable datetimes).
    before = len(df)
    df = df.dropna(subset=_REQUIRED_COLUMNS)
    df = df[df[TxnCol.PRODUCT_NAME].str.len() > 0]
    report.missing_removed = before - len(df)

    # 4) Validate quantities: must be a positive whole number.
    before = len(df)
    df = df[df[TxnCol.QUANTITY] > 0]
    report.invalid_quantity_removed = before - len(df)

    # 5) Validate prices: no negative unit prices or line totals.
    before = len(df)
    df = df[(df[TxnCol.UNIT_PRICE] >= 0) & (df[TxnCol.LINE_TOTAL] >= 0)]
    report.invalid_price_removed = before - len(df)

    # 6) Restrict to the analysis year: a few stray receipts (e.g. a trailing
    # 2024 file dropped in the 2025 folder) fall outside the scoped trading
    # year and must not influence product-level aggregates.
    before = len(df)
    df = df[df[TxnCol.TRANSACTION_DATETIME].dt.year == config.ANALYSIS_YEAR]
    report.out_of_scope_year_removed = before - len(df)

    # Final type tidy-up now that all rows are valid.
    df[TxnCol.QUANTITY] = df[TxnCol.QUANTITY].astype(int)
    df[TxnCol.IS_EXCISE] = coerce_bool(df[TxnCol.IS_EXCISE])
    df = df.reset_index(drop=True)

    report.rows_out = len(df)
    return df, report


def main() -> None:
    """Read raw transactions, clean them, and write the clean dataset."""
    config.ensure_output_dirs()
    raw = pd.read_csv(config.RAW_TRANSACTIONS_CSV)
    clean, report = clean_transactions(raw)
    clean.to_csv(config.TRANSACTIONS_CSV, index=False, encoding="utf-8")
    logger.info(report.summary())
    logger.info("Wrote {}", config.TRANSACTIONS_CSV)


if __name__ == "__main__":
    main()
