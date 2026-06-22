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

from . import config

# Columns whose presence/values are mandatory for a usable transaction line.
_REQUIRED_COLUMNS = [
    "transaction_uid",
    "transaction_datetime",
    "product_code",
    "product_name",
    "quantity",
    "unit_price",
    "line_total",
]


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a boolean-like column to real booleans.

    Handles both native booleans (the in-memory path) and the string values
    (``"True"`` / ``"False"``) produced by a CSV round-trip, where a naive
    ``astype(bool)`` would wrongly treat every non-empty string as ``True``.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(("true", "1", "yes"))
    )


@dataclass
class CleaningReport:
    """Counts of rows removed at each cleaning step, for auditing."""

    rows_in: int = 0
    duplicates_removed: int = 0
    missing_removed: int = 0
    invalid_quantity_removed: int = 0
    invalid_price_removed: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Cleaning: {self.rows_in:,} -> {self.rows_out:,} rows "
            f"(dups -{self.duplicates_removed:,}, "
            f"missing -{self.missing_removed:,}, "
            f"bad qty -{self.invalid_quantity_removed:,}, "
            f"bad price -{self.invalid_price_removed:,})"
        )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw transactions frame and report what was removed."""
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # 1) Validate data types up front so later filters behave predictably.
    df["product_code"] = df["product_code"].astype("string")
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["transaction_datetime"] = pd.to_datetime(
        df["transaction_datetime"], errors="coerce"
    )
    for col in ("quantity", "unit_price", "line_total", "line_discount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) Remove exact duplicate product lines.
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # 3) Drop rows with missing required fields (incl. unparseable datetimes).
    before = len(df)
    df = df.dropna(subset=_REQUIRED_COLUMNS)
    df = df[df["product_name"].str.len() > 0]
    report.missing_removed = before - len(df)

    # 4) Validate quantities: must be a positive whole number.
    before = len(df)
    df = df[df["quantity"] > 0]
    report.invalid_quantity_removed = before - len(df)

    # 5) Validate prices: no negative unit prices or line totals.
    before = len(df)
    df = df[(df["unit_price"] >= 0) & (df["line_total"] >= 0)]
    report.invalid_price_removed = before - len(df)

    # Final type tidy-up now that all rows are valid.
    df["quantity"] = df["quantity"].astype(int)
    df["is_excise"] = coerce_bool(df["is_excise"])
    df = df.reset_index(drop=True)

    report.rows_out = len(df)
    return df, report


def main() -> None:
    """Read raw transactions, clean them, and write the clean dataset."""
    config.ensure_output_dirs()
    raw = pd.read_csv(config.RAW_TRANSACTIONS_CSV)
    clean, report = clean_transactions(raw)
    clean.to_csv(config.TRANSACTIONS_CSV, index=False, encoding="utf-8")
    print(report.summary())
    print(f"Wrote {config.TRANSACTIONS_CSV}")


if __name__ == "__main__":
    main()
