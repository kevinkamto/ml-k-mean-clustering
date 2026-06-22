"""Phases 4 and 5: Product-level aggregation and feature engineering.

Collapse the transaction lines into one row per product and derive the
numerical features used for clustering.

Base aggregation (Phase 4)::

    total_quantity_sold   sum of quantity
    total_revenue         sum of line_total (net of discounts)
    transaction_count     number of distinct receipts containing the product
    average_price         effective realised price (revenue / quantity)

Engineered features (Phase 5)::

    average_quantity_per_transaction = total_quantity_sold / transaction_count
    revenue_per_transaction          = total_revenue / transaction_count

Optional descriptive features (not used for clustering, kept for profiling)::

    discount_frequency, average_discount, sales_frequency_per_day
"""

from __future__ import annotations

import pandas as pd

from src import config, preprocessing
from src.schema import ProdCol, TxnCol


def _dominant_name(names: pd.Series) -> str:
    """Return the most frequent product name for a product code.

    The same code can carry slightly different name strings across receipts;
    the modal name is the most representative label.
    """
    mode = names.mode()
    return str(mode.iloc[0]) if not mode.empty else str(names.iloc[0])


def aggregate_products(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction lines into one row per product (Phase 4)."""
    grouped = transactions.groupby(TxnCol.PRODUCT_CODE, sort=False)

    products = grouped.agg(
        **{
            ProdCol.PRODUCT_NAME: (TxnCol.PRODUCT_NAME, _dominant_name),
            ProdCol.TOTAL_QUANTITY_SOLD: (TxnCol.QUANTITY, "sum"),
            ProdCol.TOTAL_REVENUE: (TxnCol.LINE_TOTAL, "sum"),
            ProdCol.TRANSACTION_COUNT: (TxnCol.TRANSACTION_UID, "nunique"),
            ProdCol.TOTAL_DISCOUNT: (TxnCol.LINE_DISCOUNT, "sum"),
            ProdCol.DISCOUNTED_LINES: (
                TxnCol.LINE_DISCOUNT,
                lambda s: int((s > 0).sum()),
            ),
            ProdCol.LINE_COUNT: (TxnCol.LINE_TOTAL, "size"),
            ProdCol.IS_EXCISE: (TxnCol.IS_EXCISE, "max"),
        }
    ).reset_index()

    # Effective realised price per unit (revenue already net of discounts).
    # quantity is guaranteed positive by the cleaning step, so this is safe.
    products[ProdCol.AVERAGE_PRICE] = (
        products[ProdCol.TOTAL_REVENUE] / products[ProdCol.TOTAL_QUANTITY_SOLD]
    )
    return products


def engineer_features(products: pd.DataFrame, total_days: int) -> pd.DataFrame:
    """Add the engineered ratio features (Phase 5).

    ``total_days`` is the number of distinct trading days in the dataset and
    is used to express activity as a per-day rate.
    """
    products = products.copy()

    # transaction_count is >= 1 for every aggregated product, so safe denom.
    products[ProdCol.AVERAGE_QUANTITY_PER_TRANSACTION] = (
        products[ProdCol.TOTAL_QUANTITY_SOLD] / products[ProdCol.TRANSACTION_COUNT]
    )
    products[ProdCol.REVENUE_PER_TRANSACTION] = (
        products[ProdCol.TOTAL_REVENUE] / products[ProdCol.TRANSACTION_COUNT]
    )

    # Optional descriptive features (used for profiling, not for clustering).
    products[ProdCol.DISCOUNT_FREQUENCY] = (
        products[ProdCol.DISCOUNTED_LINES] / products[ProdCol.LINE_COUNT]
    )
    products[ProdCol.AVERAGE_DISCOUNT] = (
        products[ProdCol.TOTAL_DISCOUNT] / products[ProdCol.LINE_COUNT]
    )
    days = max(total_days, 1)  # guard against an empty dataset
    products[ProdCol.SALES_FREQUENCY_PER_DAY] = products[ProdCol.TRANSACTION_COUNT] / days

    return products


def build_product_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run the full product feature build from clean transactions."""
    transactions = transactions.copy()
    # Robust against a CSV round-trip where is_excise comes back as a string.
    transactions[TxnCol.IS_EXCISE] = preprocessing.coerce_bool(
        transactions[TxnCol.IS_EXCISE]
    )
    transactions[TxnCol.TRANSACTION_DATETIME] = pd.to_datetime(
        transactions[TxnCol.TRANSACTION_DATETIME], errors="coerce"
    )
    total_days = transactions[TxnCol.TRANSACTION_DATETIME].dt.normalize().nunique()
    products = aggregate_products(transactions)
    products = engineer_features(products, total_days=total_days)

    # Order columns: identifiers, clustering features, then extras.
    ordered = (
        [ProdCol.PRODUCT_CODE, ProdCol.PRODUCT_NAME]
        + config.CLUSTERING_FEATURES
        + [
            ProdCol.DISCOUNT_FREQUENCY,
            ProdCol.AVERAGE_DISCOUNT,
            ProdCol.SALES_FREQUENCY_PER_DAY,
            ProdCol.IS_EXCISE,
        ]
    )
    return products[ordered].sort_values(ProdCol.TOTAL_REVENUE, ascending=False)


def main() -> None:
    """Read clean transactions and write the product feature table."""
    config.ensure_output_dirs()
    transactions = pd.read_csv(
        config.TRANSACTIONS_CSV, parse_dates=[str(TxnCol.TRANSACTION_DATETIME)]
    )
    products = build_product_features(transactions)
    products.to_csv(config.PRODUCTS_CSV, index=False, encoding="utf-8")
    print(f"Built features for {len(products):,} products -> {config.PRODUCTS_CSV}")


if __name__ == "__main__":
    main()
