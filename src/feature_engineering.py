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

from . import config, preprocessing


def _dominant_name(names: pd.Series) -> str:
    """Return the most frequent product name for a product code.

    The same code can carry slightly different name strings across receipts;
    the modal name is the most representative label.
    """
    mode = names.mode()
    return str(mode.iloc[0]) if not mode.empty else str(names.iloc[0])


def aggregate_products(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction lines into one row per product (Phase 4)."""
    grouped = transactions.groupby("product_code", sort=False)

    products = grouped.agg(
        product_name=("product_name", _dominant_name),
        total_quantity_sold=("quantity", "sum"),
        total_revenue=("line_total", "sum"),
        transaction_count=("transaction_uid", "nunique"),
        total_discount=("line_discount", "sum"),
        discounted_lines=("line_discount", lambda s: int((s > 0).sum())),
        line_count=("line_total", "size"),
        is_excise=("is_excise", "max"),
    ).reset_index()

    # Effective realised price per unit (revenue already net of discounts).
    # quantity is guaranteed positive by the cleaning step, so this is safe.
    products["average_price"] = (
        products["total_revenue"] / products["total_quantity_sold"]
    )
    return products


def engineer_features(
    products: pd.DataFrame, total_days: int
) -> pd.DataFrame:
    """Add the engineered ratio features (Phase 5).

    ``total_days`` is the number of distinct trading days in the dataset and
    is used to express activity as a per-day rate.
    """
    products = products.copy()

    # transaction_count is >= 1 for every aggregated product, so safe denom.
    products["average_quantity_per_transaction"] = (
        products["total_quantity_sold"] / products["transaction_count"]
    )
    products["revenue_per_transaction"] = (
        products["total_revenue"] / products["transaction_count"]
    )

    # Optional descriptive features (used for profiling, not for clustering).
    products["discount_frequency"] = (
        products["discounted_lines"] / products["line_count"]
    )
    products["average_discount"] = (
        products["total_discount"] / products["line_count"]
    )
    days = max(total_days, 1)  # guard against an empty dataset
    products["sales_frequency_per_day"] = products["transaction_count"] / days

    return products


def build_product_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run the full product feature build from clean transactions."""
    transactions = transactions.copy()
    # Robust against a CSV round-trip where is_excise comes back as a string.
    transactions["is_excise"] = preprocessing.coerce_bool(transactions["is_excise"])
    transactions["transaction_datetime"] = pd.to_datetime(
        transactions["transaction_datetime"], errors="coerce"
    )
    total_days = transactions["transaction_datetime"].dt.normalize().nunique()
    products = aggregate_products(transactions)
    products = engineer_features(products, total_days=total_days)

    # Order columns: identifiers, clustering features, then extras.
    ordered = (
        ["product_code", "product_name"]
        + config.CLUSTERING_FEATURES
        + [
            "discount_frequency",
            "average_discount",
            "sales_frequency_per_day",
            "is_excise",
        ]
    )
    return products[ordered].sort_values("total_revenue", ascending=False)


def main() -> None:
    """Read clean transactions and write the product feature table."""
    config.ensure_output_dirs()
    transactions = pd.read_csv(
        config.TRANSACTIONS_CSV, parse_dates=["transaction_datetime"]
    )
    products = build_product_features(transactions)
    products.to_csv(config.PRODUCTS_CSV, index=False, encoding="utf-8")
    print(
        f"Built features for {len(products):,} products -> "
        f"{config.PRODUCTS_CSV}"
    )


if __name__ == "__main__":
    main()
