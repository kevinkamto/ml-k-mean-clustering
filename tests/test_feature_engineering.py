"""Tests for product aggregation and feature engineering."""

from __future__ import annotations

import pandas as pd

from src import config, feature_engineering
from src.schema import ProdCol, TxnCol


def test_one_row_per_product(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    assert len(products) == transactions_df[ProdCol.PRODUCT_CODE].nunique()


def test_aggregates_are_correct(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    p1 = products[products[ProdCol.PRODUCT_CODE] == "P1"].iloc[0]
    # P1 appears in two transactions: qty 5+3, revenue 5000+3000.
    assert p1[ProdCol.TOTAL_QUANTITY_SOLD] == 8
    assert p1[ProdCol.TOTAL_REVENUE] == 8000
    assert p1[ProdCol.TRANSACTION_COUNT] == 2
    assert p1[ProdCol.AVERAGE_PRICE] == 1000  # 8000 / 8
    assert p1[ProdCol.REVENUE_PER_TRANSACTION] == 4000  # 8000 / 2


def test_clustering_features_present_and_finite(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    for feature in config.CLUSTERING_FEATURES:
        assert feature in products.columns
    assert not products[config.CLUSTERING_FEATURES].isna().any().any()


def test_survives_csv_round_trip(transactions_df, tmp_path) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    path = tmp_path / "products.csv"
    products.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    # is_excise must come back usable, not a string that is always truthy.
    assert set(reloaded[ProdCol.IS_EXCISE].unique()) <= {True, False}


def _temporal_transactions() -> pd.DataFrame:
    # P1 sells across two months incl. a weekend; P2 sells once on a weekday.
    rows = [
        ("P1", "2025-01-03", 2),  # Friday
        ("P1", "2025-01-04", 8),  # Saturday (weekend)
        ("P1", "2025-02-10", 5),  # Monday, second month
        ("P2", "2025-01-06", 1),  # Monday only
    ]
    df = pd.DataFrame(rows, columns=[TxnCol.PRODUCT_CODE, "d", TxnCol.QUANTITY])
    df[TxnCol.PRODUCT_NAME] = df[TxnCol.PRODUCT_CODE]
    df[TxnCol.TRANSACTION_DATETIME] = pd.to_datetime(df.pop("d"))
    df[TxnCol.UNIT_PRICE] = 1000
    df[TxnCol.LINE_TOTAL] = df[TxnCol.QUANTITY] * 1000
    df[TxnCol.LINE_DISCOUNT] = 0
    df[TxnCol.IS_EXCISE] = False
    df[TxnCol.TRANSACTION_UID] = [f"u{i}" for i in range(len(df))]
    return df


def test_temporal_features() -> None:
    products = feature_engineering.build_product_features(_temporal_transactions())
    p1 = products[products[ProdCol.PRODUCT_CODE] == "P1"].iloc[0]
    p2 = products[products[ProdCol.PRODUCT_CODE] == "P2"].iloc[0]

    assert p1[ProdCol.ACTIVE_DAYS] == 3
    assert p2[ProdCol.ACTIVE_DAYS] == 1
    # P1 spans two months -> non-zero volatility; P2 single month -> zero.
    assert p1[ProdCol.MONTHLY_CV] > 0
    assert p2[ProdCol.MONTHLY_CV] == 0
    # P1 sold 8 of 15 units on the weekend.
    assert p1[ProdCol.WEEKEND_RATIO] == 8 / 15
    assert p2[ProdCol.WEEKEND_RATIO] == 0
    # Dataset last day is P1's 2025-02-10, so P2 (last 2025-01-06) is dormant.
    assert p2[ProdCol.RECENCY_DAYS] > 0
