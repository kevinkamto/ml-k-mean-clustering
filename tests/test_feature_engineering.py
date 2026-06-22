"""Tests for product aggregation and feature engineering."""

from __future__ import annotations

from src import config, feature_engineering
from src.schema import ProdCol


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
    import pandas as pd

    reloaded = pd.read_csv(path)
    # is_excise must come back usable, not a string that is always truthy.
    assert set(reloaded[ProdCol.IS_EXCISE].unique()) <= {True, False}
