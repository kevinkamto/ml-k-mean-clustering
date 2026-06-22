"""Tests for scaling, k selection, and the clustering driver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clustering, config, feature_engineering
from src.schema import ProdCol, ScoreCol


def test_scale_features_standardised_and_finite(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    scaled, scaler = clustering.scale_features(products)
    assert scaled.shape == (len(products), len(config.CLUSTERING_FEATURES))
    assert np.isfinite(scaled).all()
    # StandardScaler output has ~zero mean per column.
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)


def test_scale_features_rejects_negative_with_log() -> None:
    df = pd.DataFrame({c: [1.0, 2.0] for c in config.CLUSTERING_FEATURES})
    df.loc[0, config.CLUSTERING_FEATURES[0]] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        clustering.scale_features(df, log_transform=True)


def test_evaluate_k_and_select(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    scaled, _ = clustering.scale_features(products)
    scores = clustering.evaluate_k(scaled, k_min=2, k_max=4)
    assert list(scores[ScoreCol.K]) == [2, 3, 4]
    assert scores[ScoreCol.INERTIA].is_monotonic_decreasing
    k = clustering.select_optimal_k(scores)
    assert 2 <= k <= 4


def test_evaluate_k_raises_when_too_few_samples() -> None:
    tiny = np.zeros((2, len(config.CLUSTERING_FEATURES)))
    with pytest.raises(ValueError, match="Not enough samples"):
        clustering.evaluate_k(tiny, k_min=5, k_max=10)


def test_run_clustering_is_reproducible(transactions_df) -> None:
    products = feature_engineering.build_product_features(transactions_df)
    r1 = clustering.run_clustering(products)
    r2 = clustering.run_clustering(products)
    assert r1.optimal_k == r2.optimal_k
    assert list(r1.products[ProdCol.CLUSTER]) == list(r2.products[ProdCol.CLUSTER])
    # Every product receives a cluster label.
    assert r1.products[ProdCol.CLUSTER].notna().all()
    # Revenue share over clusters sums to 1.
    assert r1.profiles["revenue_share"].sum() == pytest.approx(1.0)
