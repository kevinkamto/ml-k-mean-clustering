"""Phases 6 to 8 and 10: Scaling, optimal k, K-Means, and profiling.

* Phase 6 standardises the numerical features with ``StandardScaler``.
* Phase 7 sweeps ``k`` from ``K_MIN`` to ``K_MAX`` and records the elbow
  (within-cluster sum of squares / inertia) and the silhouette score.
* Phase 8 fits the final ``KMeans`` model with the selected ``k``.
* Phase 10 builds a per-cluster profile table for business interpretation.

All stochastic steps use ``config.RANDOM_STATE`` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class ClusteringResult:
    """Bundle of everything produced by the clustering stage."""

    products: pd.DataFrame  # product table with a ``cluster`` column
    scores: pd.DataFrame  # per-k elbow and silhouette scores
    profiles: pd.DataFrame  # per-cluster profile table
    optimal_k: int
    model: KMeans
    scaler: StandardScaler


def scale_features(
    products: pd.DataFrame,
    features: list[str] | None = None,
    log_transform: bool | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the clustering features (Phase 6).

    When ``log_transform`` is enabled (the default, from ``config``) a
    ``log1p`` transform is applied first to tame the strong right skew of
    retail sales data before ``StandardScaler``. Returns the scaled matrix and
    the fitted scaler. Raises if any feature contains missing or non-finite
    values, satisfying the SPEC acceptance criteria for a clean matrix.
    """
    features = features or config.CLUSTERING_FEATURES
    if log_transform is None:
        log_transform = config.LOG_TRANSFORM_FEATURES

    matrix = products[features].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "Clustering features contain NaN or infinite values; "
            "check feature engineering before scaling."
        )
    if log_transform:
        if (matrix < 0).any():
            raise ValueError(
                "log1p transform requires non-negative features; "
                "found negative values in the clustering features."
            )
        matrix = np.log1p(matrix)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def evaluate_k(
    scaled: np.ndarray,
    k_min: int = config.K_MIN,
    k_max: int = config.K_MAX,
) -> pd.DataFrame:
    """Compute elbow (inertia) and silhouette score for each candidate k.

    The upper bound is clipped so that ``k`` never exceeds ``n_samples - 1``,
    which would make the silhouette score undefined.
    """
    n_samples = scaled.shape[0]
    upper = min(k_max, n_samples - 1)
    if upper < k_min:
        raise ValueError(
            f"Not enough samples ({n_samples}) to evaluate k in "
            f"[{k_min}, {k_max}]."
        )

    rows: list[dict] = []
    for k in range(k_min, upper + 1):
        model = KMeans(
            n_clusters=k,
            random_state=config.RANDOM_STATE,
            n_init=config.KMEANS_N_INIT,
        )
        labels = model.fit_predict(scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def select_optimal_k(scores: pd.DataFrame) -> int:
    """Pick the k with the highest silhouette score (Phase 7).

    The elbow/inertia curve is reported alongside for context, but silhouette
    gives a single, objective criterion for an automated pipeline.
    """
    best_row = scores.loc[scores["silhouette"].idxmax()]
    return int(best_row["k"])


def fit_kmeans(scaled: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """Fit the final K-Means model (Phase 8)."""
    model = KMeans(
        n_clusters=k,
        random_state=config.RANDOM_STATE,
        n_init=config.KMEANS_N_INIT,
    )
    labels = model.fit_predict(scaled)
    return model, labels


def build_cluster_profiles(products: pd.DataFrame) -> pd.DataFrame:
    """Summarise each cluster for business interpretation (Phase 10)."""
    profile = (
        products.groupby("cluster")
        .agg(
            n_products=("product_code", "count"),
            avg_quantity_sold=("total_quantity_sold", "mean"),
            avg_revenue=("total_revenue", "mean"),
            avg_price=("average_price", "mean"),
            avg_transaction_count=("transaction_count", "mean"),
            avg_qty_per_txn=("average_quantity_per_transaction", "mean"),
            avg_revenue_per_txn=("revenue_per_transaction", "mean"),
            total_revenue=("total_revenue", "sum"),
        )
        .reset_index()
    )
    # Share of overall revenue contributed by each cluster.
    profile["revenue_share"] = (
        profile["total_revenue"] / profile["total_revenue"].sum()
    )
    return profile.sort_values("total_revenue", ascending=False)


def run_clustering(products: pd.DataFrame) -> ClusteringResult:
    """Run scaling, k selection, final fit, and profiling end to end."""
    scaled, scaler = scale_features(products)
    scores = evaluate_k(scaled)
    optimal_k = select_optimal_k(scores)
    model, labels = fit_kmeans(scaled, optimal_k)

    clustered = products.copy()
    clustered["cluster"] = labels
    profiles = build_cluster_profiles(clustered)

    return ClusteringResult(
        products=clustered,
        scores=scores,
        profiles=profiles,
        optimal_k=optimal_k,
        model=model,
        scaler=scaler,
    )


def main() -> None:
    """Read product features, cluster them, and write the results."""
    config.ensure_output_dirs()
    products = pd.read_csv(config.PRODUCTS_CSV)
    result = run_clustering(products)

    result.products.to_csv(config.CLUSTERED_CSV, index=False, encoding="utf-8")
    result.profiles.to_csv(
        config.CLUSTER_PROFILE_CSV, index=False, encoding="utf-8"
    )
    print(result.scores.to_string(index=False))
    print(f"\nOptimal k = {result.optimal_k}")
    print(f"Wrote {config.CLUSTERED_CSV}")
    print(f"Wrote {config.CLUSTER_PROFILE_CSV}")


if __name__ == "__main__":
    main()
