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
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.schema import ProdCol, ProfileCol, ScoreCol


@dataclass
class ClusteringResult:
    """Bundle of everything produced by the clustering stage.

    For the segmented (excise-separated) path, ``scores``, ``model``, and
    ``scaler`` refer to the primary (general) group, and ``optimal_k`` is the
    total number of clusters across groups.
    """

    products: pd.DataFrame  # product table with a ``cluster`` column
    scores: pd.DataFrame | None  # per-k elbow and silhouette scores
    profiles: pd.DataFrame  # per-cluster profile table
    optimal_k: int
    model: KMeans | None
    scaler: StandardScaler | None


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
            f"Not enough samples ({n_samples}) to evaluate k in [{k_min}, {k_max}]."
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
                ScoreCol.K: k,
                ScoreCol.INERTIA: float(model.inertia_),
                ScoreCol.SILHOUETTE: float(silhouette_score(scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def select_optimal_k(scores: pd.DataFrame) -> int:
    """Pick the k with the highest silhouette score (Phase 7).

    The elbow/inertia curve is reported alongside for context, but silhouette
    gives a single, objective criterion for an automated pipeline.
    """
    # Use numpy positions to keep the return cleanly typed as int.
    best_pos = int(scores[ScoreCol.SILHOUETTE].to_numpy().argmax())
    return int(scores[ScoreCol.K].to_numpy()[best_pos])


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
        products.groupby(ProdCol.CLUSTER)
        .agg(
            **{
                ProfileCol.N_PRODUCTS: (ProdCol.PRODUCT_CODE, "count"),
                ProfileCol.AVG_QUANTITY_SOLD: (ProdCol.TOTAL_QUANTITY_SOLD, "mean"),
                ProfileCol.AVG_REVENUE: (ProdCol.TOTAL_REVENUE, "mean"),
                ProfileCol.AVG_PRICE: (ProdCol.AVERAGE_PRICE, "mean"),
                ProfileCol.AVG_TRANSACTION_COUNT: (ProdCol.TRANSACTION_COUNT, "mean"),
                ProfileCol.AVG_QTY_PER_TXN: (
                    ProdCol.AVERAGE_QUANTITY_PER_TRANSACTION,
                    "mean",
                ),
                ProfileCol.AVG_REVENUE_PER_TXN: (
                    ProdCol.REVENUE_PER_TRANSACTION,
                    "mean",
                ),
                ProfileCol.TOTAL_REVENUE: (ProdCol.TOTAL_REVENUE, "sum"),
            }
        )
        .reset_index()
    )
    # Share of overall revenue contributed by each cluster.
    profile[ProfileCol.REVENUE_SHARE] = (
        profile[ProfileCol.TOTAL_REVENUE] / profile[ProfileCol.TOTAL_REVENUE].sum()
    )
    return profile.sort_values(ProfileCol.TOTAL_REVENUE, ascending=False)


def run_clustering(products: pd.DataFrame) -> ClusteringResult:
    """Run scaling, k selection, final fit, and profiling end to end."""
    scaled, scaler = scale_features(products)
    scores = evaluate_k(scaled)
    optimal_k = select_optimal_k(scores)
    model, labels = fit_kmeans(scaled, optimal_k)

    clustered = products.copy()
    clustered[ProdCol.CLUSTER] = labels
    profiles = build_cluster_profiles(clustered)

    return ClusteringResult(
        products=clustered,
        scores=scores,
        profiles=profiles,
        optimal_k=optimal_k,
        model=model,
        scaler=scaler,
    )


def _cluster_group(
    subset: pd.DataFrame, label_offset: int, group_name: str
) -> tuple[pd.DataFrame, pd.DataFrame | None, int, KMeans | None, StandardScaler | None]:
    """Cluster one product group, offsetting labels to stay globally unique.

    Groups too small to sweep k (fewer than ``K_MIN + 1`` products) collapse to
    a single cluster.
    """
    out = subset.copy()
    out[ProdCol.SEGMENT_GROUP] = group_name
    n = len(subset)
    if n == 0:
        return out, None, 0, None, None
    if n < config.K_MIN + 1:
        out[ProdCol.CLUSTER] = label_offset
        return out, None, 1, None, None

    scaled, scaler = scale_features(subset)
    scores = evaluate_k(scaled)
    k = select_optimal_k(scores)
    model, labels = fit_kmeans(scaled, k)
    out[ProdCol.CLUSTER] = labels + label_offset
    return out, scores, k, model, scaler


def run_segmented_clustering(products: pd.DataFrame) -> ClusteringResult:
    """Cluster non-excise and excise products in separate passes (Phase 8b).

    Excise goods are scaled and clustered on their own so they do not skew the
    general segmentation. Cluster labels are offset so they remain unique across
    both groups, and a ``segment_group`` column records the origin.
    """
    excise_mask = products[ProdCol.IS_EXCISE].astype(bool)
    general = products[~excise_mask]
    excise = products[excise_mask]

    gen_out, gen_scores, gen_k, gen_model, gen_scaler = _cluster_group(
        general, label_offset=0, group_name="general"
    )
    exc_out, _, exc_k, _, _ = _cluster_group(
        excise, label_offset=gen_k, group_name="excise"
    )

    combined = pd.concat([gen_out, exc_out], ignore_index=True).sort_values(
        ProdCol.TOTAL_REVENUE, ascending=False
    )
    profiles = build_cluster_profiles(combined)
    # Tag each cluster with the group it came from.
    group_by_cluster = combined.groupby(ProdCol.CLUSTER)[ProdCol.SEGMENT_GROUP].first()
    profiles[ProfileCol.SEGMENT_GROUP] = profiles[ProfileCol.CLUSTER].map(
        group_by_cluster
    )

    return ClusteringResult(
        products=combined,
        scores=gen_scores,
        profiles=profiles,
        optimal_k=gen_k + exc_k,
        model=gen_model,
        scaler=gen_scaler,
    )


def main() -> None:
    """Read product features, cluster them, and write the results."""
    config.ensure_output_dirs()
    products = pd.read_csv(config.PRODUCTS_CSV)
    result = (
        run_segmented_clustering(products)
        if config.SEPARATE_EXCISE
        else run_clustering(products)
    )

    result.products.to_csv(config.CLUSTERED_CSV, index=False, encoding="utf-8")
    result.profiles.to_csv(config.CLUSTER_PROFILE_CSV, index=False, encoding="utf-8")
    if result.scores is not None:
        logger.info("Per-k scores:\n{}", result.scores.to_string(index=False))
    logger.info("Optimal k = {}", result.optimal_k)
    logger.info("Wrote {}", config.CLUSTERED_CSV)
    logger.info("Wrote {}", config.CLUSTER_PROFILE_CSV)


if __name__ == "__main__":
    main()
