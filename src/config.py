"""Central configuration for the clustering pipeline.

Keeping paths and tunables in one place avoids hardcoded values scattered
across modules and makes the pipeline reproducible.
"""

from __future__ import annotations

from pathlib import Path

from src.schema import ProdCol

# --- Reproducibility ---------------------------------------------------------
# A single seed used for every stochastic step (PCA, K-Means, ...).
RANDOM_STATE: int = 42

# --- Paths -------------------------------------------------------------------
# Project root is the parent of the ``src`` package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DATA_DIR: Path = PROJECT_ROOT / "dataset" / "struk penjualan 2025"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
EXPORTS_DIR: Path = PROJECT_ROOT / "data" / "exports"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Output files
RAW_TRANSACTIONS_CSV: Path = PROCESSED_DIR / "transactions_raw.csv"
TRANSACTIONS_CSV: Path = PROCESSED_DIR / "transactions.csv"
PRODUCTS_CSV: Path = EXPORTS_DIR / "product_features.csv"
CLUSTERED_CSV: Path = EXPORTS_DIR / "product_clusters.csv"
CLUSTER_PROFILE_CSV: Path = EXPORTS_DIR / "cluster_profiles.csv"

# --- Clustering parameters ---------------------------------------------------
# Inclusive search range for the optimal number of clusters.
K_MIN: int = 2
K_MAX: int = 10

# scikit-learn KMeans configuration (see SPEC.md, Phase 8).
KMEANS_N_INIT: int = 20

# Retail sales features are strongly right-skewed (a few products dominate
# volume and revenue). Applying log1p before standardising stops K-Means from
# simply isolating a handful of outliers and yields interpretable segments.
LOG_TRANSFORM_FEATURES: bool = True

# Features fed to the clustering model (numerical only, see SPEC.md).
# The first six are volume/value features; the last two are temporal features
# that let the model separate steady sellers from spiky / seasonal products.
CLUSTERING_FEATURES: list[str] = [
    ProdCol.TOTAL_QUANTITY_SOLD,
    ProdCol.TOTAL_REVENUE,
    ProdCol.TRANSACTION_COUNT,
    ProdCol.AVERAGE_PRICE,
    ProdCol.AVERAGE_QUANTITY_PER_TRANSACTION,
    ProdCol.REVENUE_PER_TRANSACTION,
    ProdCol.ACTIVE_DAYS,
    ProdCol.MONTHLY_CV,
]


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for directory in (PROCESSED_DIR, EXPORTS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
