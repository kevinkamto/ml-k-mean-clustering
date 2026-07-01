"""Central configuration for the clustering pipeline.

Keeping paths and tunables in one place avoids hardcoded values scattered
across modules and makes the pipeline reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.schema import ProdCol

# --- Reproducibility ---------------------------------------------------------
# A single seed used for every stochastic step (PCA, K-Means, ...).
RANDOM_STATE: int = 42

# --- Analysis scope -----------------------------------------------------------
# The dataset folder is 2025 receipts, but a handful of stray files carry the
# prior year's trailing days (e.g. a 27-12-2024 receipt logged alongside
# 2025). The analysis is scoped to a single trading year, so anything outside
# it is dropped during cleaning (see preprocessing.clean_transactions).
ANALYSIS_YEAR: int = 2025

# --- Paths -------------------------------------------------------------------
# Project root is the parent of the ``src`` package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Dataset location. Override with the DATASET_DIR env var (e.g. on Colab, point
# it at a Google Drive folder); defaults to the in-repo dataset/ folder.
_DATASET_ENV = os.environ.get("DATASET_DIR")
RAW_DATA_DIR: Path = (
    Path(_DATASET_ENV).expanduser()
    if _DATASET_ENV
    else PROJECT_ROOT / "dataset" / "struk penjualan 2025"
)
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

# Excise (cukai) goods such as cigarettes have extreme price and revenue and
# distort the general segmentation. When enabled, they are clustered in a
# separate pass so the non-excise segments stay clean.
SEPARATE_EXCISE: bool = True

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
