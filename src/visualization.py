"""Phases 3 and 9: Exploratory and cluster visualisations.

Every function renders a figure to ``reports/figures/*.png``. A non-interactive
Matplotlib backend is selected so the module is safe to run headless (in the
pipeline or CI) without opening windows or leaking figure state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src import config  # noqa: E402
from src.clustering import scale_features, select_optimal_k  # noqa: E402
from src.schema import ProdCol, ScoreCol  # noqa: E402

sns.set_theme(style="whitegrid")

# Number of products shown in "top N" bar charts.
_TOP_N = 15


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to the figures directory and close it."""
    config.ensure_output_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_products_by_quantity(products: pd.DataFrame) -> Path:
    """Bar chart of the top products by total quantity sold (Phase 3)."""
    top = products.nlargest(_TOP_N, ProdCol.TOTAL_QUANTITY_SOLD)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top,
        y=ProdCol.PRODUCT_NAME,
        x=ProdCol.TOTAL_QUANTITY_SOLD,
        ax=ax,
        color="#4C72B0",
    )
    ax.set_title(f"Top {_TOP_N} products by quantity sold")
    ax.set_xlabel("Total quantity sold")
    ax.set_ylabel("")
    return _save(fig, "eda_top_products_quantity.png")


def plot_top_products_by_revenue(products: pd.DataFrame) -> Path:
    """Bar chart of the top products by total revenue (Phase 3)."""
    top = products.nlargest(_TOP_N, ProdCol.TOTAL_REVENUE)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top,
        y=ProdCol.PRODUCT_NAME,
        x=ProdCol.TOTAL_REVENUE,
        ax=ax,
        color="#55A868",
    )
    ax.set_title(f"Top {_TOP_N} products by revenue")
    ax.set_xlabel("Total revenue (IDR)")
    ax.set_ylabel("")
    return _save(fig, "eda_top_products_revenue.png")


def plot_distributions(products: pd.DataFrame) -> Path:
    """Histograms of quantity and revenue distributions (Phase 3).

    Both are heavily right-skewed, so a log scale is used on the x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Log scale is undefined at zero, so restrict to strictly positive values.
    qty = products.loc[
        products[ProdCol.TOTAL_QUANTITY_SOLD] > 0, ProdCol.TOTAL_QUANTITY_SOLD
    ]
    rev = products.loc[products[ProdCol.TOTAL_REVENUE] > 0, ProdCol.TOTAL_REVENUE]
    sns.histplot(qty, bins=50, ax=axes[0], log_scale=True)
    axes[0].set_title("Distribution of total quantity sold")
    axes[0].set_xlabel("Total quantity sold (log scale)")

    sns.histplot(rev, bins=50, ax=axes[1], log_scale=True)
    axes[1].set_title("Distribution of total revenue")
    axes[1].set_xlabel("Total revenue, IDR (log scale)")
    return _save(fig, "eda_distributions.png")


def plot_feature_boxplots(products: pd.DataFrame) -> Path:
    """Box plots of the standardised clustering features (Phase 3)."""
    scaled, _ = scale_features(products)
    scaled_df = pd.DataFrame(scaled, columns=config.CLUSTERING_FEATURES)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=scaled_df, orient="h", ax=ax)
    ax.set_title("Standardised feature distributions (outlier view)")
    ax.set_xlabel("Standardised value (z-score)")
    return _save(fig, "eda_feature_boxplots.png")


def plot_correlation_heatmap(products: pd.DataFrame) -> Path:
    """Correlation heatmap of the clustering features (Phase 3)."""
    corr = products[config.CLUSTERING_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax
    )
    ax.set_title("Feature correlation heatmap")
    return _save(fig, "eda_correlation_heatmap.png")


def plot_elbow_and_silhouette(scores: pd.DataFrame) -> Path:
    """Elbow (inertia) and silhouette curves over k (Phase 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(scores[ScoreCol.K], scores[ScoreCol.INERTIA], marker="o")
    axes[0].set_title("Elbow method (WCSS)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster SS)")

    axes[1].plot(
        scores[ScoreCol.K], scores[ScoreCol.SILHOUETTE], marker="o", color="#C44E52"
    )
    best_k = select_optimal_k(scores)
    axes[1].axvline(best_k, linestyle="--", color="grey", label=f"best k = {best_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()
    return _save(fig, "kmeans_elbow_silhouette.png")


def plot_pca_clusters(products: pd.DataFrame) -> Path:
    """2-D PCA projection of products coloured by cluster (Phase 9)."""
    if ProdCol.CLUSTER not in products.columns:
        raise ValueError("products must contain a 'cluster' column.")
    scaled, _ = scale_features(products)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(scaled)

    plot_df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    # Treat clusters as ordered categories so the hue is discrete and the
    # palette scales to any number of clusters (segmented runs can exceed 10).
    clusters_sorted = sorted(products[ProdCol.CLUSTER].unique())
    plot_df[ProdCol.CLUSTER] = pd.Categorical(
        products[ProdCol.CLUSTER].to_numpy(), categories=clusters_sorted
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue=ProdCol.CLUSTER,
        palette=sns.color_palette("tab20", len(clusters_sorted)),
        s=40,
        alpha=0.7,
        ax=ax,
    )
    var = pca.explained_variance_ratio_ * 100
    ax.set_title("Product clusters (PCA projection)")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.legend(title="Cluster")
    return _save(fig, "kmeans_pca_clusters.png")


def plot_cluster_sizes(products: pd.DataFrame) -> Path:
    """Bar chart of the number of products per cluster (Phase 9)."""
    sizes = products[ProdCol.CLUSTER].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sizes.index, y=sizes.to_numpy(), ax=ax, color="#8172B3")
    ax.set_title("Cluster size distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of products")
    return _save(fig, "kmeans_cluster_sizes.png")


def generate_all(products: pd.DataFrame, scores: pd.DataFrame | None) -> list[Path]:
    """Render every figure and return the list of written paths."""
    paths = [
        plot_top_products_by_quantity(products),
        plot_top_products_by_revenue(products),
        plot_distributions(products),
        plot_feature_boxplots(products),
        plot_correlation_heatmap(products),
        plot_pca_clusters(products),
        plot_cluster_sizes(products),
    ]
    # The elbow/silhouette plot needs the per-k sweep, which is absent when a
    # group was too small to sweep.
    if scores is not None:
        paths.append(plot_elbow_and_silhouette(scores))
    return paths


def main() -> None:
    """Read clustered products and scores and render all figures."""
    products = pd.read_csv(config.CLUSTERED_CSV)
    # Scores are recomputed cheaply from the clustered features if absent.
    from src.clustering import evaluate_k

    scaled, _ = scale_features(products)
    scores = evaluate_k(scaled)
    paths = generate_all(products, scores)
    print(f"Wrote {len(paths)} figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
