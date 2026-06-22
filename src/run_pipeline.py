"""End-to-end pipeline driver.

Runs every phase in order, entirely in memory, and writes the deliverables:

* ``data/processed/transactions.csv``      clean transaction dataset
* ``data/exports/product_features.csv``    product feature dataset
* ``data/exports/product_clusters.csv``    products with assigned clusters
* ``data/exports/cluster_profiles.csv``    per-cluster profile table
* ``reports/figures/*.png``                EDA and cluster visualisations

Run with::

    uv run python -m src.run_pipeline
"""

from __future__ import annotations

from src import config, feature_engineering, parser, preprocessing, visualization
from src.clustering import run_clustering, run_segmented_clustering
from src.schema import ScoreCol, TxnCol


def main() -> None:
    config.ensure_output_dirs()

    # Phase 1: parse raw receipts.
    raw = parser.parse_all()
    raw.to_csv(config.RAW_TRANSACTIONS_CSV, index=False, encoding="utf-8")
    print(
        f"[1/5] Parsed {raw[TxnCol.SOURCE_FILE].nunique()} files -> "
        f"{len(raw):,} product lines."
    )

    # Phase 2: clean.
    clean, report = preprocessing.clean_transactions(raw)
    clean.to_csv(config.TRANSACTIONS_CSV, index=False, encoding="utf-8")
    print(f"[2/5] {report.summary()}")

    # Phases 4 and 5: product features.
    products = feature_engineering.build_product_features(clean)
    products.to_csv(config.PRODUCTS_CSV, index=False, encoding="utf-8")
    print(f"[3/5] Built features for {len(products):,} products.")

    # Phases 6 to 8 and 10: cluster and profile.
    result = (
        run_segmented_clustering(products)
        if config.SEPARATE_EXCISE
        else run_clustering(products)
    )
    result.products.to_csv(config.CLUSTERED_CSV, index=False, encoding="utf-8")
    result.profiles.to_csv(config.CLUSTER_PROFILE_CSV, index=False, encoding="utf-8")
    silhouette = (
        f"{result.scores[ScoreCol.SILHOUETTE].max():.3f}"
        if result.scores is not None
        else "n/a"
    )
    print(
        f"[4/5] Clustered into {result.optimal_k} clusters "
        f"(general silhouette = {silhouette})."
    )

    # Phases 3 and 9: figures.
    paths = visualization.generate_all(result.products, result.scores)
    print(f"[5/5] Wrote {len(paths)} figures to {config.FIGURES_DIR}.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
