# Retail Product Clustering

K-Means product segmentation on real Indonesian supermarket POS receipt logs
("SWALAYAN KEADILAN", 2025). The pipeline parses raw receipts, engineers
product-level features, selects the number of clusters via the elbow and
silhouette methods, fits a reproducible K-Means model, and produces business
oriented cluster profiles and figures.

See [CLAUDE.md](CLAUDE.md) for the full project guide and [SPEC.md](SPEC.md) for
the data grammar and phase-by-phase specification.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
uv sync            # create .venv and install locked dependencies
```

The raw dataset under `dataset/` is gitignored (it is large and contains member
PII). Place the receipt logs in `dataset/struk penjualan 2025/` before running.

## Run the pipeline

```bash
uv run python -m src.run_pipeline
```

Outputs:

| Path | Contents |
| ---- | -------- |
| `data/processed/transactions.csv` | clean transaction table |
| `data/exports/product_features.csv` | product feature table |
| `data/exports/product_clusters.csv` | products with assigned clusters |
| `data/exports/cluster_profiles.csv` | per-cluster profile table |
| `reports/figures/*.png` | EDA and cluster visualisations |

## Notebook

```bash
uv run jupyter lab   # open notebooks/product_clustering.ipynb
```

The notebook walks through all 15 phases and renders the figures inline.

## Development

```bash
uv run python lint.py          # ruff check --fix, ruff format, mypy
uv run python lint.py --check  # no changes; fail if not clean (CI mode)
uv run pytest -q               # run the test suite
```

Run individual stages as modules, for example:

```bash
uv run python -m src.parser
uv run python -m src.clustering
```

## Project layout

```
src/            pipeline package (parser, preprocessing, feature_engineering,
                clustering, visualization, run_pipeline, config, schema)
tests/          pytest suite (synthetic data, no dataset needed)
notebooks/      product_clustering.ipynb
reports/        final report and generated figures (gitignored)
data/           processed and exported CSVs (gitignored)
dataset/        raw POS receipts and monthly reports (gitignored)
```

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint, type-check,
and tests on every pull request into `main`. Work on a feature branch and open a
pull request; never commit directly to `main`.
