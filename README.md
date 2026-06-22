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

[`notebooks/product_clustering.ipynb`](notebooks/product_clustering.ipynb) is the
narrative deliverable: it walks through all 15 phases (parsing, cleaning, EDA,
feature engineering, scaling, elbow, silhouette, K-Means, visualization,
profiling, and business insights) and renders every figure inline.

### Launch

```bash
uv run jupyter lab   # then open notebooks/product_clustering.ipynb
```

`uv run` executes inside the project's `.venv`, so the notebook shares the exact
locked dependencies as the pipeline. If you prefer the classic interface, use
`uv run jupyter notebook`. To register the environment as a named kernel (for
editors such as VS Code), run once:

```bash
uv run python -m ipykernel install --user --name retail-clustering
```

### What it needs

- The raw receipts must be present in `dataset/struk penjualan 2025/` — the
  first cell parses them directly (no separate pipeline run is required).
- The setup cell locates the project root automatically, so the notebook runs
  from any working directory.

### How to run it

Run the cells top to bottom (`Kernel -> Restart Kernel and Run All Cells`). The
sections are ordered as a dependency chain — each builds on the variables
defined above it:

1. parse and clean the receipts,
2. engineer product features and scale them (`log1p` + `StandardScaler`),
3. sweep `k` and pick the optimal value (elbow + silhouette),
4. fit the final K-Means model and assign clusters,
5. profile the clusters and add the excise-separated refinement (section 13b),
6. summarise the business insights.

Figures are written to `reports/figures/*.png` as they are produced and
displayed inline. Re-running the notebook is idempotent: it overwrites the same
output files each time.

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
