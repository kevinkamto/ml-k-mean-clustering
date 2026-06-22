# Project

Machine Learning K-Means Clustering Analysis for Retail Product Sales Data

## Objective

Build an end-to-end machine learning pipeline that performs product segmentation using K-Means clustering on retail supermarket sales transaction data.

The project should:

1. Parse raw POS receipt logs into a structured dataset.
2. Perform data cleaning and preprocessing.
3. Conduct exploratory data analysis (EDA).
4. Engineer product-level features.
5. Determine the optimal number of clusters using:
   - Elbow Method
   - Silhouette Score

6. Train a K-Means clustering model.
7. Visualize clusters.
8. Interpret clusters from a business perspective.
9. Produce a reproducible Jupyter notebook and report.

---

## Dataset

The raw data lives in `dataset/` and is real, not synthetic.

- `dataset/struk penjualan 2025/` contains 209 daily POS receipt logs as plain text (`.TXT`), one file per trading day. File names follow `SS-YYYYMMDD.TXT`, where `SS` is the station number (for example `02-20250102.TXT`).
- `dataset/tutup data perbulan 2025/` contains monthly closing reports as PDF and RPT files (sales per receipt, per member, per division, stock positions, and so on). These are reference reports, not the parsing input.

Important properties of the receipts:

- Language is Indonesian. The store is "SWALAYAN KEADILAN".
- Currency is Indonesian Rupiah (IDR). Numbers use `.` as the thousands separator, so `5.000` means 5000 and `36.000` means 36000. The parser must strip these separators before numeric conversion.
- Header dates use `DD-MM-YYYY` (`Tanggal : 02-01-2025`); transaction lines use `DD-MM-YY/HH:MM:SS` (`02-01-25/07:41:08`).
- A `**` suffix on a product name marks an excise (cukai) item, such as cigarettes.

The primary parsing input is `dataset/struk penjualan 2025/*.TXT`. See SPEC.md for the exact receipt grammar and output schema.

---

## Technology Stack

### Language

- Python 3.11+

### Package and environment manager

- uv (https://docs.astral.sh/uv/). Do not use pip, virtualenv, or requirements.txt directly.
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Both are committed.

Common commands:

- `uv init` to scaffold the project (once).
- `uv python pin 3.11` to pin the interpreter.
- `uv add pandas numpy matplotlib seaborn scikit-learn scipy` to add runtime dependencies.
- `uv add --dev jupyter ipykernel` to add the notebook toolchain as dev dependencies.
- `uv sync` to install from the lockfile into `.venv`.
- `uv run python -m src.run_pipeline` to run the full pipeline.
- `uv run python -m src.parser` to run a single stage (modules use absolute
  imports, so invoke them with `-m`, not `python src/parser.py`).
- `uv run python lint.py` to run ruff and mypy; `uv run pytest -q` to run tests.
- `uv run jupyter lab` to launch the notebook.

### Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- jupyter (dev)
- ipykernel (dev)

---

## Expected Project Structure

project/
│
├── dataset/                      # raw source data (gitignored: large, contains member PII)
│ ├── struk penjualan 2025/       # daily receipt .TXT logs (parsing input)
│ └── tutup data perbulan 2025/   # monthly closing reports (reference)
│
├── data/
│ ├── processed/                  # cleaned transaction dataset
│ └── exports/                    # product features, cluster assignments
│
├── notebooks/
│ └── product_clustering.ipynb
│
├── src/
│ ├── config.py                   # paths, seed, tunables
│ ├── schema.py                   # StrEnum column names
│ ├── parser.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── clustering.py
│ ├── visualization.py
│ └── run_pipeline.py             # end-to-end driver
│
├── tests/                        # pytest suite (synthetic data)
│
├── reports/                      # final report and figures (gitignored)
│ └── final_report.md
│
├── .github/workflows/ci.yml      # lint + tests on PRs into main
├── lint.py                       # ruff check --fix, ruff format, mypy
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── CLAUDE.md
└── SPEC.md

---

## Coding Guidelines

### General

- Use type hints where appropriate.
- Prefer small reusable functions.
- Avoid hardcoded values.
- Write clear comments for every major step.
- Use snake_case naming.

### Reproducibility

Always use:

random_state = 42

for:

- train/test splits (if any)
- PCA
- K-Means
- any stochastic process

---

## Version Control Workflow

Never commit or push directly to the `main` branch. All changes flow through
a pull request, and every pull request tracks its own GitHub issue:

1. Open (or pick) a GitHub issue describing the work (`gh issue create`).
2. Create a feature branch off the latest `main`, for example
   `git switch -c feature/parser` or `fix/cleaning-edge-case`.
3. Commit the work on that branch with clear, focused commits.
4. Push the branch and open a pull request against `main`
   (`gh pr create --base main`). The PR body must reference and close its
   issue, for example `Closes #12`.
5. The CI workflow (lint, type-check, tests) must pass before merge.
6. Merge the pull request into `main` after review (`gh pr merge`), then
   delete the feature branch.

Keep `main` always in a working, reproducible state. Each PR maps to exactly
one issue. Branch names use `type/short-description` where type is one of
`feature`, `fix`, `docs`, `chore`, or `refactor`.

---

## Notebook Structure

1. Introduction
2. Problem Statement
3. Dataset Description
4. Data Parsing
5. Data Cleaning
6. Exploratory Data Analysis
7. Feature Engineering
8. Data Scaling
9. Elbow Method
10. Silhouette Analysis
11. K-Means Clustering
12. Cluster Visualization
13. Cluster Profiling
14. Business Insights
15. Conclusion

---

## Deliverables

Required outputs:

1. Clean transaction dataset (.csv)
2. Product feature dataset (.csv)
3. Jupyter notebook (.ipynb)
4. Cluster visualizations (.png)
5. Final report (.md or .pdf)

---

## Business Goal

Generate meaningful product groups that can support:

- inventory planning
- product placement
- promotion strategies
- identification of fast-moving products
- identification of slow-moving products
- identification of premium products

The emphasis is on interpretability and business recommendations rather than predictive performance.
