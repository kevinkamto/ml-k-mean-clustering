# Product Segmentation with K-Means Clustering

**Retail supermarket "SWALAYAN KEADILAN" - 2025 POS receipts**

> Reproduce all numbers and figures in this report with:
> `uv run python -m src.run_pipeline`
> Outputs are written to `data/processed/`, `data/exports/`, and
> `reports/figures/`. The illustrative figures in section 6 were produced from a
> 6-file validation subset and are marked as such; the full run overwrites them
> with the complete dataset.

---

## 1. Objective

Segment products by their sales behaviour so the business can treat fast-moving
staples, premium high-ticket items, and slow-moving long-tail products
differently across inventory, pricing, shelf placement, and promotions. The
emphasis is interpretability, not prediction.

## 2. Data

* **Source:** `dataset/struk penjualan 2025/*.TXT` - 209 daily POS receipt logs.
* **Language / currency:** Indonesian; Indonesian Rupiah (IDR), with `.` as the
  thousands separator (`5.000` = 5000).
* **Unit of record after parsing:** one row per product line on a receipt.

The parser (`src/parser.py`) handles the receipt grammar described in `SPEC.md`:
transaction header lines (datetime + receipt code), product name/quantity line
pairs, `Potongan` discount rows, and the `**` excise (cukai) marker.

## 3. Method

The pipeline is a sequence of small, testable stages under `src/`:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| Parse | `parser.py` | Raw `.TXT` to a transaction table |
| Clean | `preprocessing.py` | Dedup, validate types, drop bad qty/price |
| Aggregate | `feature_engineering.py` | One row per product, base + ratio + temporal features |
| Scale | `clustering.py` | `log1p` then `StandardScaler` |
| Select k | `clustering.py` | Elbow (inertia) + silhouette over k = 2..10 |
| Cluster | `clustering.py` | `KMeans(random_state=42, n_init=20)`, excise products clustered separately |
| Profile | `clustering.py` | Per-cluster averages, revenue share, segment group |
| Visualise | `visualization.py` | EDA + cluster figures |

**Clustering features** (numerical only): `total_quantity_sold`,
`total_revenue`, `transaction_count`, `average_price`,
`average_quantity_per_transaction`, `revenue_per_transaction`, plus the temporal
features `active_days` and `monthly_cv`. Descriptive temporal features
(`recency_days`, `weekend_ratio`) are kept for profiling.

**Separate excise clustering.** Excise (cukai) goods such as cigarettes have
extreme price and revenue. With `SEPARATE_EXCISE` enabled they are scaled and
clustered in their own pass (tagged `segment_group == "excise"`) so the general
segments stay clean.

**Why log1p before scaling.** Retail sales are strongly right-skewed: a handful
of products dominate volume and revenue. With `StandardScaler` alone, K-Means
simply isolates those few outliers, producing a degenerate split. Applying
`log1p` first yields balanced, interpretable segments while still satisfying the
"standardised, no missing values" requirement.

**Reproducibility.** Every stochastic step (PCA, K-Means) uses
`random_state = 42`, set centrally in `src/config.py`.

## 4. Choosing the number of clusters

`k` is swept from 2 to 10. The elbow curve (inertia) shows diminishing returns,
and the **silhouette score** provides a single objective criterion; the pipeline
selects the `k` with the highest silhouette. See
`reports/figures/kmeans_elbow_silhouette.png` and
`data/exports/` for the per-k table written at run time.

## 5. Results (fill from the full run)

After running the pipeline, record:

* Selected `k` and its silhouette score.
* Cluster sizes (`reports/figures/kmeans_cluster_sizes.png`).
* The per-cluster profile (`data/exports/cluster_profiles.csv`).
* The PCA projection (`reports/figures/kmeans_pca_clusters.png`).

### 5a. Illustrative result (6-file validation subset, not the full dataset)

Used only to demonstrate the pipeline end to end during development:

| cluster | n_products | avg_quantity_sold | avg_revenue (IDR) | avg_price (IDR) | revenue_share |
| ------- | ---------- | ----------------- | ----------------- | --------------- | ------------- |
| 0 | 158 | 10.1 | 129,758 | 18,393 | 69.8% |
| 1 | 526 | 1.5 | 16,852 | 12,431 | 30.2% |

Even on a small subset the segments are sensible: a smaller group of
higher-volume, higher-revenue products versus a large low-volume long tail. The
full dataset typically supports a richer `k`.

## 6. Cluster interpretation template

Label each cluster from its profile row:

* **Fast-moving staples** - high quantity and transaction count, low price.
  Action: keep deep stock, eye-level placement, prevent stock-outs.
* **Premium / high-ticket** - high revenue per transaction and price, lower
  volume. Action: protect margin, feature in bundles, targeted promotions.
* **Slow-moving long tail** - low volume and revenue. Action: review shelf
  space, run clearance, or consider delisting.

## 7. Business recommendations

* **Inventory management:** size reorder points by segment; deep safety stock
  for fast-movers, lean stock for the long tail.
* **Promotion strategy:** volume promotions on staples to drive footfall;
  margin-aware, targeted offers on premium items.
* **Shelf placement:** fast-movers at eye level and near entrances; premium in
  destination zones; long tail consolidated.
* **Product bundling:** pair premium items with complementary fast-movers.
* **Stock optimisation:** schedule clearance/delisting reviews for the slowest
  segment each cycle.

## 8. Limitations and next steps

* Seasonality is captured with summary features (`monthly_cv`, `active_days`,
  `recency_days`, `weekend_ratio`) rather than a full time-series model; richer
  temporal modelling remains future work.
* Excise (cukai) products are now clustered separately (`segment_group`), so they
  no longer skew the general segments.
* Clusters describe behaviour, not cause; pair with category/division data from
  the monthly reports for deeper action.

## 9. How to reproduce

```bash
uv sync                              # install locked dependencies
uv run python -m src.run_pipeline    # run all phases, write all outputs
uv run jupyter lab                   # open notebooks/product_clustering.ipynb
```
