# Product Segmentation with K-Means Clustering

**Retail supermarket "SWALAYAN KEADILAN" - 2025 POS receipts**

> Reproduce all numbers and figures in this report with:
> `uv run python -m src.run_pipeline`
> Outputs are written to `data/processed/`, `data/exports/`, and
> `reports/figures/`. The numbers in section 5 are from the full 2025 dataset
> (209 receipt logs).

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

## 5. Results (full 2025 dataset)

Parsing the 209 receipt logs yields **52,067 product lines**. The dataset
folder is scoped to trading year 2025, but one stray file (a 27 December 2024
receipt log) is bundled in; the cleaning step drops its **306 product lines**
as out-of-scope for the analysis year (`config.ANALYSIS_YEAR`), alongside no
duplicates, missing keys, or invalid quantities/prices otherwise. This leaves
**51,761 clean product lines** across **23,891 transactions** and **1,735
distinct products**.

### 5a. General segmentation

Sweeping `k` from 2 to 10 over all products, the silhouette score is highest at
**k = 2 (silhouette = 0.333)**, with the elbow curve flattening over the same
range. The two segments split cleanly into a revenue-dominant core and a long
tail:

| cluster | n_products | avg_quantity_sold | avg_revenue (IDR) | avg_price (IDR) | avg_txn_count | revenue_share |
| ------- | ---------- | ----------------- | ----------------- | --------------- | ------------- | ------------- |
| 0 | 1,010 | 88.6 | 934,481 | 12,837 | 47.9 | 92.9% |
| 1 | 725 | 5.4 | 100,003 | 19,033 | 4.7 | 7.1% |

Cluster 0 is the **fast-moving core**: high quantity and transaction counts at a
lower price point, carrying 93% of revenue. Cluster 1 is the **slow-moving long
tail**: many products, each sold rarely, contributing only 7% of revenue. See
`reports/figures/kmeans_cluster_sizes.png` and
`reports/figures/kmeans_pca_clusters.png`.

### 5b. Excise-separated refinement (pipeline default)

With `SEPARATE_EXCISE` enabled (the configured default), excise (cukai) goods are
clustered in their own pass so their extreme price and revenue do not distort the
general segments. This is what `uv run python -m src.run_pipeline` writes to
`data/exports/cluster_profiles.csv`. Each `segment_group` keeps its own
silhouette-selected `k`, yielding **12 clusters** in total across the general and
excise groups. The refinement surfaces distinct high-ticket and high-volume
sub-segments (for example, small clusters of products averaging millions of IDR
in revenue) that the two-way split folds into cluster 0.

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
