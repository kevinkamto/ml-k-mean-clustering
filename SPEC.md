# Product Sales Clustering Specification

## Problem Statement

Given raw supermarket POS receipt logs, build an unsupervised machine learning solution that segments products according to their sales behavior using K-Means clustering.

---

# Input Data

Source files: `dataset/struk penjualan 2025/*.TXT` (209 daily POS receipt logs).
File naming: `SS-YYYYMMDD.TXT`, where `SS` is the station number.

Language is Indonesian, currency is Indonesian Rupiah (IDR). Numbers use `.` as the
thousands separator (`5.000` = 5000). Strip separators before numeric conversion.

Each `.TXT` file holds many receipts. A single receipt contains:

- a station header: `Station : 02   Tanggal : 02-01-2025`, `Shift : 1   Jam : 07:39:59`
- a cashier block: `ID Kasir`, `Nama Kasir`, `Cash Awal`
- an optional member line: `Nama Anggota : <name>`
- one transaction header line per sale, of the form
  ` : DD-MM-YY/HH:MM:SS  STORE/RECEIPT/SHIFT/CASHIER` (for example `O5BS/00001/1/AGG`)
- one or more product line pairs
- optional per-line discount rows: `Potongan : <qty> x -<unit> -<total>`
- a totals block: `HARGA JUAL`, `POTONGAN PRODUK`, `TOTAL` / `TOTAL BELANJA`,
  `PEMBAYARAN TUNAI` / `PEMBAYARAN E-MONEY` / `PEMBAYARAN VOUCHER`, `KEMBALI`,
  `ANDA HEMAT`, plus tax lines `PPN`, `DPP`, and excise `Cukai`.

Product line pair (name line, then quantity line):

```
1492830 LARISST AIR MIN 1.5L
  #    1      5.000       0        5.000
```

The name line is `<product_code> <product_name>`. The quantity line is
`# <quantity> <unit_price> <line_discount> <line_total>`. A trailing `**` on the
name marks an excise (cukai) item, for example `0357330 SAMPOERNA MILD 16'S **`.

---

# Phase 1: Receipt Parsing

## Objective

Convert raw receipt text into a structured transaction table, one row per product
line. Iterate over every `.TXT` file in `dataset/struk penjualan 2025/`.

## Output Schema

transactions

| column               | type     | source                                          |
| -------------------- | -------- | ----------------------------------------------- |
| source_file          | string   | receipt file name                               |
| station              | string   | station header / file name                      |
| transaction_id       | string   | receipt code on the transaction header line     |
| transaction_datetime | datetime | transaction header line (`DD-MM-YY/HH:MM:SS`)   |
| product_code         | string   | product name line                               |
| product_name         | string   | product name line                               |
| quantity             | integer  | quantity line                                   |
| unit_price           | float    | quantity line                                   |
| line_discount        | float    | quantity line and `Potongan` rows (default 0)   |
| line_total           | float    | quantity line                                   |
| is_excise            | boolean  | `**` marker on the product name (default false) |

Acceptance Criteria:

- all receipts across all files extracted
- all product lines extracted, including multi-item receipts
- thousands separators removed and numerics converted correctly
- discounts captured (line column plus `Potongan` rows)
- excise items flagged
- no duplicated rows

---

# Phase 2: Data Cleaning

Tasks:

1. Remove duplicates
2. Handle missing values
3. Validate data types
4. Validate quantities
5. Validate prices

Acceptance Criteria:

- no invalid quantities
- no negative prices
- no duplicated transactions

---

# Phase 3: Exploratory Data Analysis

Required analyses:

1. Dataset dimensions
2. Product counts
3. Transaction counts
4. Quantity distributions
5. Revenue distributions
6. Top-selling products
7. Highest revenue products

Required plots:

- bar charts
- histograms
- box plots
- correlation heatmap

---

# Phase 4: Product-Level Aggregation

Create one row per product.

Output table:

products

| column              | type    |
| ------------------- | ------- |
| product_code        | string  |
| product_name        | string  |
| total_quantity_sold | float   |
| total_revenue       | float   |
| transaction_count   | integer |
| average_price       | float   |

---

# Phase 5: Feature Engineering

Create additional features:

average_quantity_per_transaction
= total_quantity_sold / transaction_count

revenue_per_transaction
= total_revenue / transaction_count

Optional features:

- discount_frequency
- average_discount
- sales_frequency_per_day

---

# Final Feature Set

Use only numerical features:

[
total_quantity_sold,
total_revenue,
transaction_count,
average_price,
average_quantity_per_transaction,
revenue_per_transaction
]

Exclude:

- product_code
- product_name

---

# Phase 6: Scaling

Algorithm:

- log1p transform (retail features are strongly right-skewed), then
- StandardScaler

The log1p step stops K-Means from simply isolating a few high-volume outliers
and produces interpretable segments. All clustering features are non-negative,
so log1p is well defined.

Acceptance Criteria:

- all clustering features standardized
- transformed matrix contains no missing values

---

# Phase 7: Determine Optimal K

Methods:

1. Elbow Method
2. Silhouette Score

Search Range:

k = 2 to 10

Deliverables:

- WCSS plot
- Silhouette score table
- selected optimal k with explanation

---

# Phase 8: K-Means Clustering

Algorithm:

sklearn.cluster.KMeans

Configuration:

n_clusters = optimal_k
random_state = 42
n_init = 20

Outputs:

- cluster labels
- cluster centroids
- products with assigned clusters

---

# Phase 9: Visualization

Required:

1. PCA projection to 2 dimensions
2. Scatter plot of clusters
3. Cluster size distribution
4. Cluster profile table

---

# Phase 10: Cluster Profiling

For each cluster compute:

- average quantity sold
- average revenue
- average price
- average transaction count

Provide business interpretations.

Example:

Cluster A
Fast-moving low-price products.

Cluster B
Premium products with high revenue.

Cluster C
Slow-moving products requiring promotion.

---

# Phase 11: Business Recommendations

Provide recommendations for:

Inventory Management
Promotion Strategy
Shelf Placement
Product Bundling
Stock Optimization

---

# Success Criteria

The project is complete when:

1. Raw receipts are parsed successfully.
2. Product feature table is generated.
3. Optimal k is determined.
4. K-Means model is trained.
5. Products are assigned to clusters.
6. Visualizations are generated.
7. Business interpretations are documented.
8. Notebook executes from start to finish without errors.

---

# Delivery Workflow

Changes are never committed or pushed directly to `main`. Each unit of work
is delivered through a pull request:

1. Branch off the latest `main` (`type/short-description`, where type is one
   of `feature`, `fix`, `docs`, `chore`, `refactor`).
2. Commit the work on that branch.
3. Open a pull request against `main` and merge it after review.
4. Delete the feature branch once merged.

`main` must always remain in a working, reproducible state.
