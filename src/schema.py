"""Column-name and category enums.

Centralising the DataFrame column names as ``StrEnum`` members removes raw
string literals scattered across the pipeline: a typo becomes an ``AttributeError``
instead of a silent ``KeyError``. ``StrEnum`` members are real ``str`` instances,
so they are interchangeable with plain strings as DataFrame column labels and
survive a CSV round-trip.
"""

from __future__ import annotations

from enum import StrEnum


class TxnCol(StrEnum):
    """Columns of the parsed / cleaned transactions table (one row per line)."""

    SOURCE_FILE = "source_file"
    STATION = "station"
    TRANSACTION_ID = "transaction_id"
    TRANSACTION_UID = "transaction_uid"
    TRANSACTION_DATETIME = "transaction_datetime"
    PRODUCT_CODE = "product_code"
    PRODUCT_NAME = "product_name"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    LINE_TOTAL = "line_total"
    LINE_DISCOUNT = "line_discount"
    IS_EXCISE = "is_excise"


class ProdCol(StrEnum):
    """Columns of the product feature / clustered table (one row per product)."""

    PRODUCT_CODE = "product_code"
    PRODUCT_NAME = "product_name"
    TOTAL_QUANTITY_SOLD = "total_quantity_sold"
    TOTAL_REVENUE = "total_revenue"
    TRANSACTION_COUNT = "transaction_count"
    AVERAGE_PRICE = "average_price"
    AVERAGE_QUANTITY_PER_TRANSACTION = "average_quantity_per_transaction"
    REVENUE_PER_TRANSACTION = "revenue_per_transaction"
    # Intermediate aggregates used to derive the descriptive features.
    TOTAL_DISCOUNT = "total_discount"
    DISCOUNTED_LINES = "discounted_lines"
    LINE_COUNT = "line_count"
    # Descriptive (non-clustering) features.
    DISCOUNT_FREQUENCY = "discount_frequency"
    AVERAGE_DISCOUNT = "average_discount"
    SALES_FREQUENCY_PER_DAY = "sales_frequency_per_day"
    IS_EXCISE = "is_excise"
    # Assigned by the clustering stage.
    CLUSTER = "cluster"


class ScoreCol(StrEnum):
    """Columns of the per-k evaluation table."""

    K = "k"
    INERTIA = "inertia"
    SILHOUETTE = "silhouette"


class ProfileCol(StrEnum):
    """Columns of the per-cluster profile table."""

    CLUSTER = "cluster"
    N_PRODUCTS = "n_products"
    AVG_QUANTITY_SOLD = "avg_quantity_sold"
    AVG_REVENUE = "avg_revenue"
    AVG_PRICE = "avg_price"
    AVG_TRANSACTION_COUNT = "avg_transaction_count"
    AVG_QTY_PER_TXN = "avg_qty_per_txn"
    AVG_REVENUE_PER_TXN = "avg_revenue_per_txn"
    TOTAL_REVENUE = "total_revenue"
    REVENUE_SHARE = "revenue_share"
