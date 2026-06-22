"""Shared fixtures: a tiny synthetic receipt and transactions frame.

Tests are independent of the real (PII-bearing, gitignored) dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.schema import TxnCol

# A minimal but representative receipt: two transactions, an excise item,
# an immediate discount column, and a promotional ``Potongan`` row.
SAMPLE_RECEIPT = """\
============ INISIALISASI ==============
Station : 02       Tanggal : 02-01-2025
Shift   : 1        Jam     :   07:39:59
----------------------------------------
 : 02-01-25/07:41:08 O5BS/00001/1/AGG
----------------------------------------
1492830 LARISST AIR MIN 1.5L
  #    1      5.000       0        5.000
----------------------------------------
 : 02-01-25/07:45:37 O5BS/00002/1/AGG
----------------------------------------
0357330 SAMPOERNA MILD 16'S           **
  #    2     36.000       0       72.000
0365220 SANIA MNYK GR PCH 2L
  #    1     39.100       0       39.100
       Potongan :   1 x  -1.600     -1.600
----------------------------------------
"""


@pytest.fixture
def receipt_file(tmp_path):
    """Write the sample receipt to a station-named file and return its path."""
    path = tmp_path / "02-20250102.TXT"
    path.write_text(SAMPLE_RECEIPT, encoding="utf-8")
    return path


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """A small clean transactions frame covering several products."""
    rows = [
        # product, qty, unit_price, line_total, discount, n_txn (encoded by uid)
        ("P1", "FAST CHEAP A", 5, 1000, 5000, 0, "u1", False),
        ("P1", "FAST CHEAP A", 3, 1000, 3000, 0, "u2", False),
        ("P2", "PREMIUM B", 1, 50000, 50000, 0, "u1", True),
        ("P2", "PREMIUM B", 1, 50000, 48000, 2000, "u3", True),
        ("P3", "SLOW C", 1, 2000, 2000, 0, "u4", False),
        ("P4", "MID D", 2, 8000, 16000, 0, "u2", False),
        ("P5", "MID E", 4, 3000, 12000, 0, "u3", False),
        ("P6", "BULK F", 10, 500, 5000, 0, "u4", False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            TxnCol.PRODUCT_CODE,
            TxnCol.PRODUCT_NAME,
            TxnCol.QUANTITY,
            TxnCol.UNIT_PRICE,
            TxnCol.LINE_TOTAL,
            TxnCol.LINE_DISCOUNT,
            TxnCol.TRANSACTION_UID,
            TxnCol.IS_EXCISE,
        ],
    )
    df[TxnCol.TRANSACTION_DATETIME] = pd.to_datetime("2025-01-02")
    return df
