"""Tests for the cleaning stage."""

from __future__ import annotations

import pandas as pd

from src import preprocessing
from src.schema import TxnCol


def test_coerce_bool_handles_strings_and_bools() -> None:
    s = pd.Series(["True", "False", "true", "false", "1", "0"])
    result = preprocessing.coerce_bool(s).tolist()
    assert result == [True, False, True, False, True, False]
    native = pd.Series([True, False])
    assert preprocessing.coerce_bool(native).tolist() == [True, False]


def test_clean_removes_duplicates_and_invalids(transactions_df) -> None:
    df = transactions_df.copy()
    # Add a duplicate row, a zero-quantity row, and a negative-price row.
    dup = df.iloc[[0]]
    bad_qty = df.iloc[[0]].assign(**{TxnCol.QUANTITY: 0})
    bad_price = df.iloc[[0]].assign(**{TxnCol.UNIT_PRICE: -1})
    dirty = pd.concat([df, dup, bad_qty, bad_price], ignore_index=True)

    clean, report = preprocessing.clean_transactions(dirty)

    assert report.duplicates_removed == 1
    assert report.invalid_quantity_removed == 1
    assert report.invalid_price_removed == 1
    assert (clean[TxnCol.QUANTITY] > 0).all()
    assert (clean[TxnCol.UNIT_PRICE] >= 0).all()
    assert clean[TxnCol.IS_EXCISE].dtype == bool


def test_clean_is_idempotent(transactions_df) -> None:
    once, _ = preprocessing.clean_transactions(transactions_df)
    twice, report = preprocessing.clean_transactions(once)
    assert len(once) == len(twice)
    assert report.duplicates_removed == 0
