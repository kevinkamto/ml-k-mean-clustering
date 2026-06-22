"""Tests that StrEnum column names interoperate with plain strings."""

from __future__ import annotations

import pandas as pd

from src.schema import ProdCol, TxnCol


def test_enum_members_equal_plain_strings() -> None:
    assert TxnCol.QUANTITY == "quantity"
    assert ProdCol.TOTAL_REVENUE == "total_revenue"


def test_enum_and_string_select_same_column() -> None:
    df = pd.DataFrame({"quantity": [1, 2, 3]})
    # Selecting with the enum member must hit the string-labelled column.
    assert df[TxnCol.QUANTITY].tolist() == df["quantity"].tolist()


def test_enum_survives_csv_round_trip(tmp_path) -> None:
    df = pd.DataFrame({TxnCol.QUANTITY: [1, 2], TxnCol.UNIT_PRICE: [10, 20]})
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["quantity", "unit_price"]
    assert reloaded[TxnCol.QUANTITY].tolist() == [1, 2]
