"""Tests for the receipt parser."""

from __future__ import annotations

import pandas as pd
import pytest

from src import parser
from src.schema import TxnCol


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5.000", 5000), ("36.000", 36000), ("39.100", 39100), ("0", 0), ("-", 0)],
)
def test_parse_amount_strips_thousands_separator(raw: str, expected: int) -> None:
    assert parser.parse_amount(raw) == expected


def test_parse_receipt_file_extracts_all_lines(receipt_file) -> None:
    records = parser.parse_receipt_file(receipt_file)
    # 1 product in receipt 1, 2 products in receipt 2.
    assert len(records) == 3


def test_parser_station_and_uid(receipt_file) -> None:
    records = parser.parse_receipt_file(receipt_file)
    first = records[0]
    assert first[TxnCol.STATION] == "02"
    assert first[TxnCol.SOURCE_FILE] == "02-20250102.TXT"
    # uid scopes the daily-resetting receipt code to the file.
    assert first[TxnCol.TRANSACTION_UID].startswith("02-20250102.TXT::")


def test_parser_excise_flag_and_name_cleanup(receipt_file) -> None:
    records = parser.parse_receipt_file(receipt_file)
    excise = [r for r in records if r[TxnCol.IS_EXCISE]]
    assert len(excise) == 1
    # The trailing ** marker is stripped from the name.
    assert excise[0][TxnCol.PRODUCT_NAME] == "SAMPOERNA MILD 16'S"


def test_parser_quantity_and_totals(receipt_file) -> None:
    records = parser.parse_receipt_file(receipt_file)
    sampoerna = next(r for r in records if r[TxnCol.PRODUCT_CODE] == "0357330")
    assert sampoerna[TxnCol.QUANTITY] == 2
    assert sampoerna[TxnCol.UNIT_PRICE] == 36000
    assert sampoerna[TxnCol.LINE_TOTAL] == 72000


def test_parser_promotional_discount_applied(receipt_file) -> None:
    records = parser.parse_receipt_file(receipt_file)
    sania = next(r for r in records if r[TxnCol.PRODUCT_CODE] == "0365220")
    # 39100 printed total minus a 1600 Potongan row.
    assert sania[TxnCol.LINE_DISCOUNT] == 1600
    assert sania[TxnCol.LINE_TOTAL] == 39100 - 1600


def test_parse_all_raises_without_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse_all(tmp_path)


def test_parse_all_builds_datetime(receipt_file) -> None:
    df = parser.parse_all(receipt_file.parent)
    assert pd.api.types.is_datetime64_any_dtype(df[TxnCol.TRANSACTION_DATETIME])
    assert df[TxnCol.TRANSACTION_DATETIME].notna().all()
