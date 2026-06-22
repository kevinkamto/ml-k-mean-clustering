"""Phase 1: Receipt parsing.

Convert raw Indonesian supermarket POS receipt logs (``dataset/struk
penjualan 2025/*.TXT``) into a structured, one-row-per-product-line
transaction table.

Receipt grammar (see SPEC.md for the full description)::

    ----------------------------------------
     : 02-01-25/07:41:08 O5BS/00001/1/AGG
    ----------------------------------------
    1492830 LARISST AIR MIN 1.5L
      #    1      5.000       0        5.000
           Potongan :   1 x  -1.600     -1.600

* The transaction header line carries the datetime and the receipt code.
* Each product is a *pair* of lines: a name line ``<code> <name>`` followed
  by a quantity line ``# <qty> <unit_price> <immediate_discount> <line_total>``.
* Optional ``Potongan`` rows after a product carry promotional discounts.
* A trailing ``**`` on the name marks an excise (cukai) item.

All monetary amounts use ``.`` as a thousands separator (``5.000`` == 5000).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config
from src.schema import TxnCol

# --- Regular expressions describing the receipt grammar ----------------------

# Transaction header, e.g. " : 02-01-25/07:41:08 O5BS/00001/1/AGG".
_TXN_HEADER_RE = re.compile(
    r"^\s*:\s*"
    r"(?P<dt>\d{2}-\d{2}-\d{2}/\d{2}:\d{2}:\d{2})\s+"
    r"(?P<code>\S+)"
)

# Quantity line, e.g. "  #    1      5.000       0        5.000".
_QTY_LINE_RE = re.compile(
    r"^\s*#\s+"
    r"(?P<qty>\d+)\s+"
    r"(?P<unit_price>[\d.]+)\s+"
    r"(?P<immediate_discount>[\d.]+)\s+"
    r"(?P<line_total>[\d.]+)\s*$"
)

# Product name line, e.g. "1492830 LARISST AIR MIN 1.5L" (optionally "** ").
_NAME_LINE_RE = re.compile(r"^\s*(?P<code>\d+)\s+(?P<name>\S.*?)\s*$")

# Promotional discount row, e.g. "Potongan :   1 x  -1.600     -1.600".
_POTONGAN_RE = re.compile(r"Potongan\s*:\s*\d+\s*x\s*-?[\d.]+\s+-?(?P<amount>[\d.]+)")

# Datetime format used on transaction header lines (DD-MM-YY/HH:MM:SS).
_DATETIME_FORMAT = "%d-%m-%y/%H:%M:%S"

# Marker appended to a product name for excise (cukai) goods.
_EXCISE_MARKER = "**"


def parse_amount(raw: str) -> int:
    """Convert a receipt amount string into an integer of Rupiah.

    Thousands separators (``.``) are stripped, e.g. ``"39.100" -> 39100``.
    """
    cleaned = raw.strip().replace(".", "")
    if cleaned in ("", "-"):
        return 0
    return int(cleaned)


def _read_text(path: Path) -> list[str]:
    """Read a receipt file, tolerating non-UTF-8 legacy POS encodings."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    # latin-1 maps every byte, so this is effectively unreachable.
    return path.read_text(encoding="latin-1", errors="replace").splitlines()


def _station_from_filename(path: Path) -> str:
    """Extract the station code from a ``SS-YYYYMMDD.TXT`` file name."""
    return path.stem.split("-", 1)[0]


def parse_receipt_file(path: Path) -> list[dict]:
    """Parse one receipt ``.TXT`` file into a list of product-line records."""
    lines = _read_text(path)
    source_file = path.name
    station = _station_from_filename(path)

    records: list[dict] = []
    current_txn_id: str | None = None
    current_dt: str | None = None
    last_record: dict | None = None  # the most recent product line (for Potongan)

    for idx, line in enumerate(lines):
        # 1) Transaction header: opens a new receipt context.
        header = _TXN_HEADER_RE.match(line)
        if header:
            current_txn_id = header.group("code")
            current_dt = header.group("dt")
            last_record = None
            continue

        # 2) Quantity line: pair it with the preceding product name line.
        qty_match = _QTY_LINE_RE.match(line)
        if qty_match:
            if current_txn_id is None or idx == 0:
                # Quantity line with no enclosing transaction; skip defensively.
                last_record = None
                continue
            name_match = _NAME_LINE_RE.match(lines[idx - 1])
            if name_match is None:
                last_record = None
                continue

            name = name_match.group("name")
            is_excise = name.endswith(_EXCISE_MARKER)
            if is_excise:
                name = name[: -len(_EXCISE_MARKER)].strip()

            quantity = int(qty_match.group("qty"))
            unit_price = parse_amount(qty_match.group("unit_price"))
            immediate_discount = parse_amount(qty_match.group("immediate_discount"))
            printed_total = parse_amount(qty_match.group("line_total"))

            record = {
                TxnCol.SOURCE_FILE: source_file,
                TxnCol.STATION: station,
                TxnCol.TRANSACTION_ID: current_txn_id,
                # A receipt code resets daily, so scope uniqueness to the file.
                TxnCol.TRANSACTION_UID: f"{source_file}::{current_txn_id}",
                TxnCol.TRANSACTION_DATETIME: current_dt,
                TxnCol.PRODUCT_CODE: name_match.group("code"),
                TxnCol.PRODUCT_NAME: name,
                TxnCol.QUANTITY: quantity,
                TxnCol.UNIT_PRICE: unit_price,
                # ``printed_total`` already has the immediate discount removed;
                # promotional ``Potongan`` rows are subtracted as we see them.
                TxnCol.LINE_TOTAL: printed_total,
                TxnCol.LINE_DISCOUNT: immediate_discount,
                TxnCol.IS_EXCISE: is_excise,
            }
            records.append(record)
            last_record = record
            continue

        # 3) Promotional discount: adjust the most recent product line.
        potongan = _POTONGAN_RE.search(line)
        if potongan and last_record is not None:
            promo = parse_amount(potongan.group("amount"))
            last_record[TxnCol.LINE_DISCOUNT] += promo
            last_record[TxnCol.LINE_TOTAL] -= promo

    return records


def parse_all(raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Parse every receipt file under ``raw_dir`` into a transactions frame.

    Returns a DataFrame following the Phase 1 schema in SPEC.md.
    """
    files = sorted(raw_dir.glob("*.TXT"))
    if not files:
        raise FileNotFoundError(
            f"No .TXT receipt files found under {raw_dir!r}. "
            "Check that the dataset is present."
        )

    all_records: list[dict] = []
    for path in files:
        all_records.extend(parse_receipt_file(path))

    if not all_records:
        raise ValueError(
            f"Parsed {len(files)} files but extracted 0 product lines. "
            "The receipt format may have changed."
        )

    df = pd.DataFrame.from_records(all_records)
    # Convert the datetime column from the receipt format to real datetimes.
    df[TxnCol.TRANSACTION_DATETIME] = pd.to_datetime(
        df[TxnCol.TRANSACTION_DATETIME], format=_DATETIME_FORMAT, errors="coerce"
    )
    return df


def main() -> None:
    """Parse all receipts and write the raw transactions table to CSV."""
    config.ensure_output_dirs()
    df = parse_all()
    df.to_csv(config.RAW_TRANSACTIONS_CSV, index=False, encoding="utf-8")
    print(
        f"Parsed {df[TxnCol.SOURCE_FILE].nunique()} files -> "
        f"{len(df):,} product lines, "
        f"{df[TxnCol.TRANSACTION_UID].nunique():,} transactions, "
        f"{df[TxnCol.PRODUCT_CODE].nunique():,} distinct products."
    )
    print(f"Wrote {config.RAW_TRANSACTIONS_CSV}")


if __name__ == "__main__":
    main()
