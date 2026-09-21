#!/usr/bin/env python3
"""
Preprocess raw ETF price CSVs into standardised monthly-return CSVs.

Supported sources
-----------------
  raw_curvo/  – Curvo exports
                Header: ``Data,<name>``
                Separator: comma
                Date format: ``MM/YYYY``
                Decimal: period  (e.g. ``10067.64``)

  raw_msci/   – MSCI / direct index exports
                Header: ``Date;TICKER;;;``
                Separator: semicolon
                Date format: ``YYYY-MM-DD``
                Decimal: comma  (e.g. ``99,968221``)

  BTOP50      – ``data/BTOP50_Index_historical_data 3.csv``, a two-table
                export (monthly % returns, then a VAMI price-index table).
                We parse the VAMI table (semicolon-separated, comma decimal,
                ``Year;Jan;Feb;...;Dec`` grid) directly as a price series —
                see :func:`_parse_btop50_vami`.

Both sources are normalised to the same output schema:
  month_year, price_eur, month_return, month_return_after_TER
where ``month_year`` is stored as ``MM/YYYY``.

A row's ``month_return`` is the return REALISED DURING that calendar month
— i.e. ``price[month] / price[previous month] - 1`` — so the FIRST row of
each series has an empty return (there is no prior price) and every
subsequent row's return is attributed to the month it actually happened in.
(An earlier version of this script attributed month M's return to row M-1,
which shifted every historical crash back by one month — e.g. October 1987
and March 2020 losses were mislabelled September 1987 / February 2020 — and
silently broke any ``date_start``/``date_end`` filter by one month.)

Usage:  python scripts/preprocess.py
"""

from __future__ import annotations

import csv
import os
import sys

# ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import config as cfg


# ── parsers ───────────────────────────────────────────────────────────────────

def _parse_curvo(raw_path: str) -> list[tuple[str, float]]:
    """Parse a Curvo CSV → list of (``MM/YYYY``, price)."""
    rows: list[tuple[str, float]] = []
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header  ("Data,<ETF name>")
        for line in reader:
            if len(line) < 2:
                continue
            date_str = line[0].strip()          # already MM/YYYY
            try:
                price = float(line[1].strip())
            except ValueError:
                continue
            rows.append((date_str, price))
    return rows


def _parse_msci(raw_path: str) -> list[tuple[str, float]]:
    """Parse an MSCI/direct-export CSV → list of (``MM/YYYY``, price).

    Handles:
    * Semicolon delimiter
    * European decimal separator (comma → period)
    * Date format ``YYYY-MM-DD`` → converted to ``MM/YYYY``
    """
    rows: list[tuple[str, float]] = []
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header  ("Date;TICKER;;;")
        for line in reader:
            if len(line) < 2:
                continue
            date_raw  = line[0].strip()                     # YYYY-MM-DD
            price_raw = line[1].strip().replace(",", ".")   # European decimal
            if not date_raw or not price_raw:
                continue
            try:
                price = float(price_raw)
            except ValueError:
                continue
            # YYYY-MM-DD  →  MM/YYYY
            parts = date_raw.split("-")
            date_str = f"{parts[1]}/{parts[0]}"
            rows.append((date_str, price))
    return rows


def _parse_btop50_vami(raw_path: str) -> list[tuple[str, float]]:
    """Parse the VAMI (price-index) table out of the BTOP50 export → list of
    (``MM/YYYY``, price), chronological, with a synthetic base row one month
    before the first real entry (price=1000, matching the source's own
    normalisation) so the very first real month gets a return like every
    other series.

    The file contains TWO tables (see module docstring): a monthly-%-return
    grid first, then a blank/label row, then the VAMI price grid we use here
    — ``;Jan;Feb;...;Dec`` header, ``Year;v1;v2;...;v12`` data rows,
    semicolon-separated, comma decimals, trailing months of the final
    (incomplete) year left blank.
    """
    with open(raw_path, newline="", encoding="utf-8") as f:
        raw_lines = list(csv.reader(f, delimiter=";"))

    vami_start = None
    for i, line in enumerate(raw_lines):
        if any("VAMI" in cell for cell in line):
            vami_start = i + 2   # skip the label row and the "Jan;Feb;..." header row
            break
    if vami_start is None:
        raise ValueError(f"No VAMI table found in {raw_path}")

    entries: list[tuple[int, int, float]] = []   # (year, month, price)
    for line in raw_lines[vami_start:]:
        if not line or not line[0].strip():
            continue
        year = int(line[0].strip())
        for m, cell in enumerate(line[1:13], start=1):
            cell = cell.strip().replace(",", ".")
            if not cell:
                continue
            entries.append((year, m, float(cell)))

    entries.sort(key=lambda e: (e[0], e[1]))
    if not entries:
        raise ValueError(f"VAMI table in {raw_path} had no data rows")

    first_year, first_month, _ = entries[0]
    base_year, base_month = (first_year - 1, 12) if first_month == 1 else (first_year, first_month - 1)

    rows = [(f"{base_month:02d}/{base_year:04d}", 1000.0)]
    rows += [(f"{m:02d}/{y:04d}", price) for y, m, price in entries]
    return rows


# ── shared writer ─────────────────────────────────────────────────────────────

def _write_standard(
    ticker: str,
    rows: list[tuple[str, float]],
    ter_map: dict[str, float],
    out_dir: str,
    source_label: str,
) -> None:
    """Apply TER and write the standard CSV for *ticker*.

    A row's return is the return REALISED DURING that row's own month —
    ``price[i] / price[i-1] - 1`` — so the FIRST row (no prior price) gets
    an empty return, not the last. See the module docstring for why this
    matters (date filters, "which month was the crash" labelling).
    """
    if not rows:
        print(f"[WARN] No data for {ticker}, skipping.")
        return

    annual_ter_pct    = ter_map[ticker]
    annual_ter_dec    = annual_ter_pct / 100.0
    monthly_ter_factor = (1.0 + annual_ter_dec) ** (1.0 / 12.0)

    out_rows = []
    n = len(rows)
    for i, (date_str, price) in enumerate(rows):
        if i == 0:
            out_rows.append({
                "month_year":             date_str,
                "price_eur":              round(price, 6),
                "month_return":           "",
                "month_return_after_TER": "",
            })
        else:
            prev_price = rows[i - 1][1]
            gross_ret  = price / prev_price - 1.0
            net_ret    = (1.0 + gross_ret) / monthly_ter_factor - 1.0
            out_rows.append({
                "month_year":             date_str,
                "price_eur":              round(price, 6),
                "month_return":           round(gross_ret, 8),
                "month_return_after_TER": round(net_ret, 8),
            })

    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, f"{ticker}.csv")
    fieldnames = ["month_year", "price_eur", "month_return", "month_return_after_TER"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"[OK] {ticker} ({source_label}): {n} rows → {out_path}  "
        f"(TER={annual_ter_pct}% p.a., monthly factor={monthly_ter_factor:.8f})"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(cfg.STANDARD_DIR, exist_ok=True)

    # ── load TER table ────────────────────────────────────────────────────
    ter_map: dict[str, float] = {}
    with open(cfg.TER_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            ticker = row[0].strip()
            try:
                ter_map[ticker] = float(row[1].strip())
            except ValueError:
                print(f"  [WARN] cannot parse TER for ticker '{ticker}', skipping")

    print(f"Loaded TER table: {ter_map}\n")

    # ── 1. raw_curvo ─────────────────────────────────────────────────────
    print("=== raw_curvo ===")
    for fname in sorted(os.listdir(cfg.RAW_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        ticker   = os.path.splitext(fname)[0].upper()
        raw_path = os.path.join(cfg.RAW_DIR, fname)
        if ticker not in ter_map:
            print(f"[WARN] No TER found for {ticker} (raw_curvo), skipping file.")
            continue
        rows = _parse_curvo(raw_path)
        _write_standard(ticker, rows, ter_map, cfg.STANDARD_DIR, "curvo")

    # ── 2. raw_msci ──────────────────────────────────────────────────────
    print("\n=== raw_msci ===")
    msci_dir = cfg.RAW_MSCI_DIR
    if not os.path.isdir(msci_dir):
        print(f"[INFO] {msci_dir} not found, skipping MSCI sources.")
    else:
        for fname in sorted(os.listdir(msci_dir)):
            if not fname.lower().endswith(".csv"):
                continue
            ticker   = os.path.splitext(fname)[0].upper()
            raw_path = os.path.join(msci_dir, fname)
            if ticker not in ter_map:
                print(f"[WARN] No TER found for {ticker} (raw_msci), skipping file.")
                continue
            rows = _parse_msci(raw_path)
            _write_standard(ticker, rows, ter_map, cfg.STANDARD_DIR, "msci")

    # ── 3. BTOP50 (VAMI table, single dedicated source file) ───────────────
    print("\n=== BTOP50 ===")
    btop50_ticker = "BTOP50"
    btop50_path = os.path.join(cfg.DATA_DIR, "BTOP50_Index_historical_data 3.csv")
    if not os.path.exists(btop50_path):
        print(f"[INFO] {btop50_path} not found, skipping BTOP50.")
    elif btop50_ticker not in ter_map:
        print(f"[WARN] No TER found for {btop50_ticker}, skipping file.")
    else:
        rows = _parse_btop50_vami(btop50_path)
        _write_standard(btop50_ticker, rows, ter_map, cfg.STANDARD_DIR, "btop50-vami")

    print("\nDone.")


if __name__ == "__main__":
    main()
