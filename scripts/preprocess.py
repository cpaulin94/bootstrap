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

Both sources are normalised to the same output schema:
  month_year, price_eur, month_return, month_return_after_TER
where ``month_year`` is stored as ``MM/YYYY``.

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


# ── shared writer ─────────────────────────────────────────────────────────────

def _write_standard(
    ticker: str,
    rows: list[tuple[str, float]],
    ter_map: dict[str, float],
    out_dir: str,
    source_label: str,
) -> None:
    """Apply TER and write the standard CSV for *ticker*."""
    if not rows:
        print(f"[WARN] No data for {ticker}, skipping.")
        return

    annual_ter_pct    = ter_map[ticker]
    annual_ter_dec    = annual_ter_pct / 100.0
    monthly_ter_factor = (1.0 + annual_ter_dec) ** (1.0 / 12.0)

    out_rows = []
    n = len(rows)
    for i, (date_str, price) in enumerate(rows):
        if i < n - 1:
            next_price = rows[i + 1][1]
            gross_ret  = next_price / price - 1.0
            net_ret    = (1.0 + gross_ret) / monthly_ter_factor - 1.0
            out_rows.append({
                "month_year":             date_str,
                "price_eur":              round(price, 6),
                "month_return":           round(gross_ret, 8),
                "month_return_after_TER": round(net_ret, 8),
            })
        else:
            out_rows.append({
                "month_year":             date_str,
                "price_eur":              round(price, 6),
                "month_return":           "",
                "month_return_after_TER": "",
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

    print("\nDone.")


if __name__ == "__main__":
    main()
