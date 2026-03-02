#!/usr/bin/env python3
"""
Preprocess raw Curvo CSV exports into standardised monthly-return CSVs.

Usage:  python scripts/preprocess.py
"""

from __future__ import annotations

import csv
import os
import sys

# ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import config as cfg


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

    # ── process each raw file ─────────────────────────────────────────────
    for fname in sorted(os.listdir(cfg.RAW_DIR)):
        if not fname.lower().endswith(".csv"):
            continue

        ticker = os.path.splitext(fname)[0].upper()
        raw_path = os.path.join(cfg.RAW_DIR, fname)

        if ticker not in ter_map:
            print(f"[WARN] No TER found for {ticker}, skipping file.")
            continue

        annual_ter_pct = ter_map[ticker]
        annual_ter_dec = annual_ter_pct / 100.0
        monthly_ter_factor = (1.0 + annual_ter_dec) ** (1.0 / 12.0)

        rows: list[tuple[str, float]] = []
        with open(raw_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                if len(line) < 2:
                    continue
                date_str = line[0].strip()
                try:
                    price = float(line[1].strip())
                except ValueError:
                    continue
                rows.append((date_str, price))

        if not rows:
            print(f"[WARN] No data in {fname}, skipping.")
            continue

        out_rows = []
        n = len(rows)
        for i, (date_str, price) in enumerate(rows):
            if i < n - 1:
                next_price = rows[i + 1][1]
                gross_ret = next_price / price - 1.0
                net_ret = (1.0 + gross_ret) / monthly_ter_factor - 1.0
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

        out_path = os.path.join(cfg.STANDARD_DIR, f"{ticker}.csv")
        fieldnames = ["month_year", "price_eur", "month_return", "month_return_after_TER"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"[OK] {ticker}: {n} rows → {out_path}  "
              f"(TER={annual_ter_pct}% p.a., monthly factor={monthly_ter_factor:.8f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
