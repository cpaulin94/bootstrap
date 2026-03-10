#!/usr/bin/env python3
"""
Bootstrap Portfolio Analyser — Tkinter GUI
===========================================
A production-quality desktop application wrapping the bootstrap codebase
into a multi-panel interface with three sections:

  1. Portfolio Builder
  2. Portfolio Space Explorer (multi-bootstrap)
  3. Single Bootstrap Analysis

Launch:
    python gui.py
"""

from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
import tkinter as tk
import http.server
import socketserver
import webbrowser
from multiprocessing import cpu_count
from tkinter import ttk, messagebox, simpledialog
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import plotly.graph_objects as go

# ── project imports ───────────────────────────────────────────────────────────
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from engine import config as cfg
from engine.data import (
    _load_returns,
    _apply_date_filter,
    _parse_month_year,
    load_all_returns,
    load_portfolio_csv,
)
from engine.metrics import compute_metrics, shannon_entropy
from engine.pareto import compute_pareto
from engine.runner import (
    run_bootstrap,
    run_bootstrap_preloaded,
    run_multi_streaming,
)
from engine.search import load_search_space
from engine.simulation import simulate


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

BG = "#F5F5F0"
PANEL_BG = "#FFFFFF"
TEXT = "#1A1A1A"
TEXT_SEC = "#888888"
ACCENT = "#1A1A1A"
POSITIVE = "#2D6A4F"
WARNING = "#C0392B"
CRIMSON = "#C0392B"
SCATTER_DOT = "#4A4A4A"
GRID_CLR = "#E0E0DA"
BORDER = "#DADAD4"
HOVER_BG = "#1A1A1A"

FONT_FAMILY = ("Helvetica Neue", "Helvetica", "Arial Narrow", "Arial")
FONT_LIGHT = (FONT_FAMILY[0], 10)
FONT_REG = (FONT_FAMILY[0], 11)
FONT_MED = (FONT_FAMILY[0], 12, "bold")
FONT_LABEL = (FONT_FAMILY[0], 9)
FONT_LABEL_UP = (FONT_FAMILY[0], 8)
FONT_MONO = ("Menlo", "Courier New", "Courier")
FONT_SMALL = (FONT_FAMILY[0], 9)

PAD = 6

PLOT_DPI = 150  # Higher DPI for crisp plots


def _label_text(s: str) -> str:
    """Convert label to uppercase tracked style."""
    return s.upper()


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_style(ax: plt.Axes, title: str = ""):
    """Apply Swiss-scientific style to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(TEXT)
    ax.spines["left"].set_linewidth(0.5)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(
        direction="out", length=3, width=0.5, colors=TEXT, labelsize=9
    )
    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.8)
    if title:
        ax.set_title(title, fontsize=10, fontweight="medium", loc="left", color=TEXT)


def make_figure(nrows=1, ncols=1, figsize=(8, 5)):
    """Create a Figure with the standard facecolor."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=PLOT_DPI)
    fig.set_facecolor(BG)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.12)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_assets() -> list[dict]:
    """Enumerate assets in data/standard/, read date ranges."""
    assets = []
    for fname in sorted(os.listdir(cfg.STANDARD_DIR)):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "")
        path = os.path.join(cfg.STANDARD_DIR, fname)
        first_date = last_date = None
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    first_date = rows[0]["month_year"].strip()
                    last_date = rows[-1]["month_year"].strip()
        except Exception:
            pass
        assets.append(
            {"ticker": ticker, "first_date": first_date or "?", "last_date": last_date or "?"}
        )
    return assets


def compute_date_intersection(tickers: list[str]) -> tuple[str, str]:
    """Compute the intersection of date ranges for selected tickers."""
    if not tickers:
        return ("", "")
    latest_start = (0, 0)
    earliest_end = (9999, 12)
    for t in tickers:
        path = os.path.join(cfg.STANDARD_DIR, f"{t}.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    s = _parse_month_year(rows[0]["month_year"])
                    e = _parse_month_year(rows[-1]["month_year"])
                    if s > latest_start:
                        latest_start = s
                    if e < earliest_end:
                        earliest_end = e
        except Exception:
            pass
    if latest_start > earliest_end:
        return ("N/A", "N/A")
    return (
        f"{latest_start[0]:04d}-{latest_start[1]:02d}",
        f"{earliest_end[0]:04d}-{earliest_end[1]:02d}",
    )


def load_asset_returns_with_dates(ticker: str):
    """Load returns and dates for a single ticker."""
    dates, ret = _load_returns(ticker, cfg.USE_AFTER_TER_RETURNS)
    return dates, ret


def _clean_portfolio(portfolio: dict) -> dict:
    """Remove zero-weight assets from portfolio.

    Root cause fix for NumPy matmul warnings: zero-weight entries cause
    inf*0 or nan*0 inside BLAS routines, triggering spurious
    'divide by zero / overflow / invalid value in matmul' warnings.
    Filtering them out is mathematically equivalent and eliminates
    the warnings at the source.
    """
    return {t: w for t, w in portfolio.items() if w > 1e-9}


def _build_metric_list() -> list[str]:
    """Build the metric dropdown list dynamically from engine config.

    This ensures names always match what `compute_metrics` produces,
    regardless of BAD_PERCENTILE or VOLATILITY_WINDOWS settings.
    """
    bp = str(cfg.BAD_PERCENTILE)
    return (
        [f"annualised_return_p{p}" for p in cfg.RETURN_PERCENTILES]
        + [f"volatility_{w}y" for w in cfg.VOLATILITY_WINDOWS]
        + [
            f"max_dd_depth_p{bp}",
            f"max_dd_length_months_p{bp}",
            f"mda_months_p{bp}",
        ]
        + ["shannon_entropy"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioLibrary:
    """In-memory portfolio library with optional JSON persistence."""

    _FILE = os.path.join(BASE_DIR, ".portfolio_library.json")

    def __init__(self):
        self.portfolios: dict[str, dict[str, float]] = {}
        self._callbacks: list = []
        self._load()

    def _load(self):
        if os.path.exists(self._FILE):
            try:
                with open(self._FILE, "r") as f:
                    self.portfolios = json.load(f)
            except Exception:
                self.portfolios = {}

    def _save(self):
        try:
            with open(self._FILE, "w") as f:
                json.dump(self.portfolios, f, indent=2)
        except Exception:
            pass

    def add(self, name: str, weights: dict[str, float]):
        self.portfolios[name] = weights
        self._save()
        self._notify()

    def delete(self, name: str):
        self.portfolios.pop(name, None)
        self._save()
        self._notify()

    def rename(self, old: str, new: str):
        if old in self.portfolios:
            self.portfolios[new] = self.portfolios.pop(old)
            self._save()
            self._notify()

    def names(self) -> list[str]:
        return list(self.portfolios.keys())

    def get(self, name: str) -> dict[str, float]:
        return self.portfolios.get(name, {})

    def on_change(self, cb):
        self._callbacks.append(cb)

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# STYLED WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class StyledButton(tk.Button):
    """Rectangular, 1px border, uppercase — inverts on hover."""

    def __init__(self, parent, text="", command=None, **kw):
        super().__init__(
            parent,
            text=text.upper(),
            command=command,
            font=(FONT_FAMILY[0], 9),
            bg=PANEL_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground=PANEL_BG,
            bd=1,
            relief="solid",
            highlightthickness=0,
            padx=10,
            pady=4,
            cursor="hand2",
            **kw,
        )
        self.bind("<Enter>", lambda e: self.config(bg=ACCENT, fg=PANEL_BG))
        self.bind("<Leave>", lambda e: self.config(bg=PANEL_BG, fg=TEXT))


class FieldLabel(tk.Label):
    """Uppercase label for form fields."""

    def __init__(self, parent, text="", **kw):
        super().__init__(
            parent,
            text=_label_text(text),
            font=(FONT_FAMILY[0], 8),
            fg=TEXT_SEC,
            bg=kw.pop("bg", PANEL_BG),
            anchor="w",
            **kw,
        )


class NumericEntry(tk.Entry):
    """Entry that only accepts numeric input."""

    def __init__(self, parent, **kw):
        super().__init__(
            parent,
            font=(FONT_FAMILY[0], 10),
            bg=PANEL_BG,
            fg=TEXT,
            bd=1,
            relief="solid",
            highlightthickness=0,
            **kw,
        )


class ErrorLabel(tk.Label):
    """Inline error message in WARNING color."""

    def __init__(self, parent, **kw):
        super().__init__(
            parent,
            text="",
            font=(FONT_FAMILY[0], 8),
            fg=WARNING,
            bg=kw.pop("bg", PANEL_BG),
            anchor="w",
            **kw,
        )

    def show(self, msg: str):
        self.config(text=msg)

    def clear(self):
        self.config(text="")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioBuilderSection(tk.Frame):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.library = library
        self.available_assets = get_available_assets()
        self.current_assets: list[dict] = []  # [{ticker, weight_var}]

        self._build_ui()
        library.on_change(self._refresh_library_list)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(PAD, 1))

        sep = tk.Frame(self, bg=BORDER, width=1)
        sep.pack(side="left", fill="y")

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(1, PAD))

        self._build_asset_table(left)
        self._build_portfolio_editor(left)
        self._build_heatmap(right)
        self._build_library(right)

    def _build_asset_table(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(PAD, 2))

        FieldLabel(frame, text="Available Assets", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )

        cols = ("ticker", "first_date", "last_date")
        self.asset_tree = ttk.Treeview(
            frame, columns=cols, show="headings", height=8, selectmode="browse"
        )
        self.asset_tree.heading("ticker", text="ASSET")
        self.asset_tree.heading("first_date", text="FROM")
        self.asset_tree.heading("last_date", text="TO")
        self.asset_tree.column("ticker", width=80)
        self.asset_tree.column("first_date", width=90)
        self.asset_tree.column("last_date", width=90)

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.asset_tree.yview)
        self.asset_tree.configure(yscrollcommand=sb.set)

        self.asset_tree.pack(side="left", fill="both", expand=True, padx=(PAD, 0), pady=PAD)
        sb.pack(side="right", fill="y", pady=PAD, padx=(0, PAD))

        for a in self.available_assets:
            self.asset_tree.insert("", "end", values=(a["ticker"], a["first_date"], a["last_date"]))
        self.asset_tree.bind("<Double-1>", self._on_asset_double_click)

        btn_frame = tk.Frame(frame, bg=PANEL_BG)
        btn_frame.pack(fill="x", padx=PAD, pady=(0, PAD))
        StyledButton(btn_frame, text="Add Asset", command=self._add_selected_asset).pack(
            side="left"
        )

    def _build_portfolio_editor(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(2, PAD))

        top = tk.Frame(frame, bg=PANEL_BG)
        top.pack(fill="x", padx=PAD, pady=(PAD, 0))
        FieldLabel(top, text="Current Portfolio", bg=PANEL_BG).pack(side="left")
        self.remaining_label = tk.Label(
            top, text="REMAINING: 100.0%", font=(FONT_FAMILY[0], 9, "bold"),
            fg=WARNING, bg=PANEL_BG
        )
        self.remaining_label.pack(side="right")

        self.date_range_label = tk.Label(
            frame, text="GLOBAL DATE RANGE: —",
            font=(FONT_FAMILY[0], 8), fg=TEXT_SEC, bg=PANEL_BG, anchor="w"
        )
        self.date_range_label.pack(fill="x", padx=PAD, pady=(2, 4))

        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0, height=160)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.portfolio_inner = tk.Frame(canvas, bg=PANEL_BG)
        self.portfolio_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.portfolio_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=PAD)
        scrollbar.pack(side="right", fill="y", padx=(0, PAD))

        self.portfolio_error = ErrorLabel(frame, bg=PANEL_BG)
        self.portfolio_error.pack(fill="x", padx=PAD, pady=(0, 2))

        btn_row = tk.Frame(frame, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=PAD, pady=(0, PAD))
        self.name_entry = tk.Entry(
            btn_row, font=(FONT_FAMILY[0], 10), width=18, bd=1, relief="solid",
            bg=PANEL_BG, fg=TEXT
        )
        self.name_entry.insert(0, "My Portfolio")
        self.name_entry.pack(side="left", padx=(0, 4))
        StyledButton(btn_row, text="Save to Library", command=self._save_portfolio).pack(
            side="left", padx=(0, 4)
        )
        StyledButton(btn_row, text="Clear All", command=self._clear_portfolio).pack(side="left")

    def _build_heatmap(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(PAD, 2))

        FieldLabel(frame, text="Correlation Heatmap", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )
        self.heatmap_fig = Figure(figsize=(4, 3.5), facecolor=BG, dpi=PLOT_DPI)
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=frame)
        self.heatmap_canvas.get_tk_widget().pack(fill="both", expand=True, padx=PAD, pady=PAD)
        self._draw_empty_heatmap()

    def _build_library(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(2, PAD))

        FieldLabel(frame, text="Portfolio Library", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )
        self.library_listbox = tk.Listbox(
            frame, font=(FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
            selectmode="browse", bd=0, highlightthickness=0, height=6
        )
        self.library_listbox.pack(fill="both", expand=True, padx=PAD, pady=2)

        btn_row = tk.Frame(frame, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=PAD, pady=(0, PAD))
        StyledButton(btn_row, text="Delete", command=self._delete_library_entry).pack(
            side="left", padx=(0, 4)
        )
        StyledButton(btn_row, text="Rename", command=self._rename_library_entry).pack(
            side="left", padx=(0, 4)
        )
        StyledButton(btn_row, text="Load into Editor", command=self._load_library_entry).pack(
            side="left"
        )
        self._refresh_library_list()

    # ── Actions ───────────────────────────────────────────────────────────

    def _on_asset_double_click(self, event):
        self._add_selected_asset()

    def _add_selected_asset(self):
        sel = self.asset_tree.selection()
        if not sel:
            return
        values = self.asset_tree.item(sel[0], "values")
        ticker = values[0]
        if any(a["ticker"] == ticker for a in self.current_assets):
            return
        self._add_asset_row(ticker, 0.0)
        self._update_portfolio_state()

    def _add_asset_row(self, ticker: str, weight: float):
        row_frame = tk.Frame(self.portfolio_inner, bg=PANEL_BG)
        row_frame.pack(fill="x", padx=2, pady=1)

        tk.Label(
            row_frame, text=ticker, font=(FONT_FAMILY[0], 10, "bold"),
            fg=TEXT, bg=PANEL_BG, width=8, anchor="w"
        ).pack(side="left")

        var = tk.StringVar(value=f"{weight:.1f}")
        entry = tk.Entry(
            row_frame, textvariable=var, font=(FONT_MONO[0], 10), width=8,
            bd=1, relief="solid", bg=PANEL_BG, fg=TEXT
        )
        entry.pack(side="left", padx=4)
        tk.Label(row_frame, text="%", font=(FONT_FAMILY[0], 9), fg=TEXT_SEC,
                 bg=PANEL_BG).pack(side="left")
        var.trace_add("write", lambda *_: self._update_portfolio_state())

        def remove(t=ticker, f=row_frame):
            self.current_assets = [a for a in self.current_assets if a["ticker"] != t]
            f.destroy()
            self._update_portfolio_state()

        StyledButton(row_frame, text="×", command=remove).pack(side="right", padx=2)
        self.current_assets.append({"ticker": ticker, "weight_var": var, "frame": row_frame})

    def _update_portfolio_state(self):
        total = 0.0
        for a in self.current_assets:
            try:
                total += float(a["weight_var"].get())
            except ValueError:
                pass
        remaining = 100.0 - total
        color = POSITIVE if abs(remaining) < 0.01 else WARNING
        self.remaining_label.config(text=f"REMAINING: {remaining:+.1f}%", fg=color)

        tickers = [a["ticker"] for a in self.current_assets]
        if tickers:
            ds, de = compute_date_intersection(tickers)
            self.date_range_label.config(text=f"GLOBAL DATE RANGE: {ds} → {de}")
        else:
            self.date_range_label.config(text="GLOBAL DATE RANGE: —")
        self._update_heatmap()

    def _get_current_weights(self) -> dict[str, float]:
        weights = {}
        for a in self.current_assets:
            try:
                w = float(a["weight_var"].get())
            except ValueError:
                w = 0.0
            weights[a["ticker"]] = w / 100.0
        return weights

    def _save_portfolio(self):
        self.portfolio_error.clear()
        weights = self._get_current_weights()
        if not weights:
            self.portfolio_error.show("No assets added.")
            return
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            self.portfolio_error.show(f"Weights sum to {total*100:.1f}%, must be 100%.")
            return
        name = self.name_entry.get().strip()
        if not name:
            self.portfolio_error.show("Enter a portfolio name.")
            return
        self.library.add(name, weights)
        self.portfolio_error.clear()

    def _clear_portfolio(self):
        for a in self.current_assets:
            a["frame"].destroy()
        self.current_assets.clear()
        self._update_portfolio_state()

    def _draw_empty_heatmap(self):
        self.heatmap_ax.clear()
        self.heatmap_ax.set_facecolor(PANEL_BG)
        self.heatmap_ax.text(
            0.5, 0.5, "Add assets to see\ncorrelation heatmap",
            ha="center", va="center", fontsize=9, color=TEXT_SEC,
            transform=self.heatmap_ax.transAxes,
        )
        self.heatmap_ax.set_xticks([])
        self.heatmap_ax.set_yticks([])
        for sp in self.heatmap_ax.spines.values():
            sp.set_visible(False)
        self.heatmap_canvas.draw_idle()

    def _update_heatmap(self):
        tickers = [a["ticker"] for a in self.current_assets]
        if len(tickers) < 2:
            self._draw_empty_heatmap()
            return
        try:
            ds, de = compute_date_intersection(tickers)
            all_ret = {}
            for t in tickers:
                dates, ret = _load_returns(t, cfg.USE_AFTER_TER_RETURNS)
                filtered = _apply_date_filter(dates, ret, ds, de)
                all_ret[t] = filtered

            min_len = min(len(v) for v in all_ret.values())
            matrix = np.column_stack([all_ret[t][-min_len:] for t in tickers])
            corr = np.corrcoef(matrix.T)
            cmap = LinearSegmentedColormap.from_list(
                "corr", [WARNING, "#FFFFFF", POSITIVE], N=256
            )
            self.heatmap_ax.clear()
            im = self.heatmap_ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            n = len(tickers)
            self.heatmap_ax.set_xticks(range(n))
            self.heatmap_ax.set_yticks(range(n))
            self.heatmap_ax.set_xticklabels(tickers, fontsize=8, rotation=45, ha="right")
            self.heatmap_ax.set_yticklabels(tickers, fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = corr[i, j]
                    color = PANEL_BG if abs(val) > 0.7 else TEXT
                    self.heatmap_ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color
                    )
            for sp in self.heatmap_ax.spines.values():
                sp.set_visible(False)
            self.heatmap_fig.tight_layout()
            self.heatmap_canvas.draw_idle()
        except Exception:
            self._draw_empty_heatmap()

    def _refresh_library_list(self):
        self.library_listbox.delete(0, "end")
        for name in self.library.names():
            weights = self.library.get(name)
            summary = ", ".join(f"{t}:{w:.0%}" for t, w in sorted(weights.items()) if w > 0)
            self.library_listbox.insert("end", f"{name}  [{summary}]")

    def _delete_library_entry(self):
        sel = self.library_listbox.curselection()
        if not sel:
            return
        name = self.library.names()[sel[0]]
        self.library.delete(name)

    def _rename_library_entry(self):
        sel = self.library_listbox.curselection()
        if not sel:
            return
        old_name = self.library.names()[sel[0]]
        new_name = simpledialog.askstring("Rename", "New name:", initialvalue=old_name)
        if new_name and new_name.strip():
            self.library.rename(old_name, new_name.strip())

    def _load_library_entry(self):
        sel = self.library_listbox.curselection()
        if not sel:
            return
        name = self.library.names()[sel[0]]
        weights = self.library.get(name)
        self._clear_portfolio()
        for t, w in sorted(weights.items()):
            self._add_asset_row(t, w * 100.0)
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, name)
        self._update_portfolio_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO SPACE EXPLORER (MULTI-BOOTSTRAP)
# ═══════════════════════════════════════════════════════════════════════════════

class SpaceExplorerSection(tk.Frame):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.library = library
        self.result_queue = queue.Queue()
        self.all_results = []
        self.sorted_tickers = []
        self.running = False
        self._stop_event = threading.Event()
        self._overlay_cache: dict[str, dict] = {}  # name -> {metrics, params}
        self._overlay_threads: dict[str, threading.Thread] = {}
        self._last_run_params: dict | None = None
        self._chart_path: str | None = None
        self._chart_opened = False
        self._destroyed = False
        self._click_queue: queue.Queue = queue.Queue()
        self._click_port: int | None = None
        self._click_server = None

        self._build_ui()
        library.on_change(self._refresh_overlay_panel)
        library.on_change(self._refresh_space_library_selector)
        self._start_click_server()
        self.after(500, self._poll_clicks)

    def destroy(self):
        self._destroyed = True
        if self._click_server:
            try:
                self._click_server.shutdown()
            except Exception:
                pass
        super().destroy()

    # ── Click-to-library HTTP server ──────────────────────────────────────

    def _start_click_server(self):
        """Start a tiny local HTTP server that:
          - serves scatter.html from results/ on GET requests
          - receives POST /click from the chart's JS and queues portfolio data
        This avoids CORS since the chart is opened via http://localhost:PORT/
        instead of file://, making the fetch() call same-origin.
        """
        results_dir = cfg.RESULTS_DIR
        click_queue = self._click_queue

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=results_dir, **kwargs)

            def do_OPTIONS(self):
                self.send_response(200)
                self._cors()
                self.end_headers()

            def do_POST(self):
                if self.path == "/click":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        click_queue.put(data)
                    except Exception:
                        pass
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}')
                else:
                    self.send_error(404)

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def log_message(self, *args):
                pass  # suppress console noise

        try:
            srv = socketserver.TCPServer(("localhost", 0), Handler)
            srv.allow_reuse_address = True
            self._click_port = srv.server_address[1]
            self._click_server = srv
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
        except Exception:
            self._click_port = None

    def _poll_clicks(self):
        """Poll the click queue and handle any portfolio clicks."""
        if self._destroyed:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        try:
            while True:
                data = self._click_queue.get_nowait()
                portfolio = data.get("portfolio", {})
                if portfolio:
                    self._handle_portfolio_click(portfolio)
        except queue.Empty:
            pass
        self.after(300, self._poll_clicks)

    def _handle_portfolio_click(self, portfolio: dict):
        """Prompt the user to name and save a clicked portfolio to the library."""
        # Build clean float-keyed weights dict from JS customdata
        weights = {
            k: float(v)
            for k, v in portfolio.items()
            if isinstance(v, (int, float)) and float(v) > 1e-9
        }
        if not weights:
            return
        preview_lines = [f"{t}: {w:.1%}" for t, w in sorted(weights.items())]
        preview = "\n".join(preview_lines[:8])
        if len(preview_lines) > 8:
            preview += f"\n  …+{len(preview_lines) - 8} more"
        name = simpledialog.askstring(
            "Add to Library",
            f"Portfolio:\n{preview}\n\nEnter a name:",
            initialvalue="Selected Portfolio",
            parent=self,
        )
        if name and name.strip():
            self.library.add(name.strip(), weights)
            self._log(f"Saved '{name.strip()}' to library ({len(weights)} assets)")

    def _build_ui(self):
        left = tk.Frame(self, bg=BG, width=340)
        left.pack(side="left", fill="y", padx=(PAD, 1))
        left.pack_propagate(False)

        sep = tk.Frame(self, bg=BORDER, width=1)
        sep.pack(side="left", fill="y")

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(1, PAD))

        self._build_config_panel(left)
        self._build_chart_panel(right)

    def _build_config_panel(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=320)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Search mode
        mode_frame = self._make_section(inner, "Search Mode")
        self.search_mode = tk.StringVar(value="random")
        tk.Radiobutton(
            mode_frame, text="Random Search", variable=self.search_mode, value="random",
            bg=BG, fg=TEXT, font=FONT_SMALL, activebackground=BG, selectcolor=BG,
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame, text="Grid Search", variable=self.search_mode, value="grid",
            bg=BG, fg=TEXT, font=FONT_SMALL, activebackground=BG, selectcolor=BG,
        ).pack(anchor="w")

        # Bootstrap params — defaults from engine config
        params_frame = self._make_section(inner, "Bootstrap Parameters")
        self.n_sim_var = self._add_param(params_frame, "N Simulations",
                                         str(cfg.N_SIMULATIONS))
        self.horizon_var = self._add_param(params_frame, "Horizon (Years)",
                                           str(cfg.HORIZON_YEARS))
        self.block_var = self._add_param(params_frame, "Block Size",
                                         str(cfg.BLOCK_SIZE))
        self.seed_var = self._add_param(params_frame, "Random Seed", "")

        # Multi-bootstrap params
        multi_frame = self._make_section(inner, "Search Parameters")
        self.n_portfolios_var = self._add_param(multi_frame, "N Portfolios",
                                                str(cfg.N_PORTFOLIOS))
        self.grid_step_var = self._add_param(multi_frame, "Grid Step",
                                             str(cfg.GRID_STEP))
        self.flush_interval_var = self._add_param(multi_frame, "Flush Every N", "500")
        self.n_jobs_var = self._add_param(multi_frame, "Parallel Jobs (-1=all)", "-1")

        # Search space
        space_frame = self._make_section(inner, "Search Space")
        import_row = tk.Frame(space_frame, bg=BG)
        import_row.pack(fill="x", pady=(0, 4))
        FieldLabel(import_row, text="Portfolio Template", bg=BG).pack(side="left", padx=(0, 4))
        self.space_library_var = tk.StringVar(value="")
        self.space_library_combo = ttk.Combobox(
            import_row,
            textvariable=self.space_library_var,
            state="readonly",
            width=20,
        )
        self.space_library_combo.pack(side="left", padx=(0, 4))
        StyledButton(
            import_row,
            text="Load",
            command=self._load_library_into_search_space,
        ).pack(side="left")

        delta_row = tk.Frame(space_frame, bg=BG)
        delta_row.pack(fill="x", pady=(0, 4))
        FieldLabel(delta_row, text="Delta (abs)", bg=BG).pack(side="left", padx=(0, 4))
        self.space_delta_var = tk.StringVar(value="0.02")
        tk.Entry(
            delta_row,
            textvariable=self.space_delta_var,
            width=6,
            font=(FONT_MONO[0], 9),
            bd=1,
            relief="solid",
        ).pack(side="left", padx=(0, 4))
        StyledButton(
            delta_row,
            text="Load ±Δ",
            command=self._load_library_into_search_space_with_delta,
        ).pack(side="left")

        self.space_entries = {}
        for a in get_available_assets():
            t = a["ticker"]
            row = tk.Frame(space_frame, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=t, font=(FONT_FAMILY[0], 9, "bold"), fg=TEXT,
                     bg=BG, width=6, anchor="w").pack(side="left")
            lo_var = tk.StringVar(value="0.0")
            hi_var = tk.StringVar(value="0.3")
            tk.Entry(row, textvariable=lo_var, width=5, font=(FONT_MONO[0], 9),
                     bd=1, relief="solid").pack(side="left", padx=1)
            tk.Label(row, text="–", bg=BG, fg=TEXT_SEC).pack(side="left")
            tk.Entry(row, textvariable=hi_var, width=5, font=(FONT_MONO[0], 9),
                     bd=1, relief="solid").pack(side="left", padx=1)
            self.space_entries[t] = (lo_var, hi_var)

        self._refresh_space_library_selector()

        self._load_search_csv_defaults()

        # Cutoff filters — two-line layout per filter row for readability
        cutoff_frame = self._make_section(inner, "Metric Cutoff Filters")
        self.cutoff_rows: list[dict] = []
        self.cutoff_container = tk.Frame(cutoff_frame, bg=BG)
        self.cutoff_container.pack(fill="x")
        StyledButton(cutoff_frame, text="+ Add Filter", command=self._add_cutoff_row).pack(
            anchor="w", pady=2
        )

        # Date range
        date_frame = self._make_section(inner, "Date Range")
        self.date_start_var = tk.StringVar(value="")
        self.date_end_var = tk.StringVar(value="")
        FieldLabel(date_frame, text="Start (YYYY-MM)", bg=BG).pack(anchor="w")
        tk.Entry(date_frame, textvariable=self.date_start_var, font=(FONT_FAMILY[0], 10),
                 bd=1, relief="solid", width=12).pack(anchor="w", pady=(0, 4))
        FieldLabel(date_frame, text="End (YYYY-MM)", bg=BG).pack(anchor="w")
        tk.Entry(date_frame, textvariable=self.date_end_var, font=(FONT_FAMILY[0], 10),
                 bd=1, relief="solid", width=12).pack(anchor="w")

        # Overlay selection — checkboxes (compute on toggle)
        overlay_frame = self._make_section(inner, "Current Portfolios Overlay")
        self.overlay_container = tk.Frame(overlay_frame, bg=BG)
        self.overlay_container.pack(fill="x", pady=2)
        self._overlay_vars: dict[str, tk.BooleanVar] = {}
        self._refresh_overlay_panel()

    def _build_chart_panel(self, parent):
        """Right panel: run control, chart axes, progress, log, and chart button."""
        # ── Run control at top ────────────────────────────────────────────
        run_frame = self._make_section(parent, "Run Control")
        self.run_error = ErrorLabel(run_frame, bg=BG)
        self.run_error.pack(fill="x")

        btn_row = tk.Frame(run_frame, bg=BG)
        btn_row.pack(fill="x", pady=(2, 0))
        self.run_btn = StyledButton(btn_row, text="Run", command=self._run_search)
        self.run_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = StyledButton(btn_row, text="Stop", command=self._stop_search)
        self.stop_btn.pack(side="left", padx=(0, 4))
        self.stop_btn.config(state="disabled")
        self.reset_btn = StyledButton(btn_row, text="Reset", command=self._reset_search)
        self.reset_btn.pack(side="left")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            run_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill="x", pady=4)
        self.progress_label = tk.Label(
            run_frame, text="", font=(FONT_FAMILY[0], 10), fg=TEXT_SEC, bg=BG
        )
        self.progress_label.pack(fill="x")

        # ── Chart axes (reactive) ────────────────────────────────────────
        axis_frame = self._make_section(parent, "Chart Axes")
        all_metrics = _build_metric_list()
        self.available_metrics = all_metrics

        ax_row = tk.Frame(axis_frame, bg=BG)
        ax_row.pack(fill="x")

        x_col = tk.Frame(ax_row, bg=BG)
        x_col.pack(side="left", fill="x", expand=True, padx=(0, 4))
        FieldLabel(x_col, text="X Axis", bg=BG).pack(anchor="w")
        self.x_metric_var = tk.StringVar(value="annualised_return_p50")
        ttk.Combobox(
            x_col, textvariable=self.x_metric_var, values=all_metrics,
            state="readonly", width=28
        ).pack(fill="x", pady=(0, 4))

        y_col = tk.Frame(ax_row, bg=BG)
        y_col.pack(side="left", fill="x", expand=True)
        FieldLabel(y_col, text="Y Axis", bg=BG).pack(anchor="w")
        self.y_metric_var = tk.StringVar(value="annualised_return_p1")
        ttk.Combobox(
            y_col, textvariable=self.y_metric_var, values=all_metrics,
            state="readonly", width=28
        ).pack(fill="x", pady=(0, 4))

        # Reactive: regenerate chart when axes change
        self.x_metric_var.trace_add("write", lambda *_: self._regenerate_chart())
        self.y_metric_var.trace_add("write", lambda *_: self._regenerate_chart())

        # ── Chart button ──────────────────────────────────────────────────
        chart_btn_frame = tk.Frame(axis_frame, bg=BG)
        chart_btn_frame.pack(fill="x", pady=(4, 0))
        self.chart_btn = StyledButton(
            chart_btn_frame, text="Open Interactive Chart",
            command=self._open_chart
        )
        self.chart_btn.pack(side="left", padx=(0, 4))
        self.refresh_btn = StyledButton(
            chart_btn_frame, text="Refresh Chart",
            command=self._regenerate_chart
        )
        self.refresh_btn.pack(side="left")

        self.chart_status = tk.Label(
            axis_frame, text="No results yet", font=(FONT_FAMILY[0], 9),
            fg=TEXT_SEC, bg=BG, anchor="w"
        )
        self.chart_status.pack(fill="x", pady=(4, 0))

        # ── Log panel (expanded) ─────────────────────────────────────────
        log_frame = self._make_section(parent, "Log")
        self.log_text = tk.Text(
            log_frame, font=(FONT_MONO[0], 9), bg=PANEL_BG, fg=TEXT,
            height=30, bd=1, relief="solid", wrap="word", state="normal"
        )
        self.log_text.pack(fill="both", expand=True, pady=2)

    def _make_section(self, parent, title):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="x", padx=PAD, pady=(PAD, 0))
        if title:
            tk.Label(
                frame, text=_label_text(title), font=(FONT_FAMILY[0], 9, "bold"),
                fg=TEXT, bg=BG, anchor="w"
            ).pack(fill="x", pady=(0, 2))
            tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 4))
        return frame

    def _add_param(self, parent, label, default):
        FieldLabel(parent, text=label, bg=BG).pack(anchor="w")
        var = tk.StringVar(value=default)
        tk.Entry(
            parent, textvariable=var, font=(FONT_FAMILY[0], 10), width=14,
            bd=1, relief="solid"
        ).pack(anchor="w", pady=(0, 4))
        return var

    def _load_search_csv_defaults(self):
        try:
            space = load_search_space(cfg.SEARCH_CSV)
            for item in space:
                t = item["ticker"]
                if t in self.space_entries:
                    lo_var, hi_var = self.space_entries[t]
                    lo_var.set(f"{item['lo']:.2f}")
                    hi_var.set(f"{item['hi']:.2f}")
        except Exception:
            pass

    def _refresh_space_library_selector(self):
        names = self.library.names()
        current = self.space_library_var.get().strip()
        self.space_library_combo["values"] = names
        if current in names:
            return
        if names:
            self.space_library_var.set(names[0])
        else:
            self.space_library_var.set("")

    def _load_library_into_search_space(self):
        name = self.space_library_var.get().strip()
        if not name:
            self.run_error.show("No portfolio selected in library template.")
            return

        portfolio = self.library.get(name)
        if not portfolio:
            self.run_error.show("Selected portfolio is empty.")
            return

        self.run_error.clear()
        for ticker, (lo_var, hi_var) in self.space_entries.items():
            weight = float(portfolio.get(ticker, 0.0))
            text = f"{weight:.4f}".rstrip("0").rstrip(".")
            if "." not in text:
                text += ".0"
            lo_var.set(text)
            hi_var.set(text)

        self._log(f"Loaded '{name}' into search space (min=max for all assets).")

    def _load_library_into_search_space_with_delta(self):
        name = self.space_library_var.get().strip()
        if not name:
            self.run_error.show("No portfolio selected in library template.")
            return

        portfolio = self.library.get(name)
        if not portfolio:
            self.run_error.show("Selected portfolio is empty.")
            return

        try:
            delta = float(self.space_delta_var.get().strip())
        except ValueError:
            self.run_error.show("Delta must be a number.")
            return

        if delta < 0:
            self.run_error.show("Delta must be >= 0.")
            return

        self.run_error.clear()
        for ticker, (lo_var, hi_var) in self.space_entries.items():
            weight = float(portfolio.get(ticker, 0.0))
            lo = max(0.0, weight - delta)
            hi = min(1.0, weight + delta)

            lo_text = f"{lo:.4f}".rstrip("0").rstrip(".")
            hi_text = f"{hi:.4f}".rstrip("0").rstrip(".")
            if "." not in lo_text:
                lo_text += ".0"
            if "." not in hi_text:
                hi_text += ".0"
            lo_var.set(lo_text)
            hi_var.set(hi_text)

        self._log(
            f"Loaded '{name}' into search space with delta ±{delta:.4f} (clamped to [0,1])."
        )

    # ── Cutoff filters (two-line layout) ──────────────────────────────────

    def _add_cutoff_row(self):
        """Add a cutoff filter with a two-row layout that fits 320px."""
        outer = tk.Frame(self.cutoff_container, bg=BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="x", pady=2, padx=2)

        # Row 1: metric dropdown (full width)
        row1 = tk.Frame(outer, bg=BG)
        row1.pack(fill="x", padx=4, pady=(4, 0))
        metric_var = tk.StringVar(value=self.available_metrics[0])
        ttk.Combobox(
            row1, textvariable=metric_var, values=self.available_metrics,
            state="readonly", width=35
        ).pack(fill="x")

        # Row 2: operator, value, remove button
        row2 = tk.Frame(outer, bg=BG)
        row2.pack(fill="x", padx=4, pady=(2, 4))

        op_var = tk.StringVar(value=">=")
        ttk.Combobox(
            row2, textvariable=op_var, values=[">=", "<=", ">", "<"],
            state="readonly", width=4
        ).pack(side="left", padx=(0, 4))

        val_var = tk.StringVar(value="0.0")
        tk.Entry(
            row2, textvariable=val_var, font=(FONT_MONO[0], 9), width=10,
            bd=1, relief="solid"
        ).pack(side="left", padx=(0, 4))

        entry = {"metric_var": metric_var, "op_var": op_var, "val_var": val_var, "frame": outer}
        self.cutoff_rows.append(entry)

        def remove(e=entry):
            self.cutoff_rows.remove(e)
            e["frame"].destroy()

        remove_btn = tk.Button(
            row2, text="REMOVE", font=(FONT_FAMILY[0], 8),
            command=remove, bg=WARNING, fg=PANEL_BG, bd=1, relief="solid",
            padx=6, pady=1, cursor="hand2"
        )
        remove_btn.pack(side="right")

    def _get_cutoffs(self) -> list[tuple[str, str, float]]:
        result = []
        for c in self.cutoff_rows:
            try:
                val = float(c["val_var"].get())
                result.append((c["metric_var"].get(), c["op_var"].get(), val))
            except ValueError:
                pass
        return result

    # ── Overlay (checkboxes with background cache) ────────────────────────

    def _refresh_overlay_panel(self):
        """Rebuild overlay checkbox panel from library."""
        for w in self.overlay_container.winfo_children():
            w.destroy()
        self._overlay_vars.clear()
        for name in self.library.names():
            var = tk.BooleanVar(value=name in self._overlay_cache)
            self._overlay_vars[name] = var
            cb = tk.Checkbutton(
                self.overlay_container, text=name,
                variable=var, font=(FONT_FAMILY[0], 9),
                bg=BG, fg=TEXT, activebackground=BG, selectcolor=BG,
                anchor="w",
                command=lambda n=name: self._toggle_overlay(n),
            )
            cb.pack(fill="x", anchor="w")

    def _toggle_overlay(self, name: str):
        """Compute or remove overlay for a portfolio."""
        var = self._overlay_vars.get(name)
        if var is None:
            return
        if var.get():
            params = self._current_overlay_params()
            if params is None:
                return

            cached = self._overlay_cache.get(name)
            if cached and cached.get("params") == params:
                if self.all_results:
                    self._regenerate_chart()
                return

            existing = self._overlay_threads.get(name)
            if existing and existing.is_alive():
                return

            # Compute overlay in background
            self._log(f"Computing overlay for '{name}'...")
            thread = threading.Thread(
                target=self._compute_overlay_worker,
                args=(name, params),
                daemon=True,
            )
            thread.start()
            self._overlay_threads[name] = thread
        else:
            self._overlay_cache.pop(name, None)
            self._overlay_threads.pop(name, None)
            # Regenerate chart to remove overlay
            if self.all_results:
                self._regenerate_chart()

    def _current_overlay_params(self) -> dict | None:
        """Return the effective params overlays must use to match current results."""
        if self._last_run_params and self.all_results:
            return dict(self._last_run_params)
        try:
            return {
                "n_sim": int(self.n_sim_var.get()),
                "horizon_years": int(self.horizon_var.get()),
                "block_size": int(self.block_var.get()),
                "date_start": self.date_start_var.get().strip() or None,
                "date_end": self.date_end_var.get().strip() or None,
            }
        except ValueError:
            self.run_error.show("Invalid simulation parameters for overlay.")
            return None

    def _compute_overlay_worker(self, name: str, params: dict):
        """Background thread: compute overlay metrics for one portfolio."""
        try:
            portfolio = _clean_portfolio(self.library.get(name))
            if not portfolio:
                return
            metrics = run_bootstrap(
                portfolio,
                n_sim=params["n_sim"],
                horizon_years=params["horizon_years"],
                block_size=params["block_size"],
                random_seed=None,
                date_start=params["date_start"],
                date_end=params["date_end"],
            )
            self._overlay_cache[name] = {"metrics": metrics, "params": dict(params)}
            # Signal UI to redraw
            self.result_queue.put(("overlay_done", name))
        except Exception as e:
            self._overlay_cache.pop(name, None)
            self.result_queue.put(("overlay_error", name, str(e)))

    def _recompute_selected_overlays(self):
        """Ensure checked overlays are computed with current run parameters."""
        params = self._current_overlay_params()
        if params is None:
            return
        for name, var in self._overlay_vars.items():
            if not var.get():
                continue
            cached = self._overlay_cache.get(name)
            if cached and cached.get("params") == params:
                continue
            existing = self._overlay_threads.get(name)
            if existing and existing.is_alive():
                continue
            self._log(f"Computing overlay for '{name}'...")
            thread = threading.Thread(
                target=self._compute_overlay_worker,
                args=(name, dict(params)),
                daemon=True,
            )
            thread.start()
            self._overlay_threads[name] = thread

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        """Append timestamped message to the log panel."""
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    # ── Run control ───────────────────────────────────────────────────────

    def _run_search(self):
        self.run_error.clear()
        if self.running:
            return

        try:
            n_sim = int(self.n_sim_var.get())
            horizon = int(self.horizon_var.get())
            block = int(self.block_var.get())
            n_port = int(self.n_portfolios_var.get())
            grid_step = float(self.grid_step_var.get())
            flush_n = int(self.flush_interval_var.get())
        except ValueError as e:
            self.run_error.show(f"Invalid parameter: {e}")
            return

        seed_str = self.seed_var.get().strip()
        seed = int(seed_str) if seed_str else None
        method = self.search_mode.get()
        date_start = self.date_start_var.get().strip() or None
        date_end = self.date_end_var.get().strip() or None
        self._last_run_params = {
            "n_sim": n_sim,
            "horizon_years": horizon,
            "block_size": block,
            "date_start": date_start,
            "date_end": date_end,
        }

        # Build search space from GUI entries
        space = []
        for t, (lo_var, hi_var) in self.space_entries.items():
            try:
                lo = float(lo_var.get())
                hi = float(hi_var.get())
                if hi > 0:
                    space.append({"ticker": t, "lo": lo, "hi": hi})
            except ValueError:
                pass
        if not space:
            self.run_error.show("No assets in search space.")
            return

        n_jobs_str = self.n_jobs_var.get().strip()
        n_jobs = int(n_jobs_str) if n_jobs_str else -1
        n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)

        # Log run parameters
        self._log(f"--- Starting {method} search ---")
        self._log(f"  N_portfolios={n_port}, N_sim={n_sim}")
        self._log(f"  horizon={horizon}y, block={block}, flush={flush_n}")
        if date_start or date_end:
            self._log(f"  date_range={date_start or '?'} → {date_end or '?'}")
        self._log(f"  assets: {', '.join(s['ticker'] for s in space)}")
        self._log(f"  workers: {n_workers} cores")

        self.running = True
        self._stop_event.clear()
        self.all_results = []
        self._chart_opened = False

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)

        thread = threading.Thread(
            target=self._search_worker,
            args=(space, method, n_port, grid_step, n_sim, horizon, block,
                  seed, date_start, date_end, flush_n, n_jobs),
            daemon=True,
        )
        thread.start()
        self.after(200, self._poll_results)

    def _search_worker(self, space, method, n_portfolios, grid_step,
                       n_sim, horizon, block, seed, date_start, date_end,
                       flush_n, n_jobs):
        """Thin wrapper: delegates all heavy work to engine.run_multi_streaming."""

        def on_start(sorted_tickers, total):
            self.result_queue.put(("info", sorted_tickers, total))

        def on_batch(results, n_done, total, speed):
            self.result_queue.put(("batch", results, n_done, total, speed))

        def on_done(n_done, elapsed, avg_speed):
            self.result_queue.put(("done", n_done, elapsed, avg_speed))

        def on_error(msg):
            self.result_queue.put(("error", msg))

        run_multi_streaming(
            space,
            method=method,
            n_portfolios=n_portfolios,
            grid_step=grid_step,
            n_sim=n_sim,
            horizon_years=horizon,
            block_size=block,
            n_jobs=n_jobs,
            date_start=date_start,
            date_end=date_end,
            seed=seed,
            flush_every=flush_n,
            return_weights=True,
            on_start=on_start,
            on_batch=on_batch,
            on_done=on_done,
            on_error=on_error,
            stop_event=self._stop_event,
        )

    def _stop_search(self):
        self._stop_event.set()
        self.running = False
        self._log("Stopping...")

    def _reset_search(self):
        """Stop run and clear all results."""
        self._stop_event.set()
        self.running = False
        self.all_results = []
        self.sorted_tickers = []
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.chart_status.config(text="No results yet")
        self.run_error.clear()

        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._chart_opened = False
        self._chart_path = None
        self._log("Reset complete.")

    def _poll_results(self):
        if self._destroyed:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        try:
            while True:
                msg = self.result_queue.get_nowait()
                if msg[0] == "error":
                    self.run_error.show(msg[1])
                    self._log(f"ERROR: {msg[1]}")
                    self._finish_run()
                    return
                elif msg[0] == "info":
                    self.sorted_tickers = msg[1]
                    total = msg[2]
                    self.progress_label.config(text=f"0 / {total} portfolios")
                    self._log(f"Evaluating {total} portfolios "
                              f"({len(self.sorted_tickers)} assets)...")
                elif msg[0] == "batch":
                    results, done_count, total, speed = (
                        msg[1], msg[2], msg[3], msg[4]
                    )
                    self.all_results.extend(results)
                    pct = (done_count / total) * 100
                    self.progress_var.set(pct)
                    self.progress_label.config(
                        text=f"{done_count} / {total} portfolios  ({speed:.0f} p/s)"
                    )
                    self.chart_status.config(
                        text=f"{len(self.all_results)} results available"
                    )
                elif msg[0] == "done":
                    n_done, elapsed, avg_speed = msg[1], msg[2], msg[3]
                    self._log(
                        f"Done: {n_done} portfolios in {elapsed:.1f}s "
                        f"({avg_speed:.0f} p/s avg)"
                    )
                    # Drop stale overlay caches that were computed with different params.
                    current = self._current_overlay_params()
                    if current is not None:
                        self._overlay_cache = {
                            n: c for n, c in self._overlay_cache.items()
                            if c.get("params") == current
                        }
                    self.chart_status.config(
                        text=f"{len(self.all_results)} results — "
                             f"chart ready"
                    )
                    # Auto-generate and open chart on completion
                    chart_ready = self._regenerate_chart()
                    self._recompute_selected_overlays()
                    if not chart_ready:
                        self._log("Chart not generated: all results filtered out by cutoffs.")
                    elif not self._chart_opened:
                        self._open_chart()
                    self._finish_run()
                    return
                elif msg[0] == "overlay_done":
                    name = msg[1]
                    self._log(f"Overlay '{name}' ready.")
                    if self.all_results:
                        self._regenerate_chart()
                elif msg[0] == "overlay_error":
                    name, err = msg[1], msg[2]
                    self._log(f"Overlay '{name}' failed: {err}")
        except queue.Empty:
            pass
        if self.running or any(t.is_alive() for t in self._overlay_threads.values()):
            self.after(200, self._poll_results)

    def _finish_run(self):
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ── Plotly chart generation ───────────────────────────────────────────

    def _apply_cutoffs(self, results: list[dict]) -> list[dict]:
        """Filter results by the current cutoff rules."""
        cutoffs = self._get_cutoffs()
        if not cutoffs:
            return results
        filtered = []
        for r in results:
            ok = True
            for metric, op, val in cutoffs:
                rv = r.get(metric)
                if rv is None:
                    ok = False
                    break
                if op == ">=" and rv < val:
                    ok = False
                elif op == "<=" and rv > val:
                    ok = False
                elif op == ">" and rv <= val:
                    ok = False
                elif op == "<" and rv >= val:
                    ok = False
                if not ok:
                    break
            if ok:
                filtered.append(r)
        return filtered

    def _regenerate_chart(self, ignore_cutoffs: bool = False) -> bool:
        """Generate (or regenerate) the Plotly interactive scatter chart."""
        if not self.all_results:
            return False

        x_key = self.x_metric_var.get()
        y_key = self.y_metric_var.get()

        filtered = self.all_results if ignore_cutoffs else self._apply_cutoffs(self.all_results)
        if not filtered:
            self.chart_status.config(text="All results filtered out")
            return False

        xs = [r.get(x_key, 0) for r in filtered]
        ys = [r.get(y_key, 0) for r in filtered]

        # ── Build customdata and hover text (notebook-style) ─────────────
        customdata = []
        hover_texts = []
        for r in filtered:
            lines = []
            cd: dict = {}
            if "_weights" in r and self.sorted_tickers:
                for t, w in zip(self.sorted_tickers, r["_weights"]):
                    lines.append(f"{t}: {w:.1%}")
                    cd[t] = float(w)
                lines.append("────────")
            for k in sorted(r.keys()):
                if k.startswith("_"):
                    continue
                v = r[k]
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v}")
            customdata.append(cd)
            hover_texts.append("<br>".join(lines))

        fig = go.Figure()

        # Main scatter (use Scattergl for 10k+ points)
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=5,
                color="rgba(74, 74, 74, 0.5)",
                line=dict(width=0),
            ),
            customdata=customdata,
            hovertext=hover_texts,
            hoverinfo="text",
            name="Portfolios (click to add to library)",
        ))

        # Pareto front
        try:
            pareto_names = [o["name"] for o in cfg.PARETO_METRICS]
            pareto_dirs = [o["direction"] for o in cfg.PARETO_METRICS]
            data_p = np.array(
                [[r.get(n, 0) for n in pareto_names] for r in filtered]
            )
            valid = np.all(np.isfinite(data_p), axis=1)
            if valid.sum() > 1:
                pareto_idx = compute_pareto(data_p[valid], pareto_dirs)
                valid_indices = np.where(valid)[0][pareto_idx]
                pareto_x = [xs[i] for i in valid_indices]
                pareto_y = [ys[i] for i in valid_indices]
                pareto_cd = [customdata[i] for i in valid_indices]
                pareto_hover = [hover_texts[i] for i in valid_indices]
                sorted_quads = sorted(zip(pareto_x, pareto_y, pareto_cd, pareto_hover))
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in sorted_quads],
                    y=[p[1] for p in sorted_quads],
                    mode="markers+lines",
                    marker=dict(size=8, color="black", symbol="diamond"),
                    line=dict(color="black", width=1.5),
                    customdata=[p[2] for p in sorted_quads],
                    hovertext=[p[3] for p in sorted_quads],
                    hoverinfo="text",
                    name="Pareto Front (click to add to library)",
                ))
        except Exception:
            pass

        # Overlay cached portfolios (library items)
        current_overlay_params = self._current_overlay_params()
        for name, cache_entry in self._overlay_cache.items():
            if current_overlay_params is not None and cache_entry.get("params") != current_overlay_params:
                continue
            metrics = cache_entry.get("metrics", {})
            cx = metrics.get(x_key)
            cy = metrics.get(y_key)
            if cx is not None and cy is not None:
                # Build full hover: composition + all metrics (same style as main scatter)
                composition = self.library.get(name)
                hover_lines = [f"<b>{name}</b>"]
                cd: dict = {}
                if composition:
                    hover_lines.append("────────")
                    for t, w in sorted(composition.items()):
                        hover_lines.append(f"{t}: {w:.1%}")
                        cd[t] = float(w)
                hover_lines.append("────────")
                for k in sorted(metrics.keys()):
                    if k.startswith("_"):
                        continue
                    v = metrics[k]
                    if isinstance(v, float):
                        hover_lines.append(f"{k}: {v:.4f}")
                    else:
                        hover_lines.append(f"{k}: {v}")
                full_hover = "<br>".join(hover_lines)
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode="markers+text",
                    marker=dict(size=16, symbol="star", color="crimson"),
                    text=[name],
                    textposition="top right",
                    textfont=dict(size=11, color="crimson"),
                    name=name,
                    customdata=[cd],
                    hoverinfo="text",
                    hovertext=full_hover,
                ))

        fig.update_layout(
            title=dict(
                text=f"{y_key}  vs  {x_key}",
                font=dict(size=16),
            ),
            xaxis_title=x_key,
            yaxis_title=y_key,
            template="plotly_white",
            hovermode="closest",
            width=1400,
            height=850,
            margin=dict(l=80, r=40, t=60, b=60),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            dragmode="zoom",   # box-zoom by default
        )

        chart_path = os.path.join(cfg.RESULTS_DIR, "scatter.html")
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        fig.write_html(chart_path, auto_open=False, div_id="bootstrap-scatter")
        # ── Inject click-to-library JS (same-origin POST, no CORS issues) ──
        if self._click_port:
            click_script = (
                "<script>\n"
                "(function() {\n"
                "    var div = document.getElementById('bootstrap-scatter');\n"
                "    if (!div || !div.on) return;\n"
                "    div.on('plotly_click', function(data) {\n"
                "        if (!data.points || data.points.length === 0) return;\n"
                "        var pt = data.points[0];\n"
                "        if (!pt.customdata || Object.keys(pt.customdata).length === 0) return;\n"
                "        fetch('/click', {\n"
                "            method: 'POST',\n"
                "            headers: {'Content-Type': 'application/json'},\n"
                "            body: JSON.stringify({portfolio: pt.customdata})\n"
                "        }).catch(function(e) { console.warn('Click server error:', e); });\n"
                "    });\n"
                "})();\n"
                "</script>"
            )
            with open(chart_path, "r", encoding="utf-8") as fh:
                html = fh.read()
            html = html.replace("</body>", click_script + "\n</body>")
            with open(chart_path, "w", encoding="utf-8") as fh:
                fh.write(html)
        self._chart_path = chart_path
        n_filtered = len(filtered)
        n_total = len(self.all_results)
        status = f"{n_total} results"
        if n_filtered < n_total:
            status += f" ({n_filtered} after filters)"
        self.chart_status.config(text=status + " — chart ready")
        return True

    def _open_chart(self):
        """Open the scatter chart via the local HTTP server.

        Serving from http://localhost:PORT/ (not file://) makes the
        chart's fetch('/click') call same-origin, avoiding CORS.
        """
        if not (self._chart_path and os.path.exists(self._chart_path)):
            chart_ready = self._regenerate_chart()
            if not chart_ready and self.all_results:
                self._log("All points filtered out by cutoffs — opening chart with all results.")
                self._regenerate_chart(ignore_cutoffs=True)
        if self._chart_path and os.path.exists(self._chart_path):
            if self._click_port:
                webbrowser.open(f"http://localhost:{self._click_port}/scatter.html")
            else:
                webbrowser.open(f"file://{self._chart_path}")
            self._chart_opened = True
        else:
            self._log("No chart to open — run a search first.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SINGLE BOOTSTRAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class SingleBootstrapSection(tk.Frame):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.library = library
        self.result_queue = queue.Queue()
        self.running = False

        self._build_ui()
        library.on_change(self._refresh_portfolio_lists)

    def _build_ui(self):
        tab_bar = tk.Frame(self, bg=BG)
        tab_bar.pack(fill="x", padx=PAD, pady=(PAD, 0))

        self.mode_var = tk.StringVar(value="compare")
        self.compare_btn = tk.Button(
            tab_bar, text="MULTI-PORTFOLIO COMPARISON", font=(FONT_FAMILY[0], 9),
            command=lambda: self._switch_mode("compare"),
            bg=ACCENT, fg=PANEL_BG, bd=1, relief="solid", padx=10, pady=4,
        )
        self.compare_btn.pack(side="left")

        self.sweep_btn = tk.Button(
            tab_bar, text="BLOCK-SIZE SENSITIVITY", font=(FONT_FAMILY[0], 9),
            command=lambda: self._switch_mode("sweep"),
            bg=PANEL_BG, fg=TEXT, bd=1, relief="solid", padx=10, pady=4,
        )
        self.sweep_btn.pack(side="left")

        self.content = tk.Frame(self, bg=BG)
        self.content.pack(fill="both", expand=True)

        self.compare_frame = CompareSubMode(self.content, self.library)
        self.sweep_frame = SweepSubMode(self.content, self.library)
        self.compare_frame.pack(fill="both", expand=True)

    def _switch_mode(self, mode):
        self.mode_var.set(mode)
        if mode == "compare":
            self.sweep_frame.pack_forget()
            self.compare_frame.pack(fill="both", expand=True)
            self.compare_btn.config(bg=ACCENT, fg=PANEL_BG)
            self.sweep_btn.config(bg=PANEL_BG, fg=TEXT)
        else:
            self.compare_frame.pack_forget()
            self.sweep_frame.pack(fill="both", expand=True)
            self.sweep_btn.config(bg=ACCENT, fg=PANEL_BG)
            self.compare_btn.config(bg=PANEL_BG, fg=TEXT)

    def _refresh_portfolio_lists(self):
        self.compare_frame.refresh_list()
        self.sweep_frame.refresh_list()


# ── Sub-mode A: Multi-portfolio comparison ────────────────────────────────────

PORTFOLIO_COLORS = ["#1A1A1A", "#2D6A4F", "#C0392B", "#4A4A4A", "#8E7CC3", "#E67E22"]


class CompareSubMode(tk.Frame):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.library = library
        self.result_queue = queue.Queue()
        self.running = False
        self.results = {}  # name -> {metrics, paths, ann_ret, weights_arr, historical, hist_label}
        self._destroyed = False

        self._build_ui()

    def destroy(self):
        self._destroyed = True
        super().destroy()

    def _build_ui(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=PAD, pady=PAD)

        # Left config
        config = tk.Frame(top, bg=PANEL_BG, bd=1, relief="solid",
                          highlightbackground=BORDER, highlightthickness=1)
        config.pack(side="left", fill="y", padx=(0, PAD))

        FieldLabel(config, text="Select Portfolios", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )
        self.portfolio_listbox = tk.Listbox(
            config, font=(FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
            selectmode="multiple", height=6, bd=0, highlightthickness=0
        )
        self.portfolio_listbox.pack(fill="both", expand=True, padx=PAD, pady=2)
        self.refresh_list()

        self.warning_label = tk.Label(
            config, text="", font=(FONT_FAMILY[0], 8), fg=WARNING, bg=PANEL_BG
        )
        self.warning_label.pack(padx=PAD)

        # Params
        params = tk.Frame(config, bg=PANEL_BG)
        params.pack(fill="x", padx=PAD, pady=2)

        self._param_entries = {}
        for label, key, default in [
            ("N Simulations", "n_sim", "5000"),
            ("Horizon (Years)", "horizon", "10"),
            ("Block Size", "block", "6"),
            ("Seed", "seed", "42"),
            ("Date Start", "date_start", ""),
            ("Date End", "date_end", ""),
            ("Hist. Start (YYYY-MM)", "hist_start", ""),
        ]:
            FieldLabel(params, text=label, bg=PANEL_BG).pack(anchor="w")
            var = tk.StringVar(value=default)
            tk.Entry(
                params, textvariable=var, font=(FONT_FAMILY[0], 10), width=14,
                bd=1, relief="solid"
            ).pack(anchor="w", pady=(0, 2))
            self._param_entries[key] = var

        self.error_label = ErrorLabel(config, bg=PANEL_BG)
        self.error_label.pack(fill="x", padx=PAD)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(config, variable=self.progress_var, maximum=100).pack(
            fill="x", padx=PAD, pady=2
        )
        self.progress_label = tk.Label(
            config, text="", font=(FONT_FAMILY[0], 8), fg=TEXT_SEC, bg=PANEL_BG
        )
        self.progress_label.pack(padx=PAD)

        StyledButton(config, text="Run Bootstrap", command=self._run).pack(
            padx=PAD, pady=(2, PAD), anchor="w"
        )

        # Plot area
        self.plot_frame = tk.Frame(top, bg=BG)
        self.plot_frame.pack(side="left", fill="both", expand=True)

        # Bottom: summary table
        self.table_frame = tk.Frame(self, bg=BG)
        self.table_frame.pack(fill="both", expand=True, padx=PAD, pady=(0, PAD))

    def refresh_list(self):
        self.portfolio_listbox.delete(0, "end")
        for name in self.library.names():
            self.portfolio_listbox.insert("end", name)

    def _get_selected(self) -> list[str]:
        indices = self.portfolio_listbox.curselection()
        names = self.library.names()
        return [names[i] for i in indices if i < len(names)]

    def _run(self):
        self.error_label.clear()
        selected = self._get_selected()
        if len(selected) < 1:
            self.error_label.show("Select at least 1 portfolio.")
            return
        if len(selected) > 6:
            self.warning_label.config(text="Warning: >6 portfolios may clutter the plots.")
        else:
            self.warning_label.config(text="")

        try:
            n_sim = int(self._param_entries["n_sim"].get())
            horizon = int(self._param_entries["horizon"].get())
            block = int(self._param_entries["block"].get())
            seed_str = self._param_entries["seed"].get().strip()
            seed = int(seed_str) if seed_str else 42
            date_start = self._param_entries["date_start"].get().strip() or None
            date_end = self._param_entries["date_end"].get().strip() or None
            hist_start = self._param_entries["hist_start"].get().strip() or None
        except ValueError as e:
            self.error_label.show(f"Invalid param: {e}")
            return

        self.running = True
        self.results = {}
        self.progress_var.set(0)

        thread = threading.Thread(
            target=self._worker,
            args=(selected, n_sim, horizon, block, seed,
                  date_start, date_end, hist_start),
            daemon=True,
        )
        thread.start()
        self.after(200, self._poll)

    def _worker(self, names, n_sim, horizon, block, seed,
                date_start, date_end, hist_start):
        try:
            total = len(names)
            for i, name in enumerate(names):
                if not self.running:
                    break
                portfolio = _clean_portfolio(self.library.get(name))
                if not portfolio:
                    self.result_queue.put(("error", f"Portfolio '{name}' is empty or all-zero."))
                    return

                rng = np.random.default_rng(seed)
                weights_arr, ret_matrix = load_all_returns(
                    portfolio, cfg.USE_AFTER_TER_RETURNS,
                    date_start=date_start, date_end=date_end,
                )
                paths = simulate(weights_arr, ret_matrix, n_sim, horizon * 12, rng,
                                 block_size=block)
                metrics = compute_metrics(
                    paths, horizon, horizon * 12, block_size=block, weights=weights_arr
                )

                # Compute historical performance
                if hist_start:
                    # Use user-specified historical start for horizon years
                    try:
                        yr, mo = int(hist_start.split("-")[0]), int(hist_start.split("-")[1])
                        end_yr = yr + horizon
                        hist_end = f"{end_yr:04d}-{mo:02d}"
                        w_hist, ret_hist = load_all_returns(
                            portfolio, cfg.USE_AFTER_TER_RETURNS,
                            date_start=hist_start, date_end=hist_end,
                        )
                        # errstate: Apple Accelerate BLAS false positives
                        with np.errstate(all="ignore"):
                            port_hist = ret_hist @ w_hist
                        cum_hist = np.cumprod(1.0 + port_hist)
                        cum_hist = np.insert(cum_hist, 0, 1.0)
                        hist_label = f"Hist. ({hist_start} \u2192 {hist_end})"
                    except Exception:
                        # Fallback to full data
                        with np.errstate(all="ignore"):
                            port_monthly = ret_matrix @ weights_arr
                        cum_hist = np.cumprod(1.0 + port_monthly)
                        cum_hist = np.insert(cum_hist, 0, 1.0)
                        hist_label = "Historical (full data)"
                else:
                    # errstate: Apple Accelerate BLAS false positives
                    with np.errstate(all="ignore"):
                        port_monthly = ret_matrix @ weights_arr
                    cum_hist = np.cumprod(1.0 + port_monthly)
                    cum_hist = np.insert(cum_hist, 0, 1.0)
                    hist_label = "Historical (full data)"

                ann_ret = paths[:, -1] ** (1.0 / horizon) - 1.0

                self.result_queue.put((
                    "result", name, {
                        "metrics": metrics,
                        "paths": paths,
                        "ann_ret": ann_ret,
                        "weights_arr": weights_arr,
                        "historical": cum_hist,
                        "hist_label": hist_label,
                    },
                    i + 1, total
                ))
            self.result_queue.put(("done",))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _poll(self):
        if self._destroyed:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        try:
            while True:
                msg = self.result_queue.get_nowait()
                if msg[0] == "error":
                    self.error_label.show(msg[1])
                    self.running = False
                    return
                elif msg[0] == "result":
                    _, name, data, done, total = msg
                    self.results[name] = data
                    self.progress_var.set((done / total) * 100)
                    self.progress_label.config(text=f"{done}/{total} portfolios")
                elif msg[0] == "done":
                    self.running = False
                    self._render_results()
                    return
        except queue.Empty:
            pass
        if self.running:
            self.after(200, self._poll)

    def _render_results(self):
        if not self.results:
            return

        for w in self.plot_frame.winfo_children():
            w.destroy()
        for w in self.table_frame.winfo_children():
            w.destroy()

        names = list(self.results.keys())
        horizon = int(self._param_entries["horizon"].get())
        colors = PORTFOLIO_COLORS[:len(names)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=PLOT_DPI)
        fig.set_facecolor(BG)
        fig.subplots_adjust(hspace=0.35, wspace=0.30, left=0.08, right=0.95,
                            top=0.93, bottom=0.08)

        self._plot_fan_chart(axes[0, 0], names, colors, horizon)
        self._plot_return_distribution(axes[0, 1], names, colors, horizon)
        self._plot_percentile_bars(axes[1, 0], names, colors)
        self._plot_volatility(axes[1, 1], names, colors)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        self._render_summary_table(names)

    def _plot_fan_chart(self, ax, names, colors, horizon):
        apply_style(ax, "Simulated Wealth Paths")
        bands = [(5, 95, 0.10), (20, 80, 0.20), (35, 65, 0.35)]

        for ni, name in enumerate(names):
            paths = self.results[name]["paths"]
            n_steps = paths.shape[1]
            years = np.linspace(0, horizon, n_steps)

            for lo, hi, alpha in bands:
                p_lo = np.percentile(paths, lo, axis=0)
                p_hi = np.percentile(paths, hi, axis=0)
                ax.fill_between(years, p_lo, p_hi, alpha=alpha, color=colors[ni],
                                linewidth=0)

            median = np.median(paths, axis=0)
            ax.plot(years, median, color=colors[ni], linewidth=1.5, label=name)

            # Historical overlay with user-selected or full date range
            hist = self.results[name]["historical"]
            hist_label = self.results[name].get("hist_label", "Historical")
            hist_years = np.linspace(0, len(hist) / 12, len(hist))
            # Clip to horizon
            mask = hist_years <= horizon
            ax.plot(hist_years[mask], hist[mask], color=TEXT, linewidth=1,
                    linestyle="-", alpha=0.7,
                    label=hist_label if ni == 0 else None)

        ax.set_xlabel("Years", fontsize=9, color=TEXT)
        ax.set_ylabel("Portfolio Value", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False)

    def _plot_return_distribution(self, ax, names, colors, horizon):
        apply_style(ax, "Distribution of Annualised Returns")

        for ni, name in enumerate(names):
            ann_ret = self.results[name]["ann_ret"]
            ax.hist(ann_ret, bins=60, alpha=0.4, color=colors[ni], label=name,
                    edgecolor="none")

        first = names[0]
        ann_ret = self.results[first]["ann_ret"]
        for p in cfg.RETURN_PERCENTILES:
            v = np.percentile(ann_ret, p)
            ax.axvline(v, linestyle="--", color=TEXT_SEC, linewidth=0.8)
            ax.text(v, ax.get_ylim()[1] * 0.95, f"P{p}: {v:.1%}",
                    fontsize=7, color=TEXT, ha="center", va="top",
                    rotation=90)

        ax.set_xlabel("Annualised Return", fontsize=9, color=TEXT)
        ax.set_ylabel("Count", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False)

    def _plot_percentile_bars(self, ax, names, colors):
        apply_style(ax, "Annualised Return Percentiles")

        percentiles = cfg.RETURN_PERCENTILES
        n_ports = len(names)
        bar_width = 0.8 / n_ports
        x = np.arange(len(percentiles))

        for ni, name in enumerate(names):
            metrics = self.results[name]["metrics"]
            vals = [metrics.get(f"annualised_return_p{p}", 0) for p in percentiles]
            bar_colors = [POSITIVE if v >= 0 else WARNING for v in vals]
            offset = (ni - n_ports / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, color=bar_colors,
                          label=name, alpha=0.8, edgecolor="none")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1%}", ha="center", va="bottom", fontsize=6, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in percentiles], fontsize=8)
        ax.axhline(0, color=TEXT_SEC, linewidth=0.5)
        ax.set_ylabel("Annualised Return", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False)

    def _plot_volatility(self, ax, names, colors):
        apply_style(ax, "Volatility of N-Year Returns")

        for ni, name in enumerate(names):
            metrics = self.results[name]["metrics"]
            vol_keys = [k for k in metrics if k.startswith("volatility_")]
            vol_keys.sort(key=lambda k: int(k.split("_")[1].replace("y", "")))
            years = [int(k.split("_")[1].replace("y", "")) for k in vol_keys]
            vals = [metrics[k] for k in vol_keys]

            ax.plot(years, vals, color=colors[ni], marker="o", markersize=5,
                    linewidth=1.5, label=name)
            for yr, v in zip(years, vals):
                ax.text(yr, v, f"{v:.1%}", fontsize=6, ha="center",
                        va="bottom", color=colors[ni])

        ax.set_xlabel("Window (Years)", fontsize=9, color=TEXT)
        ax.set_ylabel("Std-dev of Return", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False)

    def _render_summary_table(self, names):
        if not self.results:
            return
        all_keys = set()
        for r in self.results.values():
            all_keys.update(r["metrics"].keys())
        metric_keys = sorted(all_keys)

        frame = tk.Frame(self.table_frame, bg=PANEL_BG, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True)

        FieldLabel(frame, text="Summary Statistics", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )

        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0)
        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        table_inner = tk.Frame(canvas, bg=PANEL_BG)
        table_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=table_inner, anchor="nw")
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Header row
        tk.Label(
            table_inner, text="METRIC", font=(FONT_FAMILY[0], 9, "bold"),
            bg=PANEL_BG, fg=TEXT, anchor="w", width=30
        ).grid(row=0, column=0, sticky="w", padx=4, pady=1)
        for ci, name in enumerate(names):
            tk.Label(
                table_inner, text=name, font=(FONT_FAMILY[0], 9, "bold"),
                bg=PANEL_BG, fg=PORTFOLIO_COLORS[ci % len(PORTFOLIO_COLORS)],
                anchor="e", width=16
            ).grid(row=0, column=ci + 1, sticky="e", padx=4, pady=1)

        sep = tk.Frame(table_inner, bg=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=len(names) + 1, sticky="ew", pady=2)

        for ri, key in enumerate(metric_keys):
            tk.Label(
                table_inner, text=key, font=(FONT_FAMILY[0], 9),
                bg=PANEL_BG, fg=TEXT, anchor="w"
            ).grid(row=ri + 2, column=0, sticky="w", padx=4, pady=0)

            for ci, name in enumerate(names):
                v = self.results[name]["metrics"].get(key, "")
                if isinstance(v, float):
                    txt = f"{v:.4f}"
                else:
                    txt = str(v)
                tk.Label(
                    table_inner, text=txt, font=(FONT_MONO[0], 9),
                    bg=PANEL_BG, fg=TEXT, anchor="e"
                ).grid(row=ri + 2, column=ci + 1, sticky="e", padx=4, pady=0)

            if ri % 2 == 0:
                for col in range(len(names) + 1):
                    try:
                        w = table_inner.grid_slaves(row=ri + 2, column=col)[0]
                        w.config(bg="#FAFAF5")
                    except Exception:
                        pass


# ── Sub-mode B: Block-size sensitivity sweep ──────────────────────────────────

class SweepSubMode(tk.Frame):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.library = library
        self.result_queue = queue.Queue()
        self.running = False
        self.sweep_df = None
        self._destroyed = False

        self._build_ui()

    def destroy(self):
        self._destroyed = True
        super().destroy()

    def _build_ui(self):
        left = tk.Frame(self, bg=BG, width=280)
        left.pack(side="left", fill="y", padx=(PAD, 1))
        left.pack_propagate(False)

        sep = tk.Frame(self, bg=BORDER, width=1)
        sep.pack(side="left", fill="y")

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(1, PAD))

        config = tk.Frame(left, bg=PANEL_BG, bd=1, relief="solid",
                          highlightbackground=BORDER, highlightthickness=1)
        config.pack(fill="both", expand=True, pady=PAD)

        FieldLabel(config, text="Select Portfolio", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )
        self.portfolio_listbox = tk.Listbox(
            config, font=(FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
            selectmode="browse", height=5, bd=0, highlightthickness=0
        )
        self.portfolio_listbox.pack(fill="x", padx=PAD, pady=2)
        self.refresh_list()

        self._params = {}
        for label, key, default in [
            ("Block Size Min", "bs_min", "1"),
            ("Block Size Max", "bs_max", "36"),
            ("N Simulations", "n_sim", "30000"),
            ("Horizon (Years)", "horizon", "10"),
            ("Seed", "seed", "42"),
            ("Date Start", "date_start", ""),
            ("Date End", "date_end", ""),
        ]:
            FieldLabel(config, text=label, bg=PANEL_BG).pack(anchor="w", padx=PAD)
            var = tk.StringVar(value=default)
            tk.Entry(
                config, textvariable=var, font=(FONT_FAMILY[0], 10), width=14,
                bd=1, relief="solid"
            ).pack(anchor="w", padx=PAD, pady=(0, 2))
            self._params[key] = var

        all_metrics = _build_metric_list()
        FieldLabel(config, text="X Metric", bg=PANEL_BG).pack(anchor="w", padx=PAD)
        self.x_metric_var = tk.StringVar(value="annualised_return_p50")
        ttk.Combobox(
            config, textvariable=self.x_metric_var, values=all_metrics,
            state="readonly", width=28
        ).pack(padx=PAD, anchor="w", pady=(0, 4))

        FieldLabel(config, text="Y Metric", bg=PANEL_BG).pack(anchor="w", padx=PAD)
        self.y_metric_var = tk.StringVar(value="max_dd_depth_p2.0")
        ttk.Combobox(
            config, textvariable=self.y_metric_var, values=all_metrics,
            state="readonly", width=28
        ).pack(padx=PAD, anchor="w", pady=(0, 4))

        self.error_label = ErrorLabel(config, bg=PANEL_BG)
        self.error_label.pack(fill="x", padx=PAD)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(config, variable=self.progress_var, maximum=100).pack(
            fill="x", padx=PAD, pady=2
        )
        self.progress_label = tk.Label(
            config, text="", font=(FONT_FAMILY[0], 8), fg=TEXT_SEC, bg=PANEL_BG
        )
        self.progress_label.pack(padx=PAD)

        StyledButton(config, text="Run Sweep", command=self._run).pack(
            padx=PAD, pady=(2, PAD), anchor="w"
        )

        self.plot_frame = right

    def refresh_list(self):
        self.portfolio_listbox.delete(0, "end")
        for name in self.library.names():
            self.portfolio_listbox.insert("end", name)

    def _run(self):
        self.error_label.clear()
        sel = self.portfolio_listbox.curselection()
        if not sel:
            self.error_label.show("Select a portfolio.")
            return

        name = self.library.names()[sel[0]]
        portfolio = _clean_portfolio(self.library.get(name))
        if not portfolio:
            self.error_label.show("Portfolio is empty or all-zero weights.")
            return

        try:
            bs_min = int(self._params["bs_min"].get())
            bs_max = int(self._params["bs_max"].get())
            n_sim = int(self._params["n_sim"].get())
            horizon = int(self._params["horizon"].get())
            seed = int(self._params["seed"].get()) if self._params["seed"].get().strip() else 42
            date_start = self._params["date_start"].get().strip() or None
            date_end = self._params["date_end"].get().strip() or None
        except ValueError as e:
            self.error_label.show(f"Invalid: {e}")
            return

        self.running = True
        self.progress_var.set(0)

        thread = threading.Thread(
            target=self._worker,
            args=(portfolio, bs_min, bs_max, n_sim, horizon, seed, date_start, date_end),
            daemon=True,
        )
        thread.start()
        self.after(200, self._poll)

    def _worker(self, portfolio, bs_min, bs_max, n_sim, horizon, seed, date_start, date_end):
        try:
            weights_arr, ret_matrix = load_all_returns(
                portfolio, cfg.USE_AFTER_TER_RETURNS,
                date_start=date_start, date_end=date_end,
            )
            block_sizes = list(range(bs_min, bs_max + 1))
            total = len(block_sizes)
            records = []

            for i, bs in enumerate(block_sizes):
                if not self.running:
                    break
                rng = np.random.default_rng(seed)
                m = run_bootstrap_preloaded(
                    weights_arr, ret_matrix,
                    n_sim=n_sim, horizon_years=horizon, block_size=bs, rng=rng,
                )
                m["block_size"] = bs
                records.append(m)
                self.result_queue.put(("progress", i + 1, total))

            df = pd.DataFrame(records).set_index("block_size")
            self.result_queue.put(("done", df))

        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _poll(self):
        if self._destroyed:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        try:
            while True:
                msg = self.result_queue.get_nowait()
                if msg[0] == "error":
                    self.error_label.show(msg[1])
                    self.running = False
                    return
                elif msg[0] == "progress":
                    done, total = msg[1], msg[2]
                    self.progress_var.set((done / total) * 100)
                    self.progress_label.config(text=f"Block size {done}/{total}")
                elif msg[0] == "done":
                    self.running = False
                    self.sweep_df = msg[1]
                    self._render_plot()
                    return
        except queue.Empty:
            pass
        if self.running:
            self.after(200, self._poll)

    def _render_plot(self):
        if self.sweep_df is None:
            return

        for w in self.plot_frame.winfo_children():
            w.destroy()

        x_key = self.x_metric_var.get()
        y_key = self.y_metric_var.get()

        if x_key not in self.sweep_df.columns or y_key not in self.sweep_df.columns:
            self.error_label.show("Metric not found in results.")
            return

        # Some volatility windows are skipped for block sizes that don't
        # align (m % block_size != 0), producing NaN — filter those rows.
        nan_mask = self.sweep_df[x_key].notna() & self.sweep_df[y_key].notna()
        df_valid = self.sweep_df[nan_mask]
        if df_valid.empty:
            self.error_label.show(
                "No valid data for selected metrics across this block-size range."
            )
            return

        x_vals = df_valid[x_key].values
        y_vals = df_valid[y_key].values
        bs_vals = df_valid.index.values

        fig = Figure(figsize=(9, 6), facecolor=BG, dpi=PLOT_DPI)
        ax = fig.add_subplot(111)
        apply_style(ax, f"Block-Size Sensitivity: {x_key} vs {y_key}")

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.axhspan(y_min, y_max, color=TEXT_SEC, alpha=0.10, zorder=0)
        ax.axvspan(x_min, x_max, color=TEXT_SEC, alpha=0.10, zorder=0)

        ax.text(x_max, y_max, f"Y range: {y_range:.2%}", fontsize=7,
                color=TEXT_SEC, ha="right", va="bottom")
        ax.text(x_max, y_min, f"X range: {x_range:.2%}", fontsize=7,
                color=TEXT_SEC, ha="right", va="top")

        ax.plot(x_vals, y_vals, color="#CCCCCC", linewidth=1.5, zorder=1)

        cmap = plt.cm.RdYlGn
        norm = Normalize(vmin=bs_vals.min(), vmax=bs_vals.max())
        sc = ax.scatter(
            x_vals, y_vals, c=bs_vals, cmap=cmap, norm=norm,
            s=60, zorder=3, edgecolors="#888888", linewidths=0.5
        )

        for i, bs in enumerate(bs_vals):
            ax.annotate(
                str(bs), (x_vals[i], y_vals[i]),
                textcoords="offset points", xytext=(0, 8),
                fontsize=7, ha="center", color=TEXT
            )

        if len(bs_vals) > 1:
            ax.annotate(
                "bs=1 (iid)", (x_vals[0], y_vals[0]),
                textcoords="offset points", xytext=(-20, -25),
                fontsize=9, color=TEXT,
                arrowprops=dict(arrowstyle="->", color=TEXT_SEC)
            )
            ax.annotate(
                f"bs={bs_vals[-1]}", (x_vals[-1], y_vals[-1]),
                textcoords="offset points", xytext=(20, 25),
                fontsize=9, color=TEXT,
                arrowprops=dict(arrowstyle="->", color=TEXT_SEC)
            )

        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Block size (months)", fontsize=9, color=TEXT)
        cb.ax.tick_params(labelsize=8, colors=TEXT)

        ax.set_xlabel(x_key, fontsize=9, color=TEXT)
        ax.set_ylabel(y_key, fontsize=9, color=TEXT)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=PAD, pady=PAD)

        # Hover annotation
        self._sweep_annotation = None
        self._sweep_x = x_vals
        self._sweep_y = y_vals
        self._sweep_bs = bs_vals
        self._sweep_ax = ax
        self._sweep_canvas = canvas

        def on_hover(event):
            if event.inaxes != self._sweep_ax:
                if self._sweep_annotation:
                    self._sweep_annotation.set_visible(False)
                    self._sweep_canvas.draw_idle()
                return
            if event.xdata is None:
                return

            xr = self._sweep_x.max() - self._sweep_x.min()
            yr = self._sweep_y.max() - self._sweep_y.min()
            if xr == 0: xr = 1
            if yr == 0: yr = 1

            dx = (self._sweep_x - event.xdata) / xr
            dy = (self._sweep_y - event.ydata) / yr
            dist = dx ** 2 + dy ** 2
            idx = np.argmin(dist)

            if dist[idx] > 0.01:
                if self._sweep_annotation:
                    self._sweep_annotation.set_visible(False)
                    self._sweep_canvas.draw_idle()
                return

            text = (
                f"Block size: {self._sweep_bs[idx]}\n"
                f"{x_key}: {self._sweep_x[idx]:.4f}\n"
                f"{y_key}: {self._sweep_y[idx]:.4f}"
            )

            if self._sweep_annotation:
                self._sweep_annotation.set_visible(False)

            self._sweep_annotation = self._sweep_ax.annotate(
                text,
                xy=(self._sweep_x[idx], self._sweep_y[idx]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=8, color=PANEL_BG,
                bbox=dict(boxstyle="square,pad=0.4", fc=HOVER_BG, ec="none", alpha=0.95),
                zorder=20,
            )
            self._sweep_canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_hover)
        canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class BootstrapApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Portfolio Bootstrap Analyser")
        self.geometry("1400x900")
        self.configure(bg=BG)

        self.library = PortfolioLibrary()

        self._active_btn = None
        self._build_nav()
        self._build_sections()
        self._show_section("builder")

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG)
        nav.pack(fill="x", padx=0, pady=0)

        tk.Frame(nav, bg=BORDER, height=1).pack(fill="x", side="bottom")

        self.nav_buttons = {}
        sections = [
            ("builder", "1  PORTFOLIO BUILDER"),
            ("explorer", "2  SPACE EXPLORER"),
            ("single", "3  SINGLE BOOTSTRAP"),
        ]

        for key, label in sections:
            btn = tk.Button(
                nav, text=label, font=(FONT_FAMILY[0], 10),
                bg=PANEL_BG, fg=TEXT, bd=1, relief="solid",
                padx=16, pady=8, cursor="hand2",
                command=lambda k=key: self._show_section(k),
            )
            btn.pack(side="left")
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=ACCENT, fg=PANEL_BG)
                     if b != self._active_btn else None)
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=PANEL_BG, fg=TEXT)
                     if b != self._active_btn else None)
            self.nav_buttons[key] = btn

    def _build_sections(self):
        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.sections = {
            "builder": PortfolioBuilderSection(self.container, self.library),
            "explorer": SpaceExplorerSection(self.container, self.library),
            "single": SingleBootstrapSection(self.container, self.library),
        }

    def _show_section(self, key):
        for s in self.sections.values():
            s.pack_forget()
        self.sections[key].pack(fill="both", expand=True)
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.config(bg=ACCENT, fg=PANEL_BG)
                self._active_btn = btn
            else:
                btn.config(bg=PANEL_BG, fg=TEXT)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = BootstrapApp()
    app.mainloop()
