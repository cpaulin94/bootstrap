#!/usr/bin/env python3
"""
Bootstrap Portfolio Analyser — Tkinter GUI
===========================================
A production-quality desktop application wrapping the bootstrap codebase
into a multi-panel interface with four sections:

  1. Portfolio Builder
  2. Portfolio Space Explorer (multi-bootstrap)
  3. Single Bootstrap Analysis
  4. Life Strategy Simulator (bootstrap_gui.sections.lifecycle)

Shared theming, widgets, background-job threading, and persistence live in
the ``bootstrap_gui`` package; this module still hosts the three original
sections plus the application shell.

Launch:
    uv run main.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
import http.server
import socketserver
import webbrowser
from multiprocessing import cpu_count
from tkinter import ttk, simpledialog
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# ── project imports ───────────────────────────────────────────────────────────
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from engine import config as cfg
from engine.data import (
    load_returns,
    apply_date_filter,
    parse_month_year,
    load_all_returns,
    load_independent_returns,
    load_portfolios_on_common_window,
    load_portfolio_csv,
)
from engine.metrics import compute_metrics, effective_n_assets, effective_n_types
from engine.pareto import compute_pareto
from engine.plotprep import results_to_arrays, thin_scatter
from engine.runner import (
    run_bootstrap,
    run_bootstrap_preloaded,
    run_evolutionary_streaming,
    run_multi_streaming,
)
from engine.search import load_search_space
from engine.simulation import simulate, simulate_independent

from bootstrap_gui import theme
from bootstrap_gui.theme import (
    BG, PANEL_BG, TEXT, TEXT_SEC, ACCENT, POSITIVE, WARNING, CRIMSON,
    SCATTER_DOT, GRID_CLR, BORDER, HOVER_BG, PAD,
    apply_style, make_figure,
)
from bootstrap_gui.widgets import (
    StyledButton, FieldLabel, NumericEntry, ErrorLabel, ScrollableFrame,
)
from bootstrap_gui.library import PortfolioLibrary
from bootstrap_gui.logsetup import gui_log_queue as _gui_log_queue, setup_logging
from bootstrap_gui.assets import (
    get_available_assets,
    compute_date_intersection,
    clean_portfolio as _clean_portfolio,
    build_metric_list as _build_metric_list,
    default_pareto_direction as _default_pareto_direction,
    overlay_key as _overlay_key,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP — verbose terminal + GUI forwarding
# ═══════════════════════════════════════════════════════════════════════════════

setup_logging()
_gui_log = logging.getLogger("bootstrap.gui")


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
        self.remaining_label = ttk.Label(
            top, text="REMAINING: 100.0%", style="Card.TLabel",
            font=(theme.FONT_FAMILY[0], 9, "bold"), foreground=WARNING,
        )
        self.remaining_label.pack(side="right")

        self.date_range_label = ttk.Label(
            frame, text="GLOBAL DATE RANGE: —", style="Field.Card.TLabel", anchor="w",
        )
        self.date_range_label.pack(fill="x", padx=PAD, pady=(2, 4))

        scroll_holder = tk.Frame(frame, bg=PANEL_BG, height=160)
        scroll_holder.pack(fill="x", expand=False, padx=PAD)
        scroll_holder.pack_propagate(False)
        scroll = ScrollableFrame(scroll_holder, bg=PANEL_BG)
        scroll.pack(fill="both", expand=True)
        self.portfolio_inner = scroll.inner

        self.portfolio_error = ErrorLabel(frame, bg=PANEL_BG)
        self.portfolio_error.pack(fill="x", padx=PAD, pady=(0, 2))

        btn_row = tk.Frame(frame, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=PAD, pady=(0, PAD))
        self.name_entry = ttk.Entry(btn_row, font=(theme.FONT_FAMILY[0], 10), width=18)
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
        self.heatmap_fig = Figure(figsize=(4, 3.5), facecolor=BG, dpi=theme.PLOT_DPI)
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
            frame, font=(theme.FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
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

        ttk.Label(
            row_frame, text=ticker, style="Card.TLabel",
            font=(theme.FONT_FAMILY[0], 10, "bold"), width=8, anchor="w",
        ).pack(side="left")

        var = tk.StringVar(value=f"{weight:.1f}")
        entry = NumericEntry(row_frame, textvariable=var, font=(theme.FONT_MONO[0], 10), width=8)
        entry.pack(side="left", padx=4)
        ttk.Label(row_frame, text="%", style="Field.Card.TLabel").pack(side="left")
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
        self.remaining_label.config(text=f"REMAINING: {remaining:+.1f}%", foreground=color)

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

    def _draw_empty_heatmap(self, message: str = "Add assets to see\ncorrelation heatmap"):
        self.heatmap_ax.clear()
        self.heatmap_ax.set_facecolor(PANEL_BG)
        self.heatmap_ax.text(
            0.5, 0.5, message,
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
                dates, ret = load_returns(t, cfg.USE_AFTER_TER_RETURNS)
                filtered = apply_date_filter(dates, ret, ds, de)
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
        except ValueError as e:
            # e.g. no overlapping date range across the selected tickers
            _gui_log.warning("[HEATMAP] %s", e)
            self._draw_empty_heatmap("Can't compute heatmap:\nno overlapping history\nfor these assets")
        except (OSError, KeyError) as e:
            _gui_log.warning("[HEATMAP] %s", e)
            self._draw_empty_heatmap(f"Can't compute heatmap:\n{e}")

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
        # Bumped on every new Run and on every Reset; every queued message
        # from a search worker carries the run_id it was born under, so a
        # message from a run that Reset already tore down (or a stale
        # trailing message after Stop raced with a fresh Run) is recognised
        # and dropped in _poll_results instead of corrupting current state.
        self._run_id = 0
        self._overlay_cache: dict[str, dict] = {}  # name -> {metrics, params, key, comparable}
        self._overlay_threads: dict[str, threading.Thread] = {}
        self._last_run_params: dict | None = None
        self._last_run_meta: dict | None = None  # chart-metadata only, see _run_search
        # Data the last run's cloud was actually evaluated against — an
        # overlay portfolio must reuse this (same date window, same
        # sim_seed) or its point isn't comparable to the cloud around it.
        self._last_data: object | None = None          # ret_matrix or returns_list
        self._last_sorted_tickers: list[str] = []
        self._last_sim_seed: int | None = None
        self._last_independent: bool = False
        self._chart_path: str | None = None
        self._chart_opened = False
        self._chart_version = 0
        self._chart_refresh_job: str | None = None
        self._last_live_refresh = 0.0
        self._invalidate_result_arrays()
        self._destroyed = False
        self._click_queue: queue.Queue = queue.Queue()
        self._click_port: int | None = None
        self._click_server = None

        self._build_ui()
        library.on_change(self._refresh_overlay_panel)
        library.on_change(self._refresh_space_library_selector)
        self._start_click_server()
        self.after(500, self._poll_clicks)
        self.after(200, self._poll_engine_logs)

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
        section = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=results_dir, **kwargs)

            def do_GET(self):
                # The open chart polls this and re-fetches figure.json only
                # when the counter moved, so a regenerated chart lands in
                # the already-open tab instead of waiting for a manual
                # reload — and an idle tab costs one tiny response a second.
                if self.path.split("?")[0] == "/version":
                    body = json.dumps({"v": section._chart_version}).encode()
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

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
            # Threading: the chart polls /version once a second while a
            # multi-MB figure.json download may be in flight — a
            # single-threaded server would serialise the two.
            srv = socketserver.ThreadingTCPServer(("localhost", 0), Handler)
            srv.daemon_threads = True
            srv.allow_reuse_address = True
            self._click_port = srv.server_address[1]
            self._click_server = srv
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
        except OSError as e:
            self._click_port = None
            _gui_log.warning(
                "[EXPLORER] Click-to-library server could not start (%s) — "
                "clicking chart points to save them to the library will not work "
                "this session; charts still open and render normally.", e,
            )

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
        """Queue a clicked chart point for the user to name/save at their own
        pace — never blocks the mainloop with a modal (see comment above
        ``pending_container`` in ``_build_config_panel``)."""
        weights = {
            k: float(v)
            for k, v in portfolio.items()
            if isinstance(v, (int, float)) and float(v) > 1e-9
        }
        if not weights:
            return
        self._add_pending_selection(weights)
        self._log(f"Chart point queued ({len(weights)} assets) — "
                  f"name and save it in the 'Pending Chart Selections' panel.")

    def _add_pending_selection(self, weights: dict[str, float]) -> None:
        n = len(self._pending_selections) + 1
        preview = " · ".join(f"{t} {w:.0%}" for t, w in sorted(weights.items()))

        row = tk.Frame(self.pending_container, bg=PANEL_BG, bd=1, relief="solid",
                       highlightbackground=BORDER, highlightthickness=1)
        row.pack(fill="x", pady=2)

        ttk.Label(row, text=preview, style="Field.Card.TLabel", wraplength=260,
                  justify="left").pack(anchor="w", padx=4, pady=(4, 2))

        name_row = tk.Frame(row, bg=PANEL_BG)
        name_row.pack(fill="x", padx=4, pady=(0, 4))
        name_var = tk.StringVar(value=f"Selected Portfolio {n}")
        entry = ttk.Entry(name_row, textvariable=name_var, width=16)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        entry_dict = {"weights": weights, "name_var": name_var, "frame": row}
        self._pending_selections.append(entry_dict)

        def save(e=entry_dict):
            name = e["name_var"].get().strip()
            if not name:
                return
            self.library.add(name, e["weights"])
            self._log(f"Saved '{name}' to library ({len(e['weights'])} assets)")
            self._remove_pending_selection(e)

        def discard(e=entry_dict):
            self._remove_pending_selection(e)

        StyledButton(name_row, text="Save", command=save).pack(side="left", padx=(0, 2))
        StyledButton(name_row, text="Discard", command=discard).pack(side="left")
        entry.bind("<Return>", lambda _e, s=save: s())

        self.pending_hint.pack_forget()

    def _remove_pending_selection(self, entry: dict) -> None:
        if entry in self._pending_selections:
            self._pending_selections.remove(entry)
        entry["frame"].destroy()
        if not self._pending_selections:
            self.pending_hint.pack(anchor="w", fill="x")

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
        scroll = ScrollableFrame(parent, bg=BG, width=320)
        scroll.pack(fill="both", expand=True)
        inner = scroll.inner

        # Search mode
        mode_frame = self._make_section(inner, "Search Mode")
        self.search_mode = tk.StringVar(value="mixed")
        ttk.Radiobutton(
            mode_frame, text="Mixed Search (recommended)", variable=self.search_mode,
            value="mixed",
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame, text="Random Search", variable=self.search_mode, value="random",
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame, text="Grid Search", variable=self.search_mode, value="grid",
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame, text="Evolutionary Search", variable=self.search_mode,
            value="evolutionary",
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
        self.independent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params_frame, text="Independent Resampling (break correlations)",
            variable=self.independent_var,
        ).pack(anchor="w")

        # Multi-bootstrap params
        multi_frame = self._make_section(inner, "Search Parameters")
        self.n_portfolios_var = self._add_param(multi_frame, "N Portfolios",
                                                str(cfg.N_PORTFOLIOS))
        self.grid_step_var = self._add_param(multi_frame, "Grid Step",
                                             str(cfg.GRID_STEP))
        self.n_generations_var = self._add_param(
            multi_frame, "Generations (evolutionary only)", "5")
        self.flush_interval_var = self._add_param(multi_frame, "Flush Every N", "500")
        self.n_jobs_var = self._add_param(multi_frame, "Parallel Jobs (-1=all)", "-1")

        # Pareto objectives — which metrics define "the frontier", both for
        # the diamond markers/line drawn on the chart and for what an
        # evolutionary search refines around. Built dynamically (not a
        # fixed list) so it always matches what compute_metrics can
        # actually produce for the current BAD_PERCENTILE/VOLATILITY_WINDOWS
        # config, same principle as the X/Y axis dropdowns.
        pareto_frame = self._make_section(inner, "Pareto Objectives")
        FieldLabel(
            pareto_frame,
            text="Which metrics define the frontier (chart + evolutionary search):",
            bg=BG,
        ).pack(anchor="w", pady=(0, 2))
        self._pareto_metric_vars: dict[str, tk.BooleanVar] = {}
        default_names = {m["name"] for m in cfg.PARETO_METRICS}
        for metric_name in _build_metric_list():
            var = tk.BooleanVar(value=metric_name in default_names)
            var.trace_add("write", lambda *a: self._on_pareto_selection_changed())
            self._pareto_metric_vars[metric_name] = var
            direction = _default_pareto_direction(metric_name)
            arrow = "↑ max" if direction == "maximize" else "↓ min"
            ttk.Checkbutton(
                pareto_frame, text=f"{metric_name}  ({arrow})", variable=var,
            ).pack(anchor="w")

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
        NumericEntry(
            delta_row, textvariable=self.space_delta_var, width=6, font=(theme.FONT_MONO[0], 9),
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
            ttk.Label(row, text=t, font=(theme.FONT_FAMILY[0], 9, "bold"),
                      width=6, anchor="w").pack(side="left")
            lo_var = tk.StringVar(value="0.0")
            hi_var = tk.StringVar(value="0.3")
            NumericEntry(row, textvariable=lo_var, width=5,
                         font=(theme.FONT_MONO[0], 9)).pack(side="left", padx=1)
            ttk.Label(row, text="–", style="Secondary.TLabel").pack(side="left")
            NumericEntry(row, textvariable=hi_var, width=5,
                         font=(theme.FONT_MONO[0], 9)).pack(side="left", padx=1)
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
        ttk.Entry(date_frame, textvariable=self.date_start_var, font=(theme.FONT_FAMILY[0], 10),
                  width=12).pack(anchor="w", pady=(0, 4))
        FieldLabel(date_frame, text="End (YYYY-MM)", bg=BG).pack(anchor="w")
        ttk.Entry(date_frame, textvariable=self.date_end_var, font=(theme.FONT_FAMILY[0], 10),
                  width=12).pack(anchor="w")

        # Overlay selection — checkboxes (compute on toggle)
        overlay_frame = self._make_section(inner, "Current Portfolios Overlay")
        self.overlay_container = tk.Frame(overlay_frame, bg=BG)
        self.overlay_container.pack(fill="x", pady=2)
        self._overlay_vars: dict[str, tk.BooleanVar] = {}
        self._refresh_overlay_panel()

        # Pending chart selections — non-modal. Clicking a point on the chart
        # (which lives in the browser) used to pop a simpledialog.askstring()
        # modal on top of the Tk window; if the browser had focus that modal
        # was easy to miss and it blocked the Tk mainloop until dismissed.
        # Clicks now just queue up here for the user to name/save at their
        # own pace, or discard.
        pending_frame = self._make_section(inner, "Pending Chart Selections")
        self.pending_hint = ttk.Label(
            pending_frame, text="Click a point on the chart to add it here.",
            style="Secondary.TLabel", wraplength=280,
        )
        self.pending_hint.pack(anchor="w", fill="x")
        self.pending_container = tk.Frame(pending_frame, bg=BG)
        self.pending_container.pack(fill="x", pady=2)
        self._pending_selections: list[dict] = []

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
        self.progress_label = ttk.Label(
            run_frame, text="", font=(theme.FONT_FAMILY[0], 10), style="Secondary.TLabel",
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
        self.x_metric_combo = ttk.Combobox(
            x_col, textvariable=self.x_metric_var, values=all_metrics,
            state="readonly", width=28
        )
        self.x_metric_combo.pack(fill="x", pady=(0, 4))

        y_col = tk.Frame(ax_row, bg=BG)
        y_col.pack(side="left", fill="x", expand=True)
        FieldLabel(y_col, text="Y Axis", bg=BG).pack(anchor="w")
        self.y_metric_var = tk.StringVar(value="annualised_return_p1")
        self.y_metric_combo = ttk.Combobox(
            y_col, textvariable=self.y_metric_var, values=all_metrics,
            state="readonly", width=28
        )
        self.y_metric_combo.pack(fill="x", pady=(0, 4))

        # Reactive: regenerate chart when axes change (debounced — a
        # regeneration touches every result, and the combobox can fire
        # more than once per pick).
        self.x_metric_var.trace_add("write", lambda *_: self._schedule_chart_refresh())
        self.y_metric_var.trace_add("write", lambda *_: self._schedule_chart_refresh())

        # ── Pareto-only toggle ───────────────────────────────────────────
        self.pareto_only_var = tk.BooleanVar(value=False)
        pareto_only_cb = ttk.Checkbutton(
            axis_frame, text="Show Pareto frontier only (hide the cloud)",
            variable=self.pareto_only_var,
            command=lambda: self._schedule_chart_refresh(delay_ms=0),
        )
        pareto_only_cb.pack(fill="x", pady=(4, 0), anchor="w")

        # ── Point budget ──────────────────────────────────────────────────
        pts_row = tk.Frame(axis_frame, bg=BG)
        pts_row.pack(fill="x", pady=(4, 0))
        FieldLabel(pts_row, text="Max points drawn", bg=BG).pack(side="left")
        self.max_points_var = tk.StringVar(value=str(self.MAX_CHART_POINTS))
        ttk.Entry(
            pts_row, textvariable=self.max_points_var,
            font=(theme.FONT_MONO[0], 9), width=8,
        ).pack(side="left", padx=(6, 0))

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

        self.chart_status = ttk.Label(
            axis_frame, text="No results yet", font=(theme.FONT_FAMILY[0], 9),
            style="Secondary.TLabel", anchor="w",
        )
        self.chart_status.pack(fill="x", pady=(4, 0))

        # ── Log panel (expanded) ─────────────────────────────────────────
        log_frame = self._make_section(parent, "Log")
        self.log_text = tk.Text(
            log_frame, font=(theme.FONT_MONO[0], 9), bg=PANEL_BG, fg=TEXT,
            height=30, bd=1, relief="solid", wrap="word", state="normal"
        )
        self.log_text.pack(fill="both", expand=True, pady=2)

    def _make_section(self, parent, title):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="x", padx=PAD, pady=(PAD, 0))
        if title:
            ttk.Label(
                frame, text=title.upper(), font=(theme.FONT_FAMILY[0], 9, "bold"), anchor="w",
            ).pack(fill="x", pady=(0, 2))
            tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 4))
        return frame

    def _add_param(self, parent, label, default):
        FieldLabel(parent, text=label, bg=BG).pack(anchor="w")
        var = tk.StringVar(value=default)
        ttk.Entry(
            parent, textvariable=var, font=(theme.FONT_FAMILY[0], 10), width=14,
        ).pack(anchor="w", pady=(0, 4))
        return var

    def _load_search_csv_defaults(self):
        try:
            space = load_search_space(cfg.SEARCH_CSV)
        except (OSError, KeyError, ValueError) as e:
            _gui_log.warning(
                "[SPACE] Could not load search space defaults from %s: %s — "
                "bounds stay at their built-in defaults.", cfg.SEARCH_CSV, e,
            )
            return
        for item in space:
            t = item["ticker"]
            if t in self.space_entries:
                lo_var, hi_var = self.space_entries[t]
                lo_var.set(f"{item['lo']:.2f}")
                hi_var.set(f"{item['hi']:.2f}")

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
        ttk.Entry(
            row2, textvariable=val_var, font=(theme.FONT_MONO[0], 9), width=10,
        ).pack(side="left", padx=(0, 4))

        entry = {"metric_var": metric_var, "op_var": op_var, "val_var": val_var, "frame": outer}
        self.cutoff_rows.append(entry)

        def remove(e=entry):
            self.cutoff_rows.remove(e)
            e["frame"].destroy()

        remove_btn = ttk.Button(
            row2, text="REMOVE", command=remove, style="Danger.TButton", cursor="hand2",
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
            cb = ttk.Checkbutton(
                self.overlay_container, text=name, variable=var,
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
            portfolio = _clean_portfolio(self.library.get(name) or {})

            cached = self._overlay_cache.get(name)
            if cached and cached.get("key") == self._current_overlay_key(params, portfolio):
                if self.all_results:
                    self._schedule_chart_refresh()
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
                self._schedule_chart_refresh()

    def _refresh_metric_dropdowns(self) -> None:
        """Repopulate the X/Y axis dropdowns from the metrics ACTUALLY
        present in this run's results, instead of the static engine-config
        list. A metric can be legitimately absent for every result (e.g.
        ``volatility_10y`` when the run's horizon is under 10 years, or any
        volatility window when ``block_size`` doesn't divide the horizon
        evenly) — offering it in the dropdown anyway used to mean every
        point silently scored 0.0 on that axis instead of the chart saying
        so.
        """
        if not self.all_results:
            return
        present: set[str] = set()
        for r in self.all_results:
            present.update(k for k in r.keys() if not k.startswith("_"))
        available = sorted(present)
        if not available:
            return
        self.x_metric_combo["values"] = available
        self.y_metric_combo["values"] = available
        if self.x_metric_var.get() not in available:
            self.x_metric_var.set(available[0])
        if self.y_metric_var.get() not in available:
            self.y_metric_var.set(available[min(1, len(available) - 1)])

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
                "independent": self.independent_var.get(),
            }
        except ValueError:
            self.run_error.show("Invalid simulation parameters for overlay.")
            return None

    def _overlay_use_preloaded(self, params: dict, portfolio: dict) -> bool:
        """Would an overlay computed now reuse the run's exact data + seed?

        Requires matching params AND that every ticker *portfolio* holds
        is actually covered by the loaded run's search-space ticker set —
        a portfolio using an asset outside the current search space can't
        be sliced out of ``self._last_data`` at all (there's no column for
        it), so it always falls back to a fresh computation instead. That
        fallback used to be a hard error; it no longer needs to be, since
        ``run_bootstrap`` now anchors to the full asset universe rather
        than just the portfolio's own tickers (AUDIT.md M9) — it always
        succeeds for any portfolio built from ``data/standard/`` assets,
        it just isn't comparable to a cloud whose search space is a
        strict subset of that universe (see ``comparable`` below).

        Kept as its own method so ``_current_overlay_key`` can predict,
        before computing anything, whether an overlay would land in the
        comparable (preloaded) or non-comparable (fallback) branch.
        """
        return bool(
            self._last_data is not None
            and self._last_sorted_tickers
            and self._last_run_params == params
            and all(t in self._last_sorted_tickers for t in portfolio)
        )

    def _current_overlay_key(self, params: dict, portfolio: dict) -> tuple:
        """Identity an overlay must match to be valid for the CURRENT cloud.

        Simulation params alone are not enough to tell whether a cached
        overlay is still comparable to what's on screen: the historical
        date window and the Monte-Carlo draw both depend on the run's
        full ticker set and ``sim_seed``, not just on n_sim/horizon/block
        — and whether THIS portfolio's own tickers are even covered by
        that ticker set is portfolio-specific, not a function of params
        alone. An overlay computed via the fallback branch (no matching
        run loaded, or this portfolio uses a ticker outside the run's
        search space) is never comparable to a cloud, no matter what
        params it used — so it always gets the ``(params, None, None)``
        key here, which can only match another fallback overlay, never a
        cloud.
        """
        if self._overlay_use_preloaded(params, portfolio):
            return _overlay_key(params, self._last_sorted_tickers, self._last_sim_seed)
        return _overlay_key(params, None, None)

    def _compute_overlay_worker(self, name: str, params: dict):
        """Background thread: compute overlay metrics for one portfolio.

        Reuses the exact data + seed the run's cloud was evaluated against
        (``self._last_data`` / ``self._last_sim_seed``, populated via
        ``on_data_ready`` — see engine.runner.run_multi_streaming) whenever
        *params* still matches that run AND every ticker the portfolio
        holds is covered by the run's search space. Otherwise falls back
        to ``run_bootstrap``, which anchors to the full asset universe
        (AUDIT.md M9) — this always succeeds for any portfolio built from
        ``data/standard/`` assets, even one using a ticker outside the
        CURRENT search space; it just isn't comparable to a cloud whose
        search space is a strict subset of that universe, which
        ``comparable``/the caller's chart-drawing code marks accordingly
        (open, dimmed marker + hover warning) instead of erroring out.
        """
        try:
            portfolio = _clean_portfolio(self.library.get(name))
            if not portfolio:
                return

            use_preloaded = self._overlay_use_preloaded(params, portfolio)
            if use_preloaded:
                weights = np.array(
                    [portfolio.get(t, 0.0) for t in self._last_sorted_tickers],
                    dtype=np.float64,
                )
                rng = np.random.default_rng(self._last_sim_seed)
                metrics = run_bootstrap_preloaded(
                    weights,
                    None if self._last_independent else self._last_data,
                    n_sim=params["n_sim"],
                    horizon_years=params["horizon_years"],
                    block_size=params["block_size"],
                    rng=rng,
                    tickers=self._last_sorted_tickers,
                    independent=self._last_independent,
                    returns_list=self._last_data if self._last_independent else None,
                )
            else:
                metrics = run_bootstrap(
                    portfolio,
                    n_sim=params["n_sim"],
                    horizon_years=params["horizon_years"],
                    block_size=params["block_size"],
                    random_seed=None,
                    date_start=params["date_start"],
                    date_end=params["date_end"],
                    independent=params.get("independent", False),
                )
            self._overlay_cache[name] = {
                "metrics": metrics,
                "params": dict(params),
                "key": self._current_overlay_key(params, portfolio),
                "comparable": use_preloaded,
            }
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
            portfolio = _clean_portfolio(self.library.get(name) or {})
            cached = self._overlay_cache.get(name)
            if cached and cached.get("key") == self._current_overlay_key(params, portfolio):
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
        # Also log to the bootstrap.gui logger for terminal visibility
        _gui_log.info(msg)

    def _drain_engine_logs(self):
        """Drain engine log messages from the queue into the GUI log panel."""
        try:
            while True:
                msg = _gui_log_queue.get_nowait()
                ts = time.strftime("%H:%M:%S")
                self.log_text.insert("end", f"[{ts}] {msg}\n")
                self.log_text.see("end")
        except queue.Empty:
            pass

    def _poll_engine_logs(self):
        """Periodic poll to drain engine log queue into GUI log."""
        if self._destroyed:
            return
        self._drain_engine_logs()
        self.after(150, self._poll_engine_logs)

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
            n_generations = int(self.n_generations_var.get())
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
            "independent": self.independent_var.get(),
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

        # Chart-metadata only — kept separate from ``_last_run_params``
        # (which feeds overlay-comparability equality checks) so a wider
        # metadata set here can never change what counts as "the same
        # run" for an overlay.
        self._last_run_meta = {
            "method": method,
            "n_portfolios": n_port,
            "seed": seed,
            "space": [dict(s) for s in space],
            "grid_step": grid_step if method == "grid" else None,
            "n_generations": n_generations if method == "evolutionary" else None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Log run parameters
        self._log(f"══════════════════════════════════════════════════")
        self._log(f"Starting {method.upper()} search")
        self._log(f"══════════════════════════════════════════════════")
        self._log(f"  N_portfolios={n_port}, N_simulations_per_portfolio={n_sim}")
        if n_sim < 5000:
            self._log(
                f"  ⚠️ n_sim={n_sim}: the CLOUD'S ABSOLUTE level has sampling "
                f"noise of roughly ±0.7pp on annualised_return_p50 and ±2pp on "
                f"annualised_return_p1 (measured at n_sim=1000; scales ~1/√n_sim). "
                f"Comparisons BETWEEN portfolios in this run stay valid (they "
                f"share sim_seed), but don't read the cloud's absolute position "
                f"as precise at this n_sim."
            )
        self._log(f"  Horizon={horizon} years ({horizon * 12} months)")
        self._log(f"  Block size={block} months (block bootstrap)")
        self._log(f"  Flush interval={flush_n} portfolios")
        self._log(f"  Random seed={seed or 'None (non-deterministic)'}")
        if date_start or date_end:
            self._log(f"  Date filter: [{date_start or 'earliest'} → {date_end or 'latest'}]")
        else:
            self._log(f"  Date filter: None (using full history)")
        self._log(f"  Search space ({len(space)} assets):")
        lo_sum = sum(s['lo'] for s in space)
        hi_sum = sum(s['hi'] for s in space)
        for s in space:
            self._log(f"    {s['ticker']:>6s}:  [{s['lo']:.4f}, {s['hi']:.4f}]")
        self._log(f"  Sum of bounds:  lo_sum={lo_sum:.4f}  hi_sum={hi_sum:.4f}")
        if lo_sum > 1.0 + 1e-9:
            self._log(f"  ⚠️ WARNING: sum(lo)={lo_sum:.4f} > 1.0 → infeasible!")
        if hi_sum < 1.0 - 1e-9:
            self._log(f"  ⚠️ WARNING: sum(hi)={hi_sum:.4f} < 1.0 → infeasible!")
        self._log(f"  Workers: {n_workers} CPU cores")
        self._log(f"  Estimated work: {n_port} portfolios × {n_sim} sims = {n_port * n_sim:,} total simulations")
        self._log(f"──────────────────────────────────────────────────")
        self._log(f"Spawning background worker thread...")

        self.running = True
        self._stop_event.clear()
        self._run_id += 1
        run_id = self._run_id
        self.all_results = []
        self.sorted_tickers = []
        self._invalidate_result_arrays()
        self._last_live_refresh = 0.0
        self._chart_opened = False

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.progress_label.config(text="")

        # Replace whatever the chart was showing (a previous run's cloud,
        # or nothing) right away — otherwise an already-open tab keeps
        # displaying stale data with no indication a new run even
        # started, which is indistinguishable from "the refresh is
        # broken" from the user's side.
        self._write_cleared_chart("Search running — no results yet.")
        self.chart_status.config(text="Running — waiting for first batch...")

        thread = threading.Thread(
            target=self._search_worker,
            args=(run_id, space, method, n_port, grid_step, n_generations, n_sim, horizon,
                  block, seed, date_start, date_end, flush_n, n_jobs,
                  self.independent_var.get(), self._current_pareto_metrics()),
            daemon=True,
        )
        thread.start()
        self.after(200, self._poll_results)

    def _search_worker(self, run_id, space, method, n_portfolios, grid_step, n_generations,
                       n_sim, horizon, block, seed, date_start, date_end,
                       flush_n, n_jobs, independent=False, pareto_metrics=None):
        """Thin wrapper: delegates all heavy work to engine.run_multi_streaming
        or, for the evolutionary method, engine.run_evolutionary_streaming."""
        _gui_log.info("[GUI_WORKER] Worker thread started — entering run_%s_streaming()",
                      "evolutionary" if method == "evolutionary" else "multi")
        t_worker_start = time.perf_counter()

        def on_start(sorted_tickers, total):
            _gui_log.info("[GUI_WORKER] on_start callback: %d tickers, %d portfolios",
                          len(sorted_tickers), total)
            self.result_queue.put(("info", run_id, sorted_tickers, total))

        def on_data_ready(sorted_tickers, data, sim_seed):
            _gui_log.info("[GUI_WORKER] on_data_ready: %d tickers, sim_seed=%s", len(sorted_tickers), sim_seed)
            self.result_queue.put(("data_ready", run_id, sorted_tickers, data, sim_seed, independent))

        def on_generation(gen, n_gens, frontier_size):
            self._log(f"  Generation {gen + 1}/{n_gens}: refining around a "
                      f"{frontier_size}-portfolio frontier...")

        def on_batch(results, n_done, total, speed):
            _gui_log.info("[GUI_WORKER] on_batch: %d/%d done (%.0f p/s), batch_size=%d",
                          n_done, total, speed, len(results))
            self.result_queue.put(("batch", run_id, results, n_done, total, speed))

        def on_done(n_done, elapsed, avg_speed):
            _gui_log.info("[GUI_WORKER] on_done: %d portfolios in %.1fs (%.0f p/s avg)",
                          n_done, elapsed, avg_speed)
            self.result_queue.put(("done", run_id, n_done, elapsed, avg_speed))

        def on_error(msg):
            _gui_log.error("[GUI_WORKER] on_error: %s", msg)
            self.result_queue.put(("error", run_id, msg))

        if method == "evolutionary":
            run_evolutionary_streaming(
                space,
                n_portfolios=n_portfolios,
                n_generations=n_generations,
                pareto_metrics=pareto_metrics,
                n_sim=n_sim,
                horizon_years=horizon,
                block_size=block,
                n_jobs=n_jobs,
                date_start=date_start,
                date_end=date_end,
                seed=seed,
                flush_every=flush_n,
                return_weights=True,
                independent=independent,
                on_start=on_start,
                on_data_ready=on_data_ready,
                on_generation=on_generation,
                on_batch=on_batch,
                on_done=on_done,
                on_error=on_error,
                stop_event=self._stop_event,
            )
        else:
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
                independent=independent,
                on_start=on_start,
                on_data_ready=on_data_ready,
                on_batch=on_batch,
                on_done=on_done,
                on_error=on_error,
                stop_event=self._stop_event,
            )

    def _stop_search(self):
        """Ask the worker to wind down — does NOT flip ``self.running``.

        The worker thread always reports back through the queue (``on_done``
        fires with whatever partial results it has even after a stop, see
        ``run_multi_streaming``/``run_evolutionary_streaming``). Setting
        ``running = False`` here used to kill ``_poll_results``'s own
        reschedule loop immediately, before that final message arrived —
        leaving it stuck in the queue forever, the Run button permanently
        disabled, and the Stop button permanently (uselessly) enabled.
        ``_finish_run`` is the only place ``running`` goes back to False now,
        and it only runs once that final message is actually drained.
        """
        if not self.running:
            return
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self._log("Stop requested — waiting for the worker pool to exit...")

    def _reset_search(self):
        """Stop any run in progress and clear all results — immediately.

        Unlike Stop, this doesn't wait for the worker's final message: it
        bumps ``_run_id`` so that message (if one is still in flight) is
        recognised as stale and ignored by ``_poll_results`` instead of
        reviving state this just cleared.
        """
        self._stop_event.set()
        self._run_id += 1
        self.running = False
        self.all_results = []
        self.sorted_tickers = []
        self._invalidate_result_arrays()
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.run_error.clear()
        self._last_run_params = None
        self._last_run_meta = None

        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._chart_opened = False
        self._chart_path = None
        # The chart tab, if open, otherwise keeps showing whatever cloud
        # was on screen before Reset with nothing telling you it's stale —
        # push an explicit "cleared" state so it can't be mistaken for
        # live data.
        self._write_cleared_chart("Reset — no results. Press Run to start a new search.")
        self.chart_status.config(text="No results yet")
        self._log("Reset complete.")

    def _poll_results(self):
        if self._destroyed:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        # Drain engine log messages first
        self._drain_engine_logs()

        try:
            while True:
                msg = self.result_queue.get_nowait()
                kind = msg[0]

                # Every message from a search worker (all but the overlay
                # ones, which aren't tied to a run) carries the run_id it
                # was spawned under as its second element. Reset() bumps
                # ``self._run_id`` precisely so a message from a run that
                # was already torn down gets ignored here instead of
                # reviving state Reset just cleared.
                if kind in ("error", "info", "data_ready", "batch", "done"):
                    run_id = msg[1]
                    msg = msg[2:]
                    if run_id != self._run_id:
                        _gui_log.info(
                            "[EXPLORER] Ignoring stale '%s' message from run %d "
                            "(current run is %d)", kind, run_id, self._run_id,
                        )
                        continue

                if kind == "error":
                    self.run_error.show(msg[0])
                    self._log(f"❌ ERROR: {msg[0]}")
                    self._finish_run()
                    return
                elif kind == "info":
                    self.sorted_tickers = msg[0]
                    total = msg[1]
                    self.progress_label.config(text=f"0 / {total} portfolios")
                    self._log(f"Data loaded. Assets: {self.sorted_tickers}")
                    self._log(f"Evaluating {total} portfolios "
                              f"across {len(self.sorted_tickers)} assets...")
                    self._log(f"Waiting for first batch from worker pool...")
                elif kind == "data_ready":
                    # The exact data + seed this run's cloud is being evaluated
                    # against — overlays reuse this so they land on the same
                    # date window and Monte-Carlo draw as the cloud around them.
                    self._last_sorted_tickers = msg[0]
                    self._last_data = msg[1]
                    self._last_sim_seed = msg[2]
                    self._last_independent = msg[3]
                    # sim_seed is shared by every candidate in the run (common
                    # random numbers — cancels relative MC noise between
                    # portfolios) but it also shifts the WHOLE cloud by one
                    # shared historical draw. Surfacing it here is what lets
                    # you tell "this cloud looks off" apart from "I compared
                    # two clouds drawn with different seeds".
                    self._log(f"  sim_seed={self._last_sim_seed} "
                              f"(shared by every candidate — comparisons across "
                              f"runs with a different sim_seed are not meaningful)")
                elif kind == "batch":
                    results, done_count, total, speed = msg[0], msg[1], msg[2], msg[3]
                    self.all_results.extend(results)
                    pct = (done_count / total) * 100
                    self.progress_var.set(pct)
                    eta = (total - done_count) / max(speed, 0.1)
                    self.progress_label.config(
                        text=f"{done_count} / {total} portfolios  "
                             f"({speed:.0f} p/s, ETA ~{eta:.0f}s)"
                    )
                    self._log(f"Batch: {done_count}/{total} ({pct:.1f}%) — "
                              f"{speed:.0f} portfolios/s — "
                              f"ETA ~{eta:.0f}s — "
                              f"{len(self.all_results)} results accumulated")
                    self.chart_status.config(
                        text=f"{len(self.all_results)} results available"
                    )
                    self._live_refresh_chart()
                elif kind == "done":
                    n_done, elapsed, avg_speed = msg[0], msg[1], msg[2]
                    stopped_early = self._stop_event.is_set()
                    self._log(f"══════════════════════════════════════════════════")
                    self._log(
                        (f"⏹ STOPPED: {n_done} portfolios evaluated before the "
                         f"stop request ({elapsed:.1f}s, {avg_speed:.0f} p/s avg)"
                         if stopped_early else
                         f"✅ COMPLETE: {n_done} portfolios evaluated in {elapsed:.1f}s "
                         f"({avg_speed:.0f} portfolios/s avg)")
                    )
                    self._log(f"Total results in memory: {len(self.all_results)}")
                    self._log(f"──────────────────────────────────────────────────")
                    self._log(f"Generating Plotly interactive chart...")
                    self._refresh_metric_dropdowns()
                    # Drop stale overlay caches — computed against different
                    # params, a different ticker set, or a different sim_seed
                    # than this run just produced. Keyed per-portfolio (not
                    # one key for all): whether a portfolio's own tickers
                    # are covered by this run's search space is portfolio-
                    # specific, not just a function of the run's params.
                    current = self._current_overlay_params()
                    if current is not None:
                        self._overlay_cache = {
                            n: c for n, c in self._overlay_cache.items()
                            if c.get("key") == self._current_overlay_key(
                                current, _clean_portfolio(self.library.get(n) or {})
                            )
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
                elif kind == "overlay_done":
                    name = msg[1]
                    self._log(f"Overlay '{name}' ready.")
                    if self.all_results:
                        # Debounced: when several overlays finish in the
                        # same poll (e.g. many library portfolios checked
                        # at once after a run), each "overlay_done" used
                        # to trigger its own synchronous chart rewrite —
                        # queue.get_nowait() drains the WHOLE queue before
                        # yielding back to Tk, so N overlays meant N
                        # back-to-back full regenerations (each one a
                        # multi-MB JSON write) freezing the UI in one go.
                        # One coalesced redraw instead.
                        self._schedule_chart_refresh()
                elif kind == "overlay_error":
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
    #
    # Everything below is built for the size this section actually reaches:
    # a 100k-portfolio run. Three rules keep it interactive.
    #
    #   1. Nothing per-point is a Python string. The figure carries ONE
    #      hover template plus a numeric ``customdata`` matrix, which
    #      plotly serialises as base64 typed arrays. The old code
    #      pre-rendered an HTML hover string and a {ticker: weight} dict
    #      for every point — that is what turned a 100k run into a 137 MB
    #      HTML file the browser had to parse before drawing anything.
    #   2. The cloud is thinned to ``MAX_CHART_POINTS`` markers
    #      (engine.plotprep.thin_scatter) before it leaves Python.
    #   3. The page is written once as a small shell; refreshes rewrite
    #      only ``figure.json`` and the already-open tab picks it up with
    #      Plotly.react — no reload, no re-parse of plotly.js, and the
    #      user's zoom survives (uirevision).

    MAX_CHART_POINTS = 25_000
    LIVE_REFRESH_SECONDS = 3.0

    def _max_points(self) -> int:
        """The user's point budget, clamped to something a browser survives."""
        try:
            return int(np.clip(int(float(self.max_points_var.get())), 500, 500_000))
        except (ValueError, AttributeError):
            return self.MAX_CHART_POINTS

    def _live_refresh_chart(self) -> None:
        """Push the partial cloud into the chart while the run is going.

        The open tab polls for new figures on its own, so this is what
        makes the space visibly fill in instead of showing nothing until
        the run ends. Throttled, because a regeneration walks every
        result accumulated so far.
        """
        if not self._click_port or not self.all_results:
            return
        now = time.monotonic()
        if now - self._last_live_refresh < self.LIVE_REFRESH_SECONDS:
            return
        self._last_live_refresh = now
        if self._regenerate_chart() and not self._chart_opened:
            self._open_chart()

    def _schedule_chart_refresh(self, delay_ms: int = 150) -> None:
        """Coalesce bursts of refresh requests into one regeneration."""
        if self._destroyed:
            return
        if self._chart_refresh_job is not None:
            try:
                self.after_cancel(self._chart_refresh_job)
            except Exception:
                pass
        self._chart_refresh_job = self.after(delay_ms, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        self._chart_refresh_job = None
        if not self._destroyed:
            self._regenerate_chart()

    def _invalidate_result_arrays(self) -> None:
        self._metric_keys: list[str] = []
        self._values: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._arrays_n = -1
        self._pareto_cache: tuple[bytes, np.ndarray] | None = None

    def _ensure_result_arrays(self) -> bool:
        """Vectorise ``all_results`` into numeric matrices (cached).

        Rebuilt only when the result count changes — i.e. once per batch
        during a run, and never again while the user flips axes or
        cutoffs, which is where the interactive cost used to be.
        """
        n = len(self.all_results)
        if n == 0:
            self._invalidate_result_arrays()
            return False
        if n == self._arrays_n and self._values is not None:
            return True
        keys, values, weights = results_to_arrays(
            self.all_results, len(self.sorted_tickers)
        )
        self._metric_keys = keys
        self._values = values
        self._weights = weights
        self._arrays_n = n
        self._pareto_cache = None
        return True

    def _apply_cutoffs(self, values: np.ndarray) -> np.ndarray:
        """Boolean keep-mask over the rows of *values* for current cutoffs.

        A cutoff on a metric no result carries drops everything — the
        rule can't be satisfied, and silently ignoring it would show a
        cloud that doesn't respect a filter the user set.
        """
        mask = np.ones(values.shape[0], dtype=bool)
        for metric, op, val in self._get_cutoffs():
            if metric not in self._metric_keys:
                return np.zeros(values.shape[0], dtype=bool)
            col = values[:, self._metric_keys.index(metric)]
            with np.errstate(invalid="ignore"):
                if op == ">=":
                    ok = col >= val
                elif op == "<=":
                    ok = col <= val
                elif op == ">":
                    ok = col > val
                elif op == "<":
                    ok = col < val
                else:
                    continue
            mask &= ok & np.isfinite(col)   # NaN never satisfies a cutoff
        return mask

    def _current_pareto_metrics(self) -> list[dict]:
        """``[{"name": ..., "direction": ...}, ...]`` from the checked
        Pareto Objectives boxes — falls back to ``cfg.PARETO_METRICS`` if
        the panel hasn't been built yet (shouldn't happen once __init__
        finishes, but keeps this method safe to call defensively)."""
        metric_vars = getattr(self, "_pareto_metric_vars", None)
        if not metric_vars:
            return list(cfg.PARETO_METRICS)
        return [
            {"name": name, "direction": _default_pareto_direction(name)}
            for name, var in metric_vars.items()
            if var.get()
        ]

    def _on_pareto_selection_changed(self) -> None:
        """A Pareto Objectives checkbox changed — the frontier for
        whatever's currently on screen is now stale."""
        self._pareto_cache = None
        if self.all_results:
            self._schedule_chart_refresh()

    def _pareto_indices(self, keep: np.ndarray) -> np.ndarray:
        """Frontier row-indices (into the full arrays) for the kept rows.

        Cached against the keep-mask so switching axes — which doesn't
        change the frontier, only how it is drawn — doesn't recompute it.
        The cache is dropped by ``_on_pareto_selection_changed`` whenever
        the objective checkboxes change, so a stale frontier from a
        different metric selection can't survive a toggle.
        """
        signature = keep.tobytes()
        if self._pareto_cache and self._pareto_cache[0] == signature:
            return self._pareto_cache[1]

        pareto_metrics = self._current_pareto_metrics()
        names = [o["name"] for o in pareto_metrics]
        dirs = [o["direction"] for o in pareto_metrics]
        present = [(n, d) for n, d in zip(names, dirs) if n in self._metric_keys]
        missing = [n for n in names if n not in self._metric_keys]
        if missing:
            _gui_log.warning(
                "[EXPLORER] Pareto objective(s) %s absent from every result "
                "(wrong horizon/block_size for this metric?) — excluded from "
                "the frontier for this run.", missing,
            )
        if not present:
            out = np.empty(0, dtype=np.intp)
            self._pareto_cache = (signature, out)
            return out

        rows = np.flatnonzero(keep)
        cols = [self._metric_keys.index(n) for n, _ in present]
        data_p = self._values[np.ix_(rows, cols)]
        # np.nan (not 0) for a missing objective: a fabricated "scored 0
        # here" would drag a portfolio onto the frontier it never earned.
        valid = np.all(np.isfinite(data_p), axis=1)
        if valid.sum() > 1:
            idx = compute_pareto(data_p[valid], [d for _, d in present])
            out = rows[np.flatnonzero(valid)[idx]]
        else:
            out = np.empty(0, dtype=np.intp)
        self._pareto_cache = (signature, out)
        return out

    def _hover_template(self) -> str:
        """One template shared by every point in the cloud.

        Column layout of ``customdata``: the weights (one per ticker, in
        ``sorted_tickers`` order — the click handler reads these back to
        rebuild the portfolio), then every metric in ``_metric_keys``.
        """
        lines = []
        for i, ticker in enumerate(self.sorted_tickers):
            lines.append(f"{ticker}: %{{customdata[{i}]:.1%}}")
        if lines:
            lines.append("────────")
        off = len(self.sorted_tickers)
        for j, key in enumerate(self._metric_keys):
            lines.append(f"{key}: %{{customdata[{off + j}]:.4f}}")
        return "<br>".join(lines) + "<extra></extra>"

    def _customdata(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), n_tickers + n_metrics) float32 matrix.

        float32 halves the payload and is far more precision than a
        hover label showing 4 decimals can use.
        """
        return np.hstack([self._weights[rows], self._values[rows]]).astype(np.float32)

    def _run_metadata_text(self) -> str:
        """What produced the cloud currently on screen — printed ON the
        chart (see the annotation in ``_regenerate_chart``), not just in
        the scrolling log, so a screenshot or a glance answers "what am I
        even looking at": which search, over what history, over what
        search space, with which seeds.
        """
        meta = self._last_run_meta
        params = self._last_run_params
        if not meta or not params:
            return ""

        date_start = params.get("date_start") or "earliest"
        date_end = params.get("date_end") or "latest"
        method = str(meta.get("method", "?")).upper()
        lines = [
            f"<b>{method} search</b> · {meta.get('n_portfolios', '?')} portfolios "
            f"· started {meta.get('started_at', '?')}",
            f"Dates: {date_start} → {date_end}",
            f"n_sim={params['n_sim']} · horizon={params['horizon_years']}y · "
            f"block={params['block_size']}mo"
            + (" · independent" if params.get("independent") else ""),
        ]
        if meta.get("grid_step") is not None:
            lines.append(f"Grid step: {meta['grid_step']}")
        if meta.get("n_generations") is not None:
            lines.append(f"Generations: {meta['n_generations']}")
        seed_bits = [f"seed={meta.get('seed')}"]
        if self._last_sim_seed is not None:
            seed_bits.append(f"sim_seed={self._last_sim_seed}")
        lines.append(" · ".join(seed_bits))

        space = meta.get("space") or []
        if space:
            if len(space) <= 8:
                space_txt = ", ".join(
                    f"{s['ticker']} [{s['lo']:.0%}-{s['hi']:.0%}]" for s in space
                )
            else:
                space_txt = f"{len(space)} assets — see log for bounds"
            lines.append(f"Search space: {space_txt}")
        return "<br>".join(lines)

    def _regenerate_chart(self, ignore_cutoffs: bool = False) -> bool:
        """Generate (or regenerate) the Plotly interactive scatter chart."""
        if not self._ensure_result_arrays():
            return False

        x_key = self.x_metric_var.get()
        y_key = self.y_metric_var.get()
        if x_key not in self._metric_keys or y_key not in self._metric_keys:
            self.chart_status.config(
                text=f"'{x_key}' / '{y_key}' not present in any result "
                     f"(wrong horizon/block_size for this metric?)"
            )
            return False

        n_total = self._values.shape[0]
        keep = (np.ones(n_total, dtype=bool) if ignore_cutoffs
                else self._apply_cutoffs(self._values))
        n_cut = int(keep.sum())
        if n_cut == 0:
            self.chart_status.config(text="All results filtered out")
            return False

        # A metric can be legitimately absent from a result (e.g. a
        # volatility window beyond the run's horizon) — plotting it as 0
        # would show a point that scored 0 on an axis it was never
        # actually evaluated on. Drop those points instead and say so.
        xi = self._metric_keys.index(x_key)
        yi = self._metric_keys.index(y_key)
        keep &= np.isfinite(self._values[:, xi]) & np.isfinite(self._values[:, yi])
        n_missing = n_cut - int(keep.sum())
        if not keep.any():
            self.chart_status.config(
                text=f"'{x_key}' / '{y_key}' not present in any result "
                     f"(wrong horizon/block_size for this metric?)"
            )
            return False

        rows = np.flatnonzero(keep)
        pareto_rows = self._pareto_indices(keep)
        pareto_only = self.pareto_only_var.get()

        # Thin the cloud, but never the frontier — it is the part the eye
        # (and the click-to-library flow) is actually looking for, and it
        # is small. Frontier points are drawn by their own trace, so they
        # are excluded from the cloud rather than plotted twice.
        if pareto_only:
            cloud_rows = np.empty(0, dtype=np.intp)
            n_cloud_full = 0
        else:
            cloud_rows = np.setdiff1d(rows, pareto_rows, assume_unique=True)
            n_cloud_full = cloud_rows.size
            thin = thin_scatter(
                self._values[cloud_rows, xi], self._values[cloud_rows, yi],
                self._max_points(),
            )
            cloud_rows = cloud_rows[thin]

        template = self._hover_template()
        fig = go.Figure()

        if not pareto_only:
            fig.add_trace(go.Scattergl(
                x=self._values[cloud_rows, xi],
                y=self._values[cloud_rows, yi],
                mode="markers",
                marker=dict(
                    size=5,
                    color="rgba(74, 74, 74, 0.5)",
                    line=dict(width=0),
                ),
                customdata=self._customdata(cloud_rows),
                hovertemplate=template,
                name="Portfolios (click to add to library)",
            ))

        # Pareto front
        if pareto_rows.size:
            px = self._values[pareto_rows, xi]
            py = self._values[pareto_rows, yi]
            order = np.argsort(px, kind="stable")
            pareto_rows = pareto_rows[order]
            fig.add_trace(go.Scattergl(
                x=px[order], y=py[order],
                mode="markers+lines",
                marker=dict(size=8, color="black", symbol="diamond"),
                line=dict(color="black", width=1.5),
                customdata=self._customdata(pareto_rows),
                hovertemplate=template,
                name="Pareto Front (click to add to library)",
            ))

        # Overlay cached portfolios (library items)
        current_overlay_params = self._current_overlay_params()
        for name, cache_entry in self._overlay_cache.items():
            if current_overlay_params is not None:
                portfolio_for_key = _clean_portfolio(self.library.get(name) or {})
                expected_key = self._current_overlay_key(current_overlay_params, portfolio_for_key)
                if cache_entry.get("key") != expected_key:
                    continue
            metrics = cache_entry.get("metrics", {})
            cx = metrics.get(x_key)
            cy = metrics.get(y_key)
            if cx is None or cy is None:
                continue
            # An overlay computed via the fallback branch (no run loaded
            # yet, or search-space ticker set / seed doesn't match) used a
            # different historical window and Monte-Carlo draw than the
            # cloud around it — its position isn't meaningful relative to
            # the cloud, so it's drawn as an open, dimmed marker instead
            # of a solid one, with a warning in the hover text.
            comparable = cache_entry.get("comparable", True)
            marker_color = "crimson" if comparable else "rgba(220, 20, 60, 0.45)"
            marker_symbol = "star" if comparable else "star-open"
            # A handful of points, so a hand-built hover string is free
            # here — unlike the cloud, where it was the whole problem.
            composition = self.library.get(name) or {}
            hover_lines = [f"<b>{name}</b>"]
            if not comparable:
                hover_lines.append(
                    "⚠ computed on a different data window / random draw "
                    "than the cloud — not directly comparable"
                )
            if composition:
                hover_lines.append("────────")
                for t, w in sorted(composition.items()):
                    hover_lines.append(f"{t}: {w:.1%}")
            hover_lines.append("────────")
            for k in sorted(metrics.keys()):
                if k.startswith("_"):
                    continue
                v = metrics[k]
                hover_lines.append(
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                )
            # customdata keeps the cloud's layout (weights first, in
            # sorted_tickers order) so the click handler needs no special
            # case for overlay points.
            cd = [float(composition.get(t, 0.0)) for t in self.sorted_tickers]
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(size=16, symbol=marker_symbol, color=marker_color),
                text=[name],
                textposition="top right",
                textfont=dict(size=11, color="crimson"),
                name=name,
                customdata=[cd],
                hoverinfo="text",
                hovertext="<br>".join(hover_lines),
            ))

        n_drawn = cloud_rows.size + pareto_rows.size
        status = f"{n_total} results"
        if n_cut < n_total:
            status += f" ({n_cut} after filters"
            status += f", {n_missing} missing '{x_key}'/'{y_key}'" if n_missing else ""
            status += ")"
        if pareto_only:
            status += f" — {pareto_rows.size} on the Pareto front"
        elif cloud_rows.size < n_cloud_full:
            status += f" — {n_drawn} plotted (thinned)"
        bar_text = f"<b>{n_drawn:,}</b> points drawn — {status}"

        meta_text = self._run_metadata_text()
        fig.update_layout(
            title=dict(
                text=f"{y_key}  vs  {x_key}",
                font=dict(size=16),
            ),
            xaxis_title=x_key,
            yaxis_title=y_key,
            template="plotly_white",
            hovermode="closest",
            autosize=True,
            margin=dict(l=80, r=40, t=60, b=60),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            dragmode="zoom",   # box-zoom by default
            # Keyed on the axes: a live refresh that only changed the data
            # (new batch, new cutoff) keeps the user's zoom, while picking
            # a different metric resets the view as it should.
            uirevision=f"{x_key}|{y_key}",
            # What is this cloud actually showing? Dates, search space,
            # method, seeds — printed ON the chart itself (not just in the
            # scrolling log) so a screenshot or a glance answers it.
            annotations=(
                [dict(
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    xanchor="left", yanchor="top",
                    text=meta_text,
                    showarrow=False,
                    align="left",
                    font=dict(size=11, color="#333"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1, borderpad=6,
                )] if meta_text else []
            ),
            # Consumed by the already-open tab's own status bar (see
            # _chart_shell_html) instead of a value baked into that page
            # at load time — the old code froze the point count at
            # whatever it was when the tab was FIRST opened, so a live
            # run's "N points drawn" never moved even though the chart
            # itself kept redrawing correctly. Every field the bar needs
            # now travels inside the same figure.json the page already
            # re-fetches on every version bump.
            meta=dict(bar_text=bar_text, meta_text=meta_text),
        )

        self._write_chart(fig)
        self.chart_status.config(text=status + " — chart ready")
        return True

    # ── Chart files ───────────────────────────────────────────────────────

    def _write_chart(self, fig: go.Figure) -> None:
        """Write the chart to ``results/``.

        With the click server up the page is a small shell that fetches
        ``figure.json`` and re-``Plotly.react``s whenever the version
        counter changes, so a refresh costs one JSON fetch in an already
        open tab. Without it (server failed to bind) there is no origin to
        fetch from, so fall back to a self-contained HTML file.
        """
        results_dir = cfg.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        chart_path = os.path.join(results_dir, "scatter.html")

        if not self._click_port:
            fig.write_html(chart_path, auto_open=False, div_id="bootstrap-scatter")
            self._chart_path = chart_path
            return

        self._write_plotly_js(results_dir)
        json_path = os.path.join(results_dir, "figure.json")
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(pio.to_json(fig))
        os.replace(tmp_path, json_path)   # never serve a half-written figure

        self._chart_version += 1
        with open(chart_path, "w", encoding="utf-8") as fh:
            fh.write(self._chart_shell_html())
        self._chart_path = chart_path

    def _write_cleared_chart(self, message: str) -> None:
        """Push an explicit placeholder chart — used the moment a new Run
        starts and on Reset — so an already-open tab is never left
        displaying a previous (or just-discarded) run's cloud with
        nothing on screen to say it's stale. Still carries whatever run
        metadata is current (there may be none, e.g. right after Reset).
        """
        meta_text = self._run_metadata_text()
        lines = [message] + ([meta_text] if meta_text else [])
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor="center", yanchor="middle",
                text="<br>".join(lines),
                showarrow=False,
                align="center",
                font=dict(size=14, color="#666"),
            )],
            uirevision="cleared",
            meta=dict(bar_text=message, meta_text=meta_text),
        )
        self._write_chart(fig)

    @staticmethod
    def _write_plotly_js(results_dir: str) -> None:
        """Drop plotly.min.js next to the chart, once per plotly version.

        Serving it as its own file (instead of inlining ~3.5 MB into every
        rewrite of the page) means the browser parses it once and caches
        it — and a live update doesn't re-parse it at all.
        """
        js_path = os.path.join(results_dir, "plotly.min.js")
        stamp_path = os.path.join(results_dir, "plotly.version")
        want = plotly.__version__
        try:
            with open(stamp_path, encoding="utf-8") as fh:
                have = fh.read().strip()
        except OSError:
            have = None
        if have == want and os.path.exists(js_path):
            return
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write(get_plotlyjs())
        with open(stamp_path, "w", encoding="utf-8") as fh:
            fh.write(want)

    def _chart_shell_html(self) -> str:
        """The page written to ``scatter.html`` — a shell, not a snapshot.

        This is written once per regeneration but the already-open tab
        never re-fetches it: it only polls ``/version`` and re-fetches
        ``figure.json``. So nothing that can change between regenerations
        (point count, run metadata) may be baked in as a JS literal here —
        it has to travel inside ``figure.json`` itself (``fig.layout.meta``)
        and be read back out on every ``load()``, or the bar goes stale
        the moment a second regeneration happens while the tab stays open.
        """
        tickers = json.dumps(self.sorted_tickers)
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Space Explorer</title>
<script src="plotly.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; font-family: system-ui, sans-serif; }}
  #bootstrap-scatter {{ width: 100vw; height: calc(100vh - 26px); }}
  #bar {{ height: 26px; line-height: 26px; padding: 0 10px; font-size: 12px;
         color: #555; background: #f4f4f4; border-top: 1px solid #ddd;
         white-space: nowrap; overflow: hidden; }}
  #bar b {{ color: #222; }}
</style>
</head>
<body>
<div id="bootstrap-scatter"></div>
<div id="bar">loading…</div>
<script>
(function () {{
  var TICKERS = {tickers};
  var div = document.getElementById('bootstrap-scatter');
  var bar = document.getElementById('bar');
  var CONFIG = {{responsive: true, scrollZoom: true, displaylogo: false}};
  var version = null;
  var bound = false;
  var busy = false;

  function status(msg) {{ bar.innerHTML = msg; }}

  function bind() {{
    if (bound || !div.on) return;
    bound = true;
    div.on('plotly_click', function (data) {{
      if (!data.points || !data.points.length) return;
      var cd = data.points[0].customdata;
      if (!cd || !cd.length) return;
      var portfolio = {{}};
      for (var i = 0; i < TICKERS.length && i < cd.length; i++) {{
        portfolio[TICKERS[i]] = cd[i];
      }}
      fetch('/click', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{portfolio: portfolio}})
      }}).catch(function (e) {{ console.warn('Click server error:', e); }});
    }});
  }}

  function load(v) {{
    busy = true;
    status('updating…');
    fetch('figure.json?v=' + v, {{cache: 'no-store'}}).then(function (r) {{ return r.json(); }})
      .then(function (fig) {{
        // react(), not newPlot(): same WebGL context, and the layout's
        // uirevision keeps the current zoom unless the axes changed.
        return Plotly.react(div, fig.data, fig.layout, CONFIG).then(function () {{ return fig; }});
      }})
      .then(function (fig) {{
        bind();
        version = v;
        busy = false;
        var meta = (fig.layout && fig.layout.meta) || {{}};
        var barText = meta.bar_text || 'chart updated';
        status(barText + ' · click a point to send it to the library · '
               + 'updated ' + new Date().toLocaleTimeString());
      }})
      .catch(function (e) {{
        busy = false;
        status('update failed: ' + e);
      }});
  }}

  function poll() {{
    if (busy) return;
    fetch('/version', {{cache: 'no-store'}})
      .then(function (r) {{ return r.json(); }})
      .then(function (d) {{ if (d.v !== version) load(d.v); }})
      .catch(function () {{}});
  }}

  poll();
  setInterval(poll, 1500);
}})();
</script>
</body>
</html>
"""

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
        self.compare_btn = StyledButton(
            tab_bar, text="Multi-Portfolio Comparison",
            command=lambda: self._switch_mode("compare"), style="Accent.TButton",
        )
        self.compare_btn.pack(side="left")

        self.sweep_btn = StyledButton(
            tab_bar, text="Block-Size Sensitivity",
            command=lambda: self._switch_mode("sweep"),
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
            self.compare_btn.configure(style="Accent.TButton")
            self.sweep_btn.configure(style="Ghost.TButton")
        else:
            self.compare_frame.pack_forget()
            self.sweep_frame.pack(fill="both", expand=True)
            self.sweep_btn.configure(style="Accent.TButton")
            self.compare_btn.configure(style="Ghost.TButton")

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
        self._shared_window: tuple[int, list[str]] | None = None
        self._destroyed = False

        self._build_ui()

    def destroy(self):
        self._destroyed = True
        super().destroy()

    def _build_ui(self):
        # The plot grid alone can be taller than the window (fixed geometry
        # from ui_state.json), which would silently clip the summary table
        # — including the drawdown rows — below the visible area. Wrapping
        # everything in a ScrollableFrame guarantees the table stays
        # reachable no matter how tall the plots render.
        scroll = ScrollableFrame(self, bg=BG)
        scroll.pack(fill="both", expand=True)
        outer = scroll.inner

        top = tk.Frame(outer, bg=BG)
        top.pack(fill="x", padx=PAD, pady=PAD)

        # Left config
        config = tk.Frame(top, bg=PANEL_BG, bd=1, relief="solid",
                          highlightbackground=BORDER, highlightthickness=1)
        config.pack(side="left", fill="y", padx=(0, PAD))

        FieldLabel(config, text="Select Portfolios", bg=PANEL_BG).pack(
            anchor="w", padx=PAD, pady=(PAD, 0)
        )
        self.portfolio_listbox = tk.Listbox(
            config, font=(theme.FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
            selectmode="multiple", height=6, bd=0, highlightthickness=0
        )
        self.portfolio_listbox.pack(fill="both", expand=True, padx=PAD, pady=2)
        self.refresh_list()

        self.warning_label = ttk.Label(
            config, text="", font=(theme.FONT_FAMILY[0], 8), style="Error.Card.TLabel",
        )
        self.warning_label.pack(padx=PAD)

        # Params
        params = tk.Frame(config, bg=PANEL_BG)
        params.pack(fill="x", padx=PAD, pady=2)

        self._param_entries = {}
        for label, key, default in [
            ("N Simulations", "n_sim", str(cfg.N_SIMULATIONS)),
            ("Horizon (Years)", "horizon", "10"),
            ("Block Size", "block", str(cfg.BLOCK_SIZE)),
            ("Seed", "seed", "42"),
            ("Date Start", "date_start", ""),
            ("Date End", "date_end", ""),
            ("Hist. Start (YYYY-MM)", "hist_start", ""),
        ]:
            FieldLabel(params, text=label, bg=PANEL_BG).pack(anchor="w")
            var = tk.StringVar(value=default)
            ttk.Entry(
                params, textvariable=var, font=(theme.FONT_FAMILY[0], 10), width=14,
            ).pack(anchor="w", pady=(0, 2))
            self._param_entries[key] = var

        self.independent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params, text="Independent Resampling",
            variable=self.independent_var, style="Card.TCheckbutton",
        ).pack(anchor="w", padx=PAD)

        self.error_label = ErrorLabel(config, bg=PANEL_BG)
        self.error_label.pack(fill="x", padx=PAD)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(config, variable=self.progress_var, maximum=100).pack(
            fill="x", padx=PAD, pady=2
        )
        self.progress_label = ttk.Label(
            config, text="", font=(theme.FONT_FAMILY[0], 8), style="Field.Card.TLabel",
        )
        self.progress_label.pack(padx=PAD)

        run_row = tk.Frame(config, bg=PANEL_BG)
        run_row.pack(fill="x", padx=PAD, pady=(2, PAD))
        self.run_btn = StyledButton(run_row, text="Run Bootstrap", command=self._run)
        self.run_btn.pack(side="left", anchor="w")
        self.stop_btn = StyledButton(run_row, text="Stop", command=self._stop)
        self.stop_btn.pack(side="left", anchor="w", padx=(4, 0))
        self.stop_btn.config(state="disabled")

        # Plot area
        self.plot_frame = tk.Frame(top, bg=BG)
        self.plot_frame.pack(side="left", fill="both", expand=True)

        # Bottom: summary table
        self.table_frame = tk.Frame(outer, bg=BG)
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
        if self.running:
            return  # a click while already running must never start a second worker
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
        self._shared_window = None
        self.progress_var.set(0)
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        independent = self.independent_var.get()

        thread = threading.Thread(
            target=self._worker,
            args=(selected, n_sim, horizon, block, seed,
                  date_start, date_end, hist_start, independent),
            daemon=True,
        )
        thread.start()
        self.after(200, self._poll)

    def _stop(self):
        self.running = False
        self._log_stop_requested()

    def _log_stop_requested(self):
        _gui_log.info("[SINGLE] Stop requested by user")

    def _worker(self, names, n_sim, horizon, block, seed,
                date_start, date_end, hist_start, independent):
        try:
            total = len(names)
            _gui_log.info("[SINGLE] Starting single-bootstrap comparison for %d portfolios: %s",
                          total, names)
            _gui_log.info("[SINGLE] Params: n_sim=%d  horizon=%dy  block=%d  seed=%s",
                          n_sim, horizon, block, seed)
            _gui_log.info("[SINGLE] Date filter: [%s, %s]  hist_start=%s",
                          date_start or "*", date_end or "*", hist_start or "None")

            # Every selected portfolio is evaluated on ONE shared
            # historical window (the intersection across the union of
            # their tickers). Loading each on its own window would rank
            # them against each other over DIFFERENT periods — silently
            # rewarding whichever portfolio happens to hold assets with
            # longer histories. Measured on this repo's data, that flips
            # the ranking outright (see
            # engine.data.load_portfolios_on_common_window). Independent
            # mode is exempt: it deliberately never aligns assets to a
            # common calendar at all.
            common_tickers = None
            common_matrix = None
            common_weights: dict[str, np.ndarray] = {}
            if not independent:
                selected_portfolios = {}
                for name in names:
                    pf = _clean_portfolio(self.library.get(name))
                    if not pf:
                        self.result_queue.put(
                            ("error", f"Portfolio '{name}' is empty or all-zero."))
                        return
                    selected_portfolios[name] = pf
                common_tickers, common_matrix, common_weights = (
                    load_portfolios_on_common_window(
                        selected_portfolios, cfg.USE_AFTER_TER_RETURNS,
                        date_start=date_start, date_end=date_end,
                    )
                )
                _gui_log.info(
                    "[SINGLE] Shared window across all %d portfolios: %d months "
                    "(%d tickers: %s)", total, common_matrix.shape[0],
                    len(common_tickers), common_tickers)
                self.result_queue.put(
                    ("window_info", common_matrix.shape[0], list(common_tickers)))

            for i, name in enumerate(names):
                if not self.running:
                    _gui_log.info("[SINGLE] Run cancelled by user")
                    break
                _gui_log.info("[SINGLE] ── Portfolio %d/%d: '%s' ──", i + 1, total, name)
                portfolio = _clean_portfolio(self.library.get(name))
                if not portfolio:
                    _gui_log.error("[SINGLE] Portfolio '%s' is empty or all-zero after cleaning!",
                                   name)
                    self.result_queue.put(("error", f"Portfolio '{name}' is empty or all-zero."))
                    return
                _gui_log.info("[SINGLE] Cleaned portfolio: %s", portfolio)

                _gui_log.info("[SINGLE] Loading return data for '%s'...", name)
                t0 = time.perf_counter()
                rng = np.random.default_rng(seed)

                if independent:
                    weights_arr, returns_list = load_independent_returns(
                        portfolio, cfg.USE_AFTER_TER_RETURNS,
                        date_start=date_start, date_end=date_end,
                    )
                    _gui_log.info("[SINGLE] Independent data loaded in %.3fs: "
                                  "per-asset lengths=%s  weights=%s",
                                  time.perf_counter() - t0,
                                  [len(r) for r in returns_list], weights_arr)

                    _gui_log.info("[SINGLE] Running independent Monte-Carlo simulation: "
                                  "%d sims × %d months (block=%d)...",
                                  n_sim, horizon * 12, block)
                    t1 = time.perf_counter()
                    paths = simulate_independent(weights_arr, returns_list,
                                                 n_sim, horizon * 12, rng,
                                                 block_size=block)
                    # Also load aligned matrix for historical overlay
                    _, ret_matrix = load_all_returns(
                        portfolio, cfg.USE_AFTER_TER_RETURNS,
                        date_start=date_start, date_end=date_end,
                    )
                    metrics_tickers = sorted(portfolio.keys())
                else:
                    # Shared window across every selected portfolio — see
                    # the block above. weights_arr is zero-padded to the
                    # union ticker list so all portfolios index the same
                    # columns of the same matrix.
                    weights_arr = common_weights[name]
                    ret_matrix = common_matrix
                    metrics_tickers = common_tickers
                    _gui_log.info("[SINGLE] Shared-window data: matrix=%s  weights=%s",
                                  ret_matrix.shape, weights_arr)

                    _gui_log.info("[SINGLE] Running Monte-Carlo simulation: "
                                  "%d sims × %d months (block=%d)...",
                                  n_sim, horizon * 12, block)
                    t1 = time.perf_counter()
                    paths = simulate(weights_arr, ret_matrix, n_sim, horizon * 12, rng,
                                     block_size=block)
                _gui_log.info("[SINGLE] Simulation done in %.3fs: paths=%s",
                              time.perf_counter() - t1, paths.shape)

                _gui_log.info("[SINGLE] Computing metrics...")
                t2 = time.perf_counter()
                metrics = compute_metrics(
                    paths, horizon, horizon * 12, block_size=block, weights=weights_arr,
                    tickers=metrics_tickers,
                )
                _gui_log.info("[SINGLE] Metrics computed in %.3fs: %s",
                              time.perf_counter() - t2,
                              {k: f"{v:.4f}" if isinstance(v, float) else v
                               for k, v in metrics.items()})

                # Compute historical performance
                _gui_log.info("[SINGLE] Computing historical performance...")
                if hist_start:
                    # Use user-specified historical start for horizon years
                    try:
                        yr, mo = int(hist_start.split("-")[0]), int(hist_start.split("-")[1])
                        # date filters are INCLUSIVE at both ends, so the end
                        # month must be one month before the horizon-years-later
                        # anniversary — otherwise [hist_start, end] spans
                        # horizon*12 + 1 months, one more than the simulation
                        # itself (horizon*12), and the overlay silently runs
                        # a month longer than the fan chart it's drawn on.
                        if mo == 1:
                            end_yr, end_mo = yr + horizon - 1, 12
                        else:
                            end_yr, end_mo = yr + horizon, mo - 1
                        hist_end = f"{end_yr:04d}-{end_mo:02d}"
                        _gui_log.info("[SINGLE] Historical window: %s → %s", hist_start, hist_end)
                        # Union ticker set again (see the shared-window
                        # block above): every compared portfolio's
                        # historical line must cover the same months, or
                        # the lines drawn on one chart start on different
                        # dates and span different periods.
                        _, ret_hist, w_hist_map = load_portfolios_on_common_window(
                            selected_portfolios, cfg.USE_AFTER_TER_RETURNS,
                            date_start=hist_start, date_end=hist_end,
                        )
                        w_hist = w_hist_map[name]
                        # errstate: Apple Accelerate BLAS false positives
                        with np.errstate(all="ignore"):
                            port_hist = ret_hist @ w_hist
                        cum_hist = np.cumprod(1.0 + port_hist)
                        cum_hist = np.insert(cum_hist, 0, 1.0)
                        hist_label = f"Hist. ({hist_start} \u2192 {hist_end})"
                        _gui_log.info("[SINGLE] Historical path computed: %d months", len(cum_hist) - 1)
                    except Exception as hist_exc:
                        _gui_log.warning("[SINGLE] Hist window failed (%s), falling back to full data",
                                         hist_exc)
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
                    _gui_log.info("[SINGLE] Using full historical data: %d months", len(cum_hist) - 1)

                ann_ret = paths[:, -1] ** (1.0 / horizon) - 1.0
                _gui_log.info("[SINGLE] Portfolio '%s' done: "
                              "ann_ret median=%.4f  p5=%.4f  p95=%.4f",
                              name, float(np.median(ann_ret)),
                              float(np.percentile(ann_ret, 5)),
                              float(np.percentile(ann_ret, 95)))

                self.result_queue.put((
                    "result", name, {
                        "metrics": metrics,
                        "paths": paths,
                        "ann_ret": ann_ret,
                        "weights_arr": weights_arr,
                        "historical": cum_hist,
                        "hist_label": hist_label,
                        "block_size": block,
                    },
                    i + 1, total
                ))
            _gui_log.info("[SINGLE] All %d portfolios processed — sending 'done' signal",
                          total)
            self.result_queue.put(("done",))
        except Exception as e:
            _gui_log.error("[SINGLE] EXCEPTION in worker: %s", e, exc_info=True)
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
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    return
                elif msg[0] == "window_info":
                    _, n_months, tickers = msg
                    self._shared_window = (n_months, tickers)
                elif msg[0] == "result":
                    _, name, data, done, total = msg
                    self.results[name] = data
                    self.progress_var.set((done / total) * 100)
                    self.progress_label.config(text=f"{done}/{total} portfolios")
                elif msg[0] == "done":
                    self.running = False
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
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

        # Explicit Figure (not plt.subplots): never registers with pyplot's
        # global figure manager, so repeated Runs can't leak memory.
        fig = Figure(figsize=(12, 8), dpi=theme.PLOT_DPI, facecolor=BG)
        axes = fig.subplots(2, 2)
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
            block_size = self.results[name].get("block_size", 1)
            # Each path column spans block_size months, so its year label is
            # k*block_size/12 — NOT np.linspace(0, horizon, n_steps), which
            # spreads the columns evenly across [0, horizon] regardless of
            # block_size. Those only coincide when block_size divides the
            # horizon evenly; otherwise (the final block gets truncated —
            # see engine.simulation.simulate) linspace stretches the last
            # column out to the nominal horizon even though fewer months
            # were actually simulated.
            years = np.arange(n_steps) * block_size / 12.0

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
            # One point per month, exactly 1/12 year apart (point 0 = t0).
            # np.linspace(0, len(hist)/12, len(hist)) instead spaces
            # len(hist) points across [0, len(hist)/12] — dividing by
            # (len(hist) - 1), not len(hist) — which stretches the
            # x-axis slightly past where each month actually falls.
            hist_years = np.arange(len(hist)) / 12.0
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
        # State the shared window explicitly: these portfolios are only
        # comparable because they were all measured on the same months,
        # and that window is set by whichever SELECTED ticker has the
        # shortest history — so it changes as the selection changes.
        if self._shared_window is not None:
            n_months, tickers = self._shared_window
            FieldLabel(
                frame,
                text=(f"All portfolios evaluated on the same {n_months} months "
                      f"({n_months / 12:.1f}y) — the overlap across "
                      f"{len(tickers)} tickers in the current selection"),
                bg=PANEL_BG,
            ).pack(anchor="w", padx=PAD)

        # Vertical scrolling is handled by the outer ScrollableFrame this
        # table lives in, so this canvas only needs to grow tall enough to
        # show every metric row (including drawdown) without clipping —
        # it keeps its own horizontal scrollbar for when many portfolios
        # are compared side by side.
        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0)
        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        table_inner = tk.Frame(canvas, bg=PANEL_BG)

        def _on_inner_configure(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"),
                             height=table_inner.winfo_reqheight())

        table_inner.bind("<Configure>", _on_inner_configure)
        canvas.create_window((0, 0), window=table_inner, anchor="nw")
        canvas.configure(xscrollcommand=h_scroll.set)

        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Header row
        ttk.Label(
            table_inner, text="METRIC", font=(theme.FONT_FAMILY[0], 9, "bold"),
            style="Card.TLabel", anchor="w", width=30,
        ).grid(row=0, column=0, sticky="w", padx=4, pady=1)
        for ci, name in enumerate(names):
            ttk.Label(
                table_inner, text=name, font=(theme.FONT_FAMILY[0], 9, "bold"),
                style="Card.TLabel", foreground=PORTFOLIO_COLORS[ci % len(PORTFOLIO_COLORS)],
                anchor="e", width=16,
            ).grid(row=0, column=ci + 1, sticky="e", padx=4, pady=1)

        sep = tk.Frame(table_inner, bg=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=len(names) + 1, sticky="ew", pady=2)

        for ri, key in enumerate(metric_keys):
            ttk.Label(
                table_inner, text=key, font=(theme.FONT_FAMILY[0], 9),
                style="Card.TLabel", anchor="w",
            ).grid(row=ri + 2, column=0, sticky="w", padx=4, pady=0)

            for ci, name in enumerate(names):
                v = self.results[name]["metrics"].get(key, "")
                if isinstance(v, float):
                    txt = f"{v:.4f}"
                else:
                    txt = str(v)
                ttk.Label(
                    table_inner, text=txt, font=(theme.FONT_MONO[0], 9),
                    style="Card.TLabel", anchor="e",
                ).grid(row=ri + 2, column=ci + 1, sticky="e", padx=4, pady=0)

            if ri % 2 == 0:
                for col in range(len(names) + 1):
                    try:
                        w = table_inner.grid_slaves(row=ri + 2, column=col)[0]
                        w.config(background="#FAFAF5")
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
            config, font=(theme.FONT_FAMILY[0], 10), bg=PANEL_BG, fg=TEXT,
            selectmode="browse", height=5, bd=0, highlightthickness=0
        )
        self.portfolio_listbox.pack(fill="x", padx=PAD, pady=2)
        self.refresh_list()

        self._params = {}
        for label, key, default in [
            ("Block Size Min", "bs_min", "1"),
            ("Block Size Max", "bs_max", "36"),
            ("N Simulations", "n_sim", str(cfg.N_SIMULATIONS)),
            ("Horizon (Years)", "horizon", "10"),
            ("Seed", "seed", "42"),
            ("Date Start", "date_start", ""),
            ("Date End", "date_end", ""),
        ]:
            FieldLabel(config, text=label, bg=PANEL_BG).pack(anchor="w", padx=PAD)
            var = tk.StringVar(value=default)
            ttk.Entry(
                config, textvariable=var, font=(theme.FONT_FAMILY[0], 10), width=14,
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
        self.progress_label = ttk.Label(
            config, text="", font=(theme.FONT_FAMILY[0], 8), style="Field.Card.TLabel",
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
            _gui_log.info("[SWEEP] Starting block-size sweep: bs=%d..%d  n_sim=%d  horizon=%dy",
                          bs_min, bs_max, n_sim, horizon)
            _gui_log.info("[SWEEP] Loading return data...")
            t0 = time.perf_counter()
            weights_arr, ret_matrix = load_all_returns(
                portfolio, cfg.USE_AFTER_TER_RETURNS,
                date_start=date_start, date_end=date_end,
            )
            sweep_tickers = sorted(portfolio.keys())
            _gui_log.info("[SWEEP] Data loaded in %.3fs: matrix=%s",
                          time.perf_counter() - t0, ret_matrix.shape)
            block_sizes = list(range(bs_min, bs_max + 1))
            total = len(block_sizes)
            records = []

            for i, bs in enumerate(block_sizes):
                if not self.running:
                    _gui_log.info("[SWEEP] Cancelled by user at block_size=%d", bs)
                    break
                _gui_log.info("[SWEEP] Block size %d/%d (bs=%d): simulating %d paths...",
                              i + 1, total, bs, n_sim)
                t1 = time.perf_counter()
                rng = np.random.default_rng(seed)
                m = run_bootstrap_preloaded(
                    weights_arr, ret_matrix,
                    n_sim=n_sim, horizon_years=horizon, block_size=bs, rng=rng,
                    tickers=sweep_tickers,
                )
                m["block_size"] = bs
                records.append(m)
                _gui_log.info("[SWEEP] Block size %d done in %.3fs", bs, time.perf_counter() - t1)
                self.result_queue.put(("progress", i + 1, total))

            df = pd.DataFrame(records).set_index("block_size")
            _gui_log.info("[SWEEP] Sweep complete: %d block sizes evaluated", len(records))
            self.result_queue.put(("done", df))

        except Exception as e:
            _gui_log.error("[SWEEP] EXCEPTION: %s", e, exc_info=True)
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

        fig = Figure(figsize=(9, 6), facecolor=BG, dpi=theme.PLOT_DPI)
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

_UI_STATE_FILE = os.path.join(BASE_DIR, ".ui_state.json")

_SECTIONS = [
    ("builder", "1  PORTFOLIO BUILDER"),
    ("explorer", "2  SPACE EXPLORER"),
    ("single", "3  SINGLE BOOTSTRAP"),
    ("lifecycle", "4  LIFE STRATEGY"),
]


class BootstrapApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Portfolio Bootstrap Analyser")
        theme.init_theme(self)
        self.configure(bg=BG)
        self.minsize(1100, 700)
        self._restore_geometry()

        self.library = PortfolioLibrary(BASE_DIR, on_error=self._on_library_error)

        self._active_key: Optional[str] = None
        self._section_factories = {
            "builder": lambda parent: PortfolioBuilderSection(parent, self.library),
            "explorer": lambda parent: SpaceExplorerSection(parent, self.library),
            "single": lambda parent: SingleBootstrapSection(parent, self.library),
            "lifecycle": self._make_lifecycle_section,
        }
        self.sections: dict[str, tk.Widget] = {}

        self._build_nav()
        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)

        from bootstrap_gui.widgets import StatusBar
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill="x", side="bottom")
        self.library.on_change(self._refresh_status_bar)
        self._refresh_status_bar()

        self._show_section("builder")

        for i, (key, _label) in enumerate(_SECTIONS, start=1):
            self.bind(f"<Command-Key-{i}>", lambda e, k=key: self._show_section(k))
            self.bind(f"<Control-Key-{i}>", lambda e, k=key: self._show_section(k))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(300, self._poll_status_log)

    def _make_lifecycle_section(self, parent):
        from bootstrap_gui.sections.lifecycle import LifeStrategySection
        return LifeStrategySection(parent, self.library)

    def _on_library_error(self, msg: str) -> None:
        _gui_log.error("[LIBRARY] %s", msg)
        if hasattr(self, "status_bar"):
            self.status_bar.set_message(msg)

    def _restore_geometry(self) -> None:
        geometry = "1400x900"
        try:
            if os.path.exists(_UI_STATE_FILE):
                with open(_UI_STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                geometry = state.get("geometry", geometry)
        except (OSError, json.JSONDecodeError) as e:
            _gui_log.warning("[UI_STATE] Could not read %s: %s", _UI_STATE_FILE, e)
        self.geometry(geometry)
        self.update_idletasks()
        # Centre on screen if this is a fresh install (no saved state).
        if not os.path.exists(_UI_STATE_FILE):
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h = 1400, 900
            self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _save_geometry(self) -> None:
        try:
            tmp_path = _UI_STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"geometry": self.geometry()}, f)
            os.replace(tmp_path, _UI_STATE_FILE)
        except OSError as e:
            _gui_log.warning("[UI_STATE] Could not save %s: %s", _UI_STATE_FILE, e)

    def _on_close(self) -> None:
        self._save_geometry()
        self.destroy()

    def _refresh_status_bar(self) -> None:
        if hasattr(self, "status_bar"):
            n = len(self.library.names())
            self.status_bar.set_detail(f"{n} portfolio{'s' if n != 1 else ''} in library")

    def _poll_status_log(self) -> None:
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        try:
            while True:
                msg = _gui_log_queue.get_nowait()
                if hasattr(self, "status_bar"):
                    self.status_bar.set_message(msg)
        except queue.Empty:
            pass
        self.after(300, self._poll_status_log)

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG)
        nav.pack(fill="x", padx=0, pady=0)

        tk.Frame(nav, bg=BORDER, height=1).pack(fill="x", side="bottom")

        self.nav_buttons: dict[str, StyledButton] = {}
        for key, label in _SECTIONS:
            btn = StyledButton(nav, text=label, command=lambda k=key: self._show_section(k))
            btn.pack(side="left")
            self.nav_buttons[key] = btn

    def _show_section(self, key: str) -> None:
        if key not in self._section_factories:
            return
        if key not in self.sections:
            try:
                widget = self._section_factories[key](self.container)
            except Exception as e:
                _gui_log.error("[APP] Failed to build section '%s': %s", key, e, exc_info=True)
                if hasattr(self, "status_bar"):
                    self.status_bar.set_message(f"Could not open '{key}': {e}")
                return
            self.sections[key] = widget

        for k, widget in self.sections.items():
            if k != key:
                widget.pack_forget()
        self.sections[key].pack(fill="both", expand=True)

        for k, btn in self.nav_buttons.items():
            btn.configure(style="Accent.TButton" if k == key else "Ghost.TButton")
        self._active_key = key


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = BootstrapApp()
    app.mainloop()
