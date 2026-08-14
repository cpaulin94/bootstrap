"""bootstrap_gui.theme — design tokens, ttk styling, and matplotlib helpers.

Call :func:`init_theme` exactly once, right after creating the ``tk.Tk()``
root and before building any widgets. It resolves the actual font family
available on this machine, computes a DPI scale factor, and configures a
single ``ttk.Style`` used by every section.

The colour palette itself is unchanged from the original design (Swiss,
light, high-contrast) — this module only centralises it and makes the
``ttk`` widgets actually use it, which the original ad-hoc ``tk.*`` styling
never did consistently.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Colour tokens (unchanged values from the original gui.py)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Spacing scale
# ═══════════════════════════════════════════════════════════════════════════════

SPACE_XS = 2
SPACE_S = 4
SPACE_M = 8
SPACE_L = 16
SPACE_XL = 24
PAD = SPACE_M  # kept for the many existing call sites written against `PAD`

# ═══════════════════════════════════════════════════════════════════════════════
# Fonts — resolved at runtime (Tk silently substitutes if you don't check)
# ═══════════════════════════════════════════════════════════════════════════════

_SANS_CANDIDATES = ("Helvetica Neue", "Helvetica", "Arial Narrow", "Arial", "TkDefaultFont")
_MONO_CANDIDATES = ("Menlo", "Consolas", "Courier New", "Courier", "TkFixedFont")

# Populated by init_theme(); usable as sane defaults even if init_theme()
# has not run yet (e.g. in a headless unit test importing this module).
FONT_FAMILY: tuple[str, ...] = _SANS_CANDIDATES
FONT_MONO: tuple[str, ...] = _MONO_CANDIDATES
UI_SCALE: float = 1.0
PLOT_DPI: int = 100

FONT_LIGHT = (FONT_FAMILY[0], 10)
FONT_REG = (FONT_FAMILY[0], 11)
FONT_MED = (FONT_FAMILY[0], 12, "bold")
FONT_LABEL = (FONT_FAMILY[0], 9)
FONT_LABEL_UP = (FONT_FAMILY[0], 8)
FONT_SMALL = (FONT_FAMILY[0], 9)
FONT_MONO_REG = (FONT_MONO[0], 10)


def resolve_font_family(root: tk.Misc, candidates: tuple[str, ...]) -> str:
    """Return the first font in *candidates* actually installed on this machine.

    The last candidate in every list passed by this module is a Tk logical
    font (``TkDefaultFont`` / ``TkFixedFont``), which always resolves, so
    this never falls through to nothing.
    """
    available = set(tkfont.families(root))
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


def _label_text(s: str) -> str:
    """Convert label to uppercase tracked style (unchanged convention)."""
    return s.upper()


# ═══════════════════════════════════════════════════════════════════════════════
# Init — call once after root creation
# ═══════════════════════════════════════════════════════════════════════════════

def init_theme(root: tk.Tk) -> ttk.Style:
    """Resolve fonts, compute the DPI scale, and configure ``ttk.Style``.

    Must be called once, immediately after constructing the ``tk.Tk()``
    root, before any widget is built.
    """
    global FONT_FAMILY, FONT_MONO, UI_SCALE, PLOT_DPI
    global FONT_LIGHT, FONT_REG, FONT_MED, FONT_LABEL, FONT_LABEL_UP, FONT_SMALL, FONT_MONO_REG

    sans = resolve_font_family(root, _SANS_CANDIDATES)
    mono = resolve_font_family(root, _MONO_CANDIDATES)
    FONT_FAMILY = (sans,) + _SANS_CANDIDATES
    FONT_MONO = (mono,) + _MONO_CANDIDATES

    FONT_LIGHT = (sans, 10)
    FONT_REG = (sans, 11)
    FONT_MED = (sans, 12, "bold")
    FONT_LABEL = (sans, 9)
    FONT_LABEL_UP = (sans, 8)
    FONT_SMALL = (sans, 9)
    FONT_MONO_REG = (mono, 10)

    try:
        UI_SCALE = root.winfo_fpixels("1i") / 72.0
        root.tk.call("tk", "scaling", UI_SCALE)
    except tk.TclError:
        UI_SCALE = 1.0
    PLOT_DPI = max(100, round(100 * UI_SCALE))

    return init_style(root, sans, mono)


def init_style(root: tk.Tk, sans: str, mono: str) -> ttk.Style:
    """Configure a single ``ttk.Style`` shared by every section."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=TEXT, font=(sans, 10))

    # Frames
    style.configure("Card.TFrame", background=PANEL_BG)
    style.configure("Panel.TFrame", background=BG)

    # Labels
    style.configure("TLabel", background=BG, foreground=TEXT, font=(sans, 10))
    style.configure("Card.TLabel", background=PANEL_BG, foreground=TEXT, font=(sans, 10))
    style.configure("Heading.TLabel", background=BG, foreground=TEXT, font=(sans, 12, "bold"))
    style.configure("Field.TLabel", background=BG, foreground=TEXT_SEC, font=(sans, 8))
    style.configure("Field.Card.TLabel", background=PANEL_BG, foreground=TEXT_SEC, font=(sans, 8))
    style.configure("Secondary.TLabel", background=BG, foreground=TEXT_SEC, font=(sans, 9))
    style.configure("Error.TLabel", background=BG, foreground=WARNING, font=(sans, 8))
    style.configure("Error.Card.TLabel", background=PANEL_BG, foreground=WARNING, font=(sans, 8))
    style.configure("Positive.TLabel", background=BG, foreground=POSITIVE, font=(sans, 9, "bold"))

    # Buttons
    style.configure(
        "Ghost.TButton", background=PANEL_BG, foreground=TEXT,
        bordercolor=TEXT, borderwidth=1, relief="solid",
        font=(sans, 9), padding=(10, 4), focuscolor=PANEL_BG,
    )
    style.map(
        "Ghost.TButton",
        background=[("disabled", PANEL_BG), ("pressed", ACCENT), ("active", ACCENT)],
        foreground=[("disabled", TEXT_SEC), ("pressed", PANEL_BG), ("active", PANEL_BG)],
        bordercolor=[("disabled", BORDER)],
    )

    style.configure(
        "Accent.TButton", background=ACCENT, foreground=PANEL_BG,
        bordercolor=ACCENT, borderwidth=1, relief="solid",
        font=(sans, 9, "bold"), padding=(10, 4), focuscolor=ACCENT,
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", BORDER), ("pressed", TEXT), ("active", TEXT)],
        foreground=[("disabled", TEXT_SEC)],
    )

    style.configure(
        "Danger.TButton", background=WARNING, foreground=PANEL_BG,
        bordercolor=WARNING, borderwidth=1, relief="solid",
        font=(sans, 8), padding=(6, 1), focuscolor=WARNING,
    )
    style.map("Danger.TButton", background=[("active", "#8E2A1E"), ("pressed", "#8E2A1E")])

    # Entries
    style.configure(
        "TEntry", fieldbackground=PANEL_BG, foreground=TEXT,
        bordercolor=BORDER, lightcolor=PANEL_BG, darkcolor=BORDER,
        insertcolor=TEXT, padding=(4, 3),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", ACCENT), ("invalid", WARNING)],
        fieldbackground=[("disabled", BG)],
    )

    # Combobox
    style.configure(
        "TCombobox", fieldbackground=PANEL_BG, background=PANEL_BG,
        foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER, padding=(4, 3),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PANEL_BG), ("disabled", BG)],
        foreground=[("readonly", TEXT), ("disabled", TEXT_SEC)],
        bordercolor=[("focus", ACCENT)],
    )
    root.option_add("*TCombobox*Listbox.background", PANEL_BG)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", PANEL_BG)
    root.option_add("*TCombobox*Listbox.font", (sans, 10))

    # Checkbutton / Radiobutton
    style.configure("TCheckbutton", background=BG, foreground=TEXT, font=(sans, 9),
                     focuscolor=BG)
    style.map("TCheckbutton", background=[("active", BG)], foreground=[("disabled", TEXT_SEC)])
    style.configure("Card.TCheckbutton", background=PANEL_BG, foreground=TEXT, font=(sans, 9),
                     focuscolor=PANEL_BG)
    style.map("Card.TCheckbutton", background=[("active", PANEL_BG)])
    style.configure("TRadiobutton", background=BG, foreground=TEXT, font=(sans, 9),
                     focuscolor=BG)
    style.map("TRadiobutton", background=[("active", BG)])

    # Treeview
    style.configure(
        "Treeview", background=PANEL_BG, fieldbackground=PANEL_BG,
        foreground=TEXT, rowheight=int(22 * (root.winfo_fpixels("1i") / 72.0)),
        font=(sans, 10), borderwidth=0,
    )
    style.configure("Treeview.Heading", background=BG, foreground=TEXT_SEC,
                     font=(sans, 8, "bold"), relief="flat")
    style.map("Treeview.Heading", background=[("active", BG)])
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", PANEL_BG)])

    # Progressbar
    style.configure(
        "Horizontal.TProgressbar", background=ACCENT, troughcolor=GRID_CLR,
        bordercolor=GRID_CLR, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6,
    )

    # Notebook (tab bar)
    style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=PANEL_BG, foreground=TEXT,
                     font=(sans, 10), padding=(16, 8), borderwidth=1)
    style.map(
        "TNotebook.Tab",
        background=[("selected", ACCENT)],
        foreground=[("selected", PANEL_BG)],
    )

    # Scrollbar
    style.configure("TScrollbar", background=PANEL_BG, troughcolor=BG,
                     bordercolor=BORDER, arrowcolor=TEXT_SEC)

    # PanedWindow sash
    style.configure("TPanedwindow", background=BG)
    style.configure("Sash", sashthickness=6)

    return style


# ═══════════════════════════════════════════════════════════════════════════════
# Matplotlib helpers — no pyplot state (never leaks Figures)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_style(ax: Any, title: str = "") -> None:
    """Apply the Swiss-scientific axes style (unchanged palette)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(TEXT)
    ax.spines["left"].set_linewidth(0.5)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(direction="out", length=3, width=0.5, colors=TEXT, labelsize=9)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.8)
    if title:
        ax.set_title(title, fontsize=10, fontweight="medium", loc="left", color=TEXT)


def make_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (8, 5)):
    """Create a ``matplotlib.figure.Figure`` explicitly.

    Unlike ``plt.subplots()``, this never registers the figure with
    pyplot's global state, so it can't leak memory across repeated Runs.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=PLOT_DPI, facecolor=BG, layout="constrained")
    axes = fig.subplots(nrows, ncols)
    return fig, axes
