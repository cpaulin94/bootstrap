"""bootstrap_gui.widgets — themed ttk widgets shared across sections.

Each widget keeps the constructor signature of its original ``tk``-based
counterpart in ``gui.py`` (``StyledButton(parent, text=..., command=...)``,
``FieldLabel(parent, text=..., bg=...)``, ...) so migrating call sites is a
drop-in replacement. Hover/pressed/disabled states are handled by the
``ttk.Style`` maps configured in :mod:`bootstrap_gui.theme` — no manual
``<Enter>``/``<Leave>`` bindings needed, which also removes a class of bugs
where the hover binding fought with the "active section" highlight.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from bootstrap_gui import theme


class StyledButton(ttk.Button):
    """Rectangular, 1px border, uppercase — ghost style, inverts on hover/press."""

    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kw):
        style = kw.pop("style", "Ghost.TButton")
        super().__init__(parent, text=text.upper(), command=command,
                          style=style, cursor="hand2", **kw)


class AccentButton(ttk.Button):
    """Primary call-to-action button — solid accent background."""

    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kw):
        style = kw.pop("style", "Accent.TButton")
        super().__init__(parent, text=text.upper(), command=command,
                          style=style, cursor="hand2", **kw)


class FieldLabel(ttk.Label):
    """Uppercase label for form fields."""

    def __init__(self, parent, text: str = "", **kw):
        bg = kw.pop("bg", theme.PANEL_BG)
        style = kw.pop("style", None) or (
            "Field.Card.TLabel" if bg == theme.PANEL_BG else "Field.TLabel"
        )
        super().__init__(parent, text=theme._label_text(text), style=style, anchor="w", **kw)


class ErrorLabel(ttk.Label):
    """Inline error message in WARNING colour. Never use messagebox for this."""

    def __init__(self, parent, **kw):
        bg = kw.pop("bg", theme.PANEL_BG)
        style = kw.pop("style", None) or (
            "Error.Card.TLabel" if bg == theme.PANEL_BG else "Error.TLabel"
        )
        super().__init__(parent, text="", style=style, anchor="w", **kw)

    def show(self, msg: str) -> None:
        self.config(text=msg)

    def clear(self) -> None:
        self.config(text="")


class NumericEntry(ttk.Entry):
    """Entry that actually validates: digits, one '.', an optional leading '-'."""

    def __init__(self, parent, **kw):
        vcmd = (parent.register(self._validate), "%P")
        super().__init__(parent, validate="key", validatecommand=vcmd, **kw)

    @staticmethod
    def _validate(proposed: str) -> bool:
        if proposed in ("", "-", ".", "-."):
            return True
        try:
            float(proposed)
            return True
        except ValueError:
            return False


class ScrollableFrame(ttk.Frame):
    """A vertically-scrollable container where the mouse wheel actually works
    while hovering it — not just by dragging the thin scrollbar.

    Wheel bindings are installed on <Enter> and removed on <Leave> so they
    don't steal scroll events from sibling widgets.
    """

    def __init__(self, parent, *, bg: Optional[str] = None, width: Optional[int] = None, **kw):
        super().__init__(parent, **kw)
        bg = bg or theme.BG
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, width=width)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.inner.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        if width is not None:
            self.canvas.bind(
                "<Configure>", lambda e: self.canvas.itemconfig(self._window_id, width=e.width)
            )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class StatusBar(ttk.Frame):
    """Bottom-of-window status line: last message + a right-aligned detail."""

    def __init__(self, parent, **kw):
        super().__init__(parent, style="Card.TFrame", **kw)
        self._message_var = tk.StringVar(value="Ready.")
        self._detail_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._message_var, style="Card.TLabel", anchor="w").pack(
            side="left", fill="x", expand=True, padx=theme.SPACE_M, pady=2
        )
        ttk.Label(self, textvariable=self._detail_var, style="Field.Card.TLabel", anchor="e").pack(
            side="right", padx=theme.SPACE_M, pady=2
        )

    def set_message(self, text: str) -> None:
        self._message_var.set(text)

    def set_detail(self, text: str) -> None:
        self._detail_var.set(text)
