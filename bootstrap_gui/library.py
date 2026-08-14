"""bootstrap_gui.library — JSON-backed named collections (portfolios, life plans).

Both libraries share the same robustness contract: writes are atomic
(write to ``.tmp``, then ``os.replace``) so a crash mid-write can never
corrupt the file, and load/save failures are logged and surfaced through
an optional ``on_error`` callback instead of being silently swallowed —
the original ``PortfolioLibrary`` caught every exception and said nothing,
which meant a corrupted JSON file quietly emptied the user's library.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger("bootstrap.gui.library")


class _JsonNamedStore:
    """Generic ``name -> JSON-serialisable value`` store with atomic persistence."""

    def __init__(self, path: str, *, on_error: Optional[Callable[[str], None]] = None):
        self.path = path
        self.items: dict[str, Any] = {}
        self._callbacks: list[Callable[[], None]] = []
        self._on_error = on_error
        self._load()

    def _report_error(self, msg: str) -> None:
        log.error("[STORE] %s", msg)
        if self._on_error:
            try:
                self._on_error(msg)
            except Exception:
                pass

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._report_error(
                f"Failed to load {os.path.basename(self.path)}: {e}. "
                f"Starting empty in this session — the file on disk was left untouched."
            )
            self.items = {}

    def _save(self) -> None:
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, indent=2)
            os.replace(tmp_path, self.path)
        except OSError as e:
            self._report_error(f"Failed to save {os.path.basename(self.path)}: {e}")

    def add(self, name: str, value: Any) -> None:
        self.items[name] = value
        self._save()
        self._notify()

    def delete(self, name: str) -> None:
        self.items.pop(name, None)
        self._save()
        self._notify()

    def rename(self, old: str, new: str) -> None:
        if old in self.items:
            self.items[new] = self.items.pop(old)
            self._save()
            self._notify()

    def names(self) -> list[str]:
        return list(self.items.keys())

    def get(self, name: str) -> Any:
        return self.items.get(name)

    def on_change(self, cb: Callable[[], None]) -> None:
        self._callbacks.append(cb)

    def _notify(self) -> None:
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                log.error("[STORE] on_change callback raised", exc_info=True)


class PortfolioLibrary(_JsonNamedStore):
    """Named portfolios: ``name -> {ticker: weight}``."""

    def __init__(self, base_dir: str, *, on_error: Optional[Callable[[str], None]] = None):
        super().__init__(os.path.join(base_dir, ".portfolio_library.json"), on_error=on_error)

    @property
    def portfolios(self) -> dict[str, dict[str, float]]:
        return self.items


class LifePlanLibrary(_JsonNamedStore):
    """Named life-strategy plans: ``name -> plan dict`` (see ``sections/lifecycle.py``
    for the ``LifePlan`` <-> dict conversion)."""

    def __init__(self, base_dir: str, *, on_error: Optional[Callable[[str], None]] = None):
        super().__init__(os.path.join(base_dir, ".life_plans.json"), on_error=on_error)
