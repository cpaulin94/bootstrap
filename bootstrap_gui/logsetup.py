"""bootstrap_gui.logsetup — verbose terminal logging + forwarding into the GUI.

Moved out of gui.py unchanged in behaviour: a ``bootstrap.*`` logger prints
everything to stderr, and INFO+ records are also forwarded into a
thread-safe queue that the GUI log panel / status bar drains on a timer.
"""

from __future__ import annotations

import logging
import queue

# Queue for forwarding engine log messages into the GUI log panel.
gui_log_queue: "queue.Queue[str]" = queue.Queue()


class QueueHandler(logging.Handler):
    """Logging handler that puts formatted records into a thread-safe queue."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            gui_log_queue.put(msg)
        except Exception:
            pass


def setup_logging() -> None:
    """Configure the ``bootstrap.*`` logger hierarchy for verbose output."""
    root_logger = logging.getLogger("bootstrap")
    if root_logger.handlers:
        return  # already configured (e.g. re-entered from a test)
    root_logger.setLevel(logging.DEBUG)

    term_handler = logging.StreamHandler()
    term_handler.setLevel(logging.DEBUG)
    term_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    term_handler.setFormatter(term_fmt)
    root_logger.addHandler(term_handler)

    gui_handler = QueueHandler()
    gui_handler.setLevel(logging.INFO)
    gui_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(gui_handler)
