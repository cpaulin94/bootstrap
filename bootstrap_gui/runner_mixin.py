"""bootstrap_gui.runner_mixin — one place for the "thread + queue + poll" pattern.

The original code duplicated this pattern three times (Space Explorer,
Compare sub-mode, Sweep sub-mode) with the same guard (`_destroyed` /
`winfo_exists()`) copy-pasted each time, and the same bug: the Run control
was not guaranteed to be re-enabled on every error path. This mixin makes
that guarantee structural instead of something every call site has to
remember.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional

log = logging.getLogger("bootstrap.gui.job")

_FINISHED = "__bg_job_finished__"
_ERROR = "__bg_job_error__"


class BackgroundJobMixin:
    """Mix into any ``tk.Widget`` subclass that runs work on a worker thread.

    The wrapped ``target`` is called as::

        target(*args, result_queue=queue.Queue(), stop_event=threading.Event(), **kwargs)

    It should put arbitrary progress messages onto ``result_queue`` (they
    are forwarded verbatim to ``on_message``) and periodically check
    ``stop_event.is_set()`` to honour cancellation.

    On every exit path — normal return, a raised exception, ``cancel_job()``,
    or the widget being destroyed mid-run — ``is_running`` becomes ``False``
    and exactly one of ``on_done`` / ``on_error`` fires (unless the widget
    was destroyed, in which case neither fires and polling simply stops).
    """

    _bg_queue: "queue.Queue"
    _bg_stop_event: threading.Event
    _bg_running: bool
    _bg_destroyed: bool
    _bg_thread: Optional[threading.Thread]

    def _bg_init(self) -> None:
        self._bg_queue = queue.Queue()
        self._bg_stop_event = threading.Event()
        self._bg_running = False
        self._bg_destroyed = False
        self._bg_thread = None

    @property
    def is_running(self) -> bool:
        return getattr(self, "_bg_running", False)

    def mark_bg_destroyed(self) -> None:
        """Call from the widget's ``destroy()`` override, before ``super().destroy()``."""
        self._bg_destroyed = True
        if hasattr(self, "_bg_stop_event"):
            self._bg_stop_event.set()

    def start_job(
        self,
        target: Callable[..., None],
        *,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        on_message: Optional[Callable[[Any], None]] = None,
        on_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        poll_ms: int = 120,
    ) -> None:
        if not hasattr(self, "_bg_queue"):
            self._bg_init()
        if self._bg_running:
            log.warning("[JOB] start_job called while a job is already running — ignored")
            return

        self._bg_queue = queue.Queue()
        self._bg_stop_event = threading.Event()
        self._bg_running = True
        self._bg_destroyed = False

        result_queue = self._bg_queue
        stop_event = self._bg_stop_event

        def _runner() -> None:
            try:
                target(*args, **(kwargs or {}), result_queue=result_queue, stop_event=stop_event)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never swallowed
                log.error("[JOB] worker raised: %s", exc, exc_info=True)
                result_queue.put((_ERROR, exc))
            finally:
                result_queue.put((_FINISHED,))

        self._bg_thread = threading.Thread(target=_runner, daemon=True)
        self._bg_thread.start()
        self._bg_poll(on_message=on_message, on_done=on_done, on_error=on_error, poll_ms=poll_ms)

    def cancel_job(self) -> None:
        if hasattr(self, "_bg_stop_event"):
            self._bg_stop_event.set()
        self._bg_running = False

    def _bg_poll(self, *, on_message, on_done, on_error, poll_ms) -> None:
        if getattr(self, "_bg_destroyed", False):
            return
        try:
            if not self.winfo_exists():  # type: ignore[attr-defined]
                return
        except Exception:
            return

        try:
            while True:
                msg = self._bg_queue.get_nowait()
                if msg[0] == _FINISHED:
                    self._bg_running = False
                    if on_done:
                        on_done()
                    return
                if msg[0] == _ERROR:
                    self._bg_running = False
                    if on_error:
                        on_error(msg[1])
                    return
                if on_message:
                    on_message(msg)
        except queue.Empty:
            pass

        if self._bg_running:
            self.after(  # type: ignore[attr-defined]
                poll_ms,
                lambda: self._bg_poll(
                    on_message=on_message, on_done=on_done, on_error=on_error, poll_ms=poll_ms
                ),
            )
