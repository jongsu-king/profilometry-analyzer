"""Background jobs on a QThread (structure from xps-autofit-studio).

Clean-up order (the lesson of the xrd-analyzer crash at the end of an analysis): the
worker never deletes itself from inside its thread.  When it reports ``finished`` or
``failed``, the GUI thread first stops the thread (``quit()``), waits for it
(``wait()``), and only then drops its references to the thread and the worker.
Deleting the worker from the thread while the GUI drops its last reference crashed in
Shiboken's destructor.
"""

from __future__ import annotations

import traceback
from collections import deque
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job: Callable[[], Any]) -> None:
        super().__init__()
        self._job = job

    @Slot()
    def run(self) -> None:
        try:
            result = self._job()
        except Exception:  # report everything; the GUI decides what to show
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class JobRunner(QObject):
    """Runs one job at a time; further jobs wait in a queue.  ``synchronous=True`` runs
    jobs inline (tests, smoke runs)."""

    started = Signal(str)
    stopped = Signal()

    def __init__(self, parent: QObject | None = None, synchronous: bool = False) -> None:
        super().__init__(parent)
        self.synchronous = synchronous
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._pending: deque[
            tuple[Callable[[], Any], Callable[[Any], None], Callable[[str], None], str]
        ] = deque()

    @property
    def busy(self) -> bool:
        return self._thread is not None

    @property
    def pending(self) -> int:
        return len(self._pending)

    def queue(
        self,
        job: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        label: str = "job",
    ) -> None:
        if self.busy:
            self._pending.append((job, on_done, on_error, label))
            return
        self.start(job, on_done, on_error, label)

    def cancel_pending(self) -> int:
        n = len(self._pending)
        self._pending.clear()
        return n

    def start(
        self,
        job: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        label: str = "job",
    ) -> bool:
        if self.busy:
            return False
        if self.synchronous:
            self.started.emit(label)
            try:
                result = job()
            except Exception:
                on_error(traceback.format_exc())
            else:
                on_done(result)
            finally:
                self.stopped.emit()
            return True
        self._on_done, self._on_error = on_done, on_error
        self._worker = Worker(job)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self.started.emit(label)
        self._thread.start()
        return True

    def _teardown(self) -> None:
        """quit() -> wait() -> drop the references (in this order)."""
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait()
        self._worker = None
        self._thread = None
        self.stopped.emit()

    def shutdown(self) -> None:
        """Called when the window closes: drop queued jobs and finish the running one."""
        self._pending.clear()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._worker = None
            self._thread = None

    def _start_next(self) -> None:
        if self._pending and not self.busy:
            job, on_done, on_error, label = self._pending.popleft()
            self.start(job, on_done, on_error, label)

    @Slot(object)
    def _finished(self, result: Any) -> None:
        cb = self._on_done
        self._teardown()
        if cb is not None:
            cb(result)
        self._start_next()

    @Slot(str)
    def _failed(self, text: str) -> None:
        cb = self._on_error
        self._teardown()
        if cb is not None:
            cb(text)
        self._start_next()
