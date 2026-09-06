"""A stall watchdog scoped to exactly one unit of work, not the process.

Two Kaggle training runs (`train-6slice-base-24ep`, `knee-train-pseudo-sel`)
each burned a full session cap (9h TPU, 12h GPU) on a silent stall with
nothing to show for it, because Kaggle's log shipping does not survive the
hard SIGKILL at the timeout — a genuine stall and a slow-but-working run were
indistinguishable after the fact. The fix was a background thread that aborts
the process if no heartbeat arrives within `timeout_s`.

The first version of that fix had two real bugs, both from being a bare
`threading.Thread(daemon=True)` started once per fold with no way to stop it:

1. It never stopped. Once a fold's own training loop finished, nothing fed
   its heartbeat again, so ~`timeout_s` later — potentially mid-way through
   the NEXT fold's legitimate work — it would fire and kill the whole
   process. A finished or failed fold should not be able to kill anything
   that comes after it.
2. Validation (`predict()`) never fed a heartbeat at all, so any validation
   pass slower than `timeout_s` (a large gold-study holdout, a slow disk) was
   indistinguishable from a stall and could false-abort a run that was
   working correctly the whole time.

`TrainingWatchdog` is a context manager instead: entering it starts the
thread, and exiting it (via a normal return OR an exception) stops the
watchdog thread and joins it, so its lifetime is exactly one `with` block —
typically one fold — and it cannot outlive that block to threaten whatever
runs next. Call `.heartbeat()` from every place that represents real
progress: after each training step, after each validation batch, and
immediately before a slow-but-legitimate operation like a checkpoint write so
that operation's own duration is not charged against the stall timer.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable


class TrainingWatchdog:
    """Aborts the process if `.heartbeat()` goes unfed for `timeout_s`.

    Usage::

        with TrainingWatchdog(timeout_s=120, log=log) as watchdog:
            for step in steps:
                ... do work ...
                watchdog.heartbeat()
            score = predict(model, rows, heartbeat=watchdog.heartbeat)
            watchdog.heartbeat()   # exclude the save itself from the timer
            torch.save(...)
        # thread is stopped and joined here, whatever happened inside

    `poll_s` controls both how quickly a stall is noticed (up to `poll_s`
    after `timeout_s` has actually elapsed) and how quickly `__exit__` can
    return (it waits for the poll loop to notice the stop request, bounded by
    `poll_s`). The default trades a few seconds of extra abort latency for a
    watchdog that shuts down promptly when a fold finishes normally.
    """

    def __init__(
        self,
        timeout_s: float = 120.0,
        poll_s: float = 5.0,
        log: Callable[[str], None] = print,
        _abort: Callable[[int], None] = os._exit,
    ) -> None:
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._log = log
        self._abort = _abort
        self._last = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.aborted = False

    def heartbeat(self) -> None:
        self._last = time.time()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            stalled_for = time.time() - self._last
            if stalled_for > self.timeout_s:
                self.aborted = True
                self._log(
                    f"!! WATCHDOG: no progress in {stalled_for:.0f}s "
                    f"(limit {self.timeout_s:.0f}s) - stall detected, aborting "
                    f"rather than silently burning the rest of the session."
                )
                self._abort(1)
                return  # unreachable when _abort is the real os._exit; kept for testability

    def __enter__(self) -> "TrainingWatchdog":
        self.heartbeat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_s * 2)
        return False  # never suppress an exception from the wrapped work
