"""TrainingWatchdog: a stall detector scoped to one `with` block, not the
process — see src/model/watchdog.py for the two real bugs this replaces.
"""

import time

import pytest

from src.model.watchdog import TrainingWatchdog


class _Recorder:
    """Stands in for os._exit: records the call instead of killing pytest."""

    def __init__(self):
        self.calls = []

    def __call__(self, code):
        self.calls.append(code)


TIMEOUT = 0.08
POLL = 0.02


def _watchdog():
    return TrainingWatchdog(timeout_s=TIMEOUT, poll_s=POLL, log=lambda msg: None,
                             _abort=_Recorder())


def test_stalled_work_still_aborts():
    """Detection must stay live: a genuine stall is still caught."""
    wd = _watchdog()
    with wd:
        time.sleep(TIMEOUT + POLL * 4)
    assert wd.aborted
    assert wd._abort.calls == [1]


def test_progressing_work_survives_past_the_timeout():
    """Heartbeats spaced under the timeout must never trigger an abort, even
    though the TOTAL time in the block exceeds `timeout_s` many times over —
    this is the validation-loop bug: predict() never called heartbeat(), so a
    validation pass slower than timeout_s looked identical to a real stall."""
    wd = _watchdog()
    with wd:
        for _ in range(20):
            time.sleep(TIMEOUT / 3)
            wd.heartbeat()
    assert not wd.aborted
    assert wd._abort.calls == []


def test_a_finished_block_cannot_kill_subsequent_work():
    """The core bug: a bare daemon thread with no stop mechanism keeps
    checking a heartbeat that nothing feeds anymore once its block ends, and
    fires during whatever runs next. Exiting the `with` block must stop the
    thread for good."""
    wd = _watchdog()
    with wd:
        pass  # a fold that finished instantly

    # Nothing feeds this watchdog's heartbeat from here on. If it were still
    # running, waiting past the timeout would trigger an abort.
    time.sleep(TIMEOUT + POLL * 4)
    assert not wd.aborted
    assert wd._abort.calls == []
    assert not wd._thread.is_alive()


def test_a_failed_block_also_cannot_kill_subsequent_work():
    """Same guarantee on the exception path — __exit__ must still stop the
    thread when the wrapped work raises, and the exception itself must
    propagate rather than being swallowed."""
    wd = _watchdog()
    with pytest.raises(RuntimeError, match="boom"):
        with wd:
            raise RuntimeError("boom")

    time.sleep(TIMEOUT + POLL * 4)
    assert not wd.aborted
    assert wd._abort.calls == []
    assert not wd._thread.is_alive()


def test_a_second_block_is_unaffected_by_the_first():
    """End-to-end: two sequential `with` blocks, standing in for two folds.
    The first stalls and aborts (in this test, "aborting" just records a
    call); the second, run afterward with its own watchdog instance, must be
    judged entirely on its own heartbeats."""
    first = _watchdog()
    with first:
        time.sleep(TIMEOUT + POLL * 4)
    assert first.aborted

    second = _watchdog()
    with second:
        for _ in range(10):
            time.sleep(TIMEOUT / 3)
            second.heartbeat()
    assert not second.aborted
