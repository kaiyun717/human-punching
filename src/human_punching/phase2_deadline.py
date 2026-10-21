"""Cooperative Phase 2 stop requests; no native imports or asynchronous raises.

This module is intentionally not integrated into any preserved execution recipe.
Install handlers around an entire session, then call ``checkpoint`` only at an
explicitly safe actor/rollout boundary. A handler merely sets two scalar fields.
The caller owns serialization and must retain a reserve before its hard deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import signal
import threading
import time
from typing import Callable, Iterator, Mapping, Any


class BoundedExecutionInterruption(RuntimeError):
    """A recorded cooperative interruption, distinct from a physical failure."""

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__(f"{receipt['reason']} at {receipt['boundary']}")


class ExecutionDeadline:
    """Reserve wall time for partial traces/checkpoints before an external kill.

    ``session_started_monotonic`` must be the runner's original start, before
    constructor/compilation work, so startup consumes the same bounded session.
    A reserve is a stopping allowance, not evidence that serialization fits it.
    The complete observed serialization cost must be measured by the caller.
    """

    def __init__(self, application_wall_s: float, *, serialization_reserve_s: float = 90.,
                 session_started_monotonic: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if (not math.isfinite(application_wall_s) or not math.isfinite(serialization_reserve_s)
                or not 0 < serialization_reserve_s < application_wall_s <= 4 * 3600):
            raise ValueError("A <=four-hour application wall and positive smaller serialization reserve are required")
        self._clock = clock
        now = clock()
        self.started = now if session_started_monotonic is None else float(session_started_monotonic)
        if not math.isfinite(self.started) or self.started > now:
            raise ValueError("Session start must be a finite current or earlier monotonic timestamp")
        self.application_wall_s = float(application_wall_s)
        self.serialization_reserve_s = float(serialization_reserve_s)
        self.hard_deadline = self.started + self.application_wall_s
        self.work_deadline = self.hard_deadline - self.serialization_reserve_s
        self.stop_requested = False
        self.signal_number: int | None = None
        self._handlers_installed = False

    def _signal_handler(self, signum, _frame):
        # No exception, clocks, logging, I/O, locks, allocations of receipts, or
        # native calls in the signal handler. The next safe boundary owns stop.
        self.signal_number = signum
        self.stop_requested = True

    def receipt(self, boundary: str, *, progress: Mapping[str, Any] | None = None):
        """Return a serializable clock observation; never infer completed work."""
        if not isinstance(boundary, str) or not boundary:
            raise ValueError("An explicit safe boundary name is required")
        now = self._clock()
        signal_stop = self.stop_requested
        deadline_stop = now >= self.work_deadline
        return dict(schema="task10b_phase2_cooperative_deadline_v1", boundary=boundary,
            reason="signal_request" if signal_stop else "serialization_reserve" if deadline_stop else None,
            stop_requested=bool(signal_stop or deadline_stop), signal_number=self.signal_number,
            elapsed_wall_s=now-self.started, remaining_application_wall_s=max(0., self.hard_deadline-now),
            application_wall_s=self.application_wall_s, serialization_reserve_s=self.serialization_reserve_s,
            reserve_remaining_in_full=now <= self.work_deadline,
            hard_deadline_already_passed=now >= self.hard_deadline,
            progress={} if progress is None else dict(progress),
            physical_failure=False, external_kill_prevented=False,
            serialization_completion_guaranteed=False)

    def checkpoint(self, boundary: str, *, progress: Mapping[str, Any] | None = None):
        """Raise only synchronously at a caller-declared safe boundary."""
        receipt = self.receipt(boundary, progress=progress)
        if receipt["stop_requested"]:
            raise BoundedExecutionInterruption(receipt)
        return receipt

    @contextmanager
    def signal_handlers(self, signals=(signal.SIGTERM, signal.SIGINT)) -> Iterator["ExecutionDeadline"]:
        """Restore prior handlers even when body work fails; never suppress it."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Signal handlers require the main thread")
        if self._handlers_installed:
            raise RuntimeError("Nested handler installation is not supported")
        if not signals or len(set(signals)) != len(signals):
            raise ValueError("A nonempty set of distinct signals is required")
        old = {}
        self._handlers_installed = True
        try:
            for signum in signals:
                old[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
            yield self
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)
            self._handlers_installed = False
