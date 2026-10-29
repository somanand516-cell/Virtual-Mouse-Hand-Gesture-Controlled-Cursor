"""Performance measurement: rates and latencies per pipeline hop.

Averages hide stutter, so latencies keep a rolling window and report p50 and p95: a p95 of 40 ms
means 1 frame in 20 is slower than 40 ms, which is what you actually feel as a hiccup.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    p50: float
    p95: float
    count: int


class Latency:
    """Rolling window of durations in milliseconds."""

    def __init__(self, window: int = 300) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._values.append(ms)

    def stats(self) -> Stats:
        with self._lock:
            values = sorted(self._values)
        if not values:
            return Stats(0.0, 0.0, 0)
        pick = lambda p: values[min(len(values) - 1, int(len(values) * p))]  # noqa: E731
        return Stats(pick(0.5), pick(0.95), len(values))


class Rate:
    """Events per second, from the average interval (averaging 1/dt would overstate it)."""

    def __init__(self, smoothing: float = 0.05) -> None:
        self._smoothing = smoothing
        self._last: float | None = None
        self._dt = 0.0

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dt = now - self._last
            self._dt = dt if not self._dt else (1 - self._smoothing) * self._dt + self._smoothing * dt
        self._last = now

    @property
    def per_second(self) -> float:
        return 1 / self._dt if self._dt > 0 else 0.0


class Metrics:
    """Everything the overlay and the exit summary report. Written by several threads."""

    def __init__(self) -> None:
        self.capture_fps = Rate()
        self.tracking_fps = Rate()
        self.output_hz = Rate(smoothing=0.01)
        self.track_ms = Latency()      # frame arrived -> hand tracked + gestures decided
        self.queue_ms = Latency()      # action decided -> sent to the OS by the output thread
        self.total_ms = Latency()      # frame arrived -> action sent (what you feel, minus the camera)
        self.dropped_frames = 0        # new frames skipped because tracking was busy (by design, not a bug)
        self.duplicate_frames = 0      # repeated images from the camera driver, ignored

    def summary(self) -> str:
        t, q, total = self.track_ms.stats(), self.queue_ms.stats(), self.total_ms.stats()
        latency = (f"latency p50 {total.p50:.0f} ms, p95 {total.p95:.0f} ms "
                   f"(tracking {t.p50:.0f} ms + output queue {q.p50:.1f} ms)" if total.count
                   else f"latency not measured (no hand seen; tracking alone {t.p50:.0f} ms)")
        return (f"camera {self.capture_fps.per_second:.0f} real fps, tracking {self.tracking_fps.per_second:.0f} fps, "
                f"output {self.output_hz.per_second:.0f} Hz | {latency} | "
                f"skipped: {self.duplicate_frames} repeated + {self.dropped_frames} while busy")
