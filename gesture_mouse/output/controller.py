"""The output thread: the only code that moves the real mouse or presses keys.

Why a separate thread: sending input can block (Alt+Tab must hold its keys briefly, a burst of
volume presses takes time). On the tracking thread that would delay the next frame; here it
can't. The tracking thread just hands over actions and returns immediately.

What it does, 240 times per second:
  1. Real-mouse override: if the physical mouse moved since our last move, the user is using it.
     Gesture movement pauses for `override_time`, then continues from wherever the user left the
     cursor. (Fixes the old bug where the cursor snapped back to a stale position.)
  2. Executes queued discrete actions (clicks, presses, keys) in order. None are ever dropped.
     Clicks/presses land on the position the cursor had when the gesture was decided.
  3. Glides the cursor toward the newest target (camera frames are only 30/s; the glide fills
     the gaps so motion looks smooth).
  4. Watchdog: if the tracking thread stops sending heartbeats while a button or key is held,
     everything is released, so a crash or freeze can never leave the mouse stuck down.

Coordinates: actions use 0..1 over the whole virtual desktop (all monitors). Targets are clamped
onto a real monitor, so the cursor never lands in the gaps between screens.

There is deliberately no "move the mouse into a corner" emergency stop any more: Windows puts
Start and Show Desktop in corners, so ordinary mouse use kept stopping the app. Use the
kill-switch hotkey (Ctrl+Alt+Q) instead.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from gesture_mouse.core.actions import Action, Click, HScroll, Key, Move, Press, Release, Scroll, Volume, Zoom
from gesture_mouse.core.metrics import Metrics
from gesture_mouse.output.monitors import Rect, bounding, clamp_to_monitors, monitor_at

log = logging.getLogger(__name__)

KEY_HOLD = 0.03          # seconds keys of a shortcut stay down (Alt+Tab needs a moment)
OVERRIDE_THRESHOLD = 2   # px the real cursor may differ from where we put it before it counts as the user


@dataclass(frozen=True)
class _Job:
    action: Action
    target: tuple[float, float] | None  # cursor target (px) when the action was decided
    captured_at: float                  # when the camera frame that caused it arrived
    decided_at: float                   # when the gesture engine produced it


class OutputController:
    """dry_run=True: nothing is sent to the OS, but timing and logic run as normal."""

    def __init__(self, metrics: Metrics | None = None, dry_run: bool = False, glide_time: float = 0.025,
                 override_time: float = 1.0, watchdog_timeout: float = 1.0, rate_hz: float = 240.0,
                 backend=None) -> None:
        if backend is None:
            from gesture_mouse.output import input_backend

            backend = input_backend.create()
        self._io = DryRunIO(backend) if dry_run else backend
        self.dry_run = dry_run
        self.metrics = metrics or Metrics()
        self.glide_time = glide_time
        self.override_time = override_time
        self.watchdog_timeout = watchdog_timeout
        self.period = 1.0 / rate_hz

        self.monitors: list[Rect] = self._io.monitors()
        self.desktop = bounding(self.monitors)

        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None
        self._target_captured_at = 0.0
        self._target_new = False
        self._pos: tuple[float, float] | None = None
        self._expected: tuple[int, int] | None = None  # where the cursor should be if only we moved it
        self._override_until = 0.0
        self._jobs: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._scheduled: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()
        self._held_button = False
        self._held_keys: set[str] = set()
        self._heartbeat = time.perf_counter()
        self._error: BaseException | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="output", daemon=True)
        self._thread.start()

    # ---- called from other threads ----

    @property
    def screen_w(self) -> int:  # kept for the overlay / older callers
        return self.desktop.width

    @property
    def screen_h(self) -> int:
        return self.desktop.height

    @property
    def overriding(self) -> bool:
        """True while the user is using the real mouse."""
        return time.perf_counter() < self._override_until

    def submit(self, actions: list[Action], captured_at: float, decided_at: float | None = None) -> None:
        """Hand over the actions from one frame. Never blocks."""
        decided_at = time.perf_counter() if decided_at is None else decided_at
        for action in actions:
            if isinstance(action, Move):
                with self._lock:
                    self._target = self._to_pixels(action.x, action.y)
                    self._target_captured_at = captured_at
                    self._target_new = True
            else:
                with self._lock:
                    target = self._target
                self._jobs.put(_Job(action, target, captured_at, decided_at))

    def apply(self, action: Action) -> None:
        """Single action, for callers without frame timing (tests, tools)."""
        now = time.perf_counter()
        self.submit([action], now, now)

    def heartbeat(self) -> None:
        """The tracking thread calls this every frame; silence means it froze or crashed."""
        self._heartbeat = time.perf_counter()

    def position_normalized(self) -> tuple[float, float]:
        x, y = self._io.position()
        d = self.desktop
        return (x - d.left) / max(d.width - 1, 1), (y - d.top) / max(d.height - 1, 1)

    def monitor_rect_normalized(self) -> tuple[float, float, float, float]:
        """The monitor under the cursor as (left, top, right, bottom) in 0..1 desktop units."""
        x, y = self._io.position()
        m, d = monitor_at(self.monitors, x, y), self.desktop
        nx = lambda v: (v - d.left) / max(d.width - 1, 1)  # noqa: E731
        ny = lambda v: (v - d.top) / max(d.height - 1, 1)  # noqa: E731
        return nx(m.left), ny(m.top), nx(m.right), ny(m.bottom)

    def check(self) -> None:
        """Re-raise any error from the output thread."""
        if self._error is not None:
            raise self._error

    def flush(self, timeout: float = 1.0) -> None:
        """Wait until all queued actions have been sent (tests, shutdown)."""
        end = time.perf_counter() + timeout
        while not self._jobs.empty() and time.perf_counter() < end:
            time.sleep(0.002)
        time.sleep(self.period * 2)

    def close(self) -> None:
        self.flush(0.5)
        self._running = False
        self._thread.join(timeout=1.0)
        self.release_all()

    def release_all(self) -> None:
        """Let go of every button and key we hold. Safe to call from any thread, any time."""
        try:
            if self._held_button:
                self._io.button("left", False)
                self._held_button = False
            for combo in list(self._held_keys):
                self._io.keys(combo, False)
            self._held_keys.clear()
        except Exception as e:  # shutdown path: never raise
            log.warning("Releasing input failed: %s", e)

    # ---- output thread ----

    def _to_pixels(self, x: float, y: float) -> tuple[float, float]:
        d = self.desktop
        return clamp_to_monitors(self.monitors, d.left + x * (d.width - 1), d.top + y * (d.height - 1))

    def _run(self) -> None:
        last = time.perf_counter()
        try:
            while self._running:
                now = time.perf_counter()
                dt, last = now - last, now
                self.metrics.output_hz.tick(now)
                self._detect_override(now)
                self._run_scheduled(now)
                self._run_jobs(now)
                self._glide(dt, now)
                self._watchdog(now)
                time.sleep(max(0.0, self.period - (time.perf_counter() - now)))
        except BaseException as e:
            log.exception("Output thread failed")
            self._error = e
            self.release_all()

    def _detect_override(self, now: float) -> None:
        # Any cursor movement we didn't cause is the user on the real mouse.
        real = self._io.position()
        expected = self._expected
        if expected is not None and max(abs(real[0] - expected[0]), abs(real[1] - expected[1])) > OVERRIDE_THRESHOLD:
            if not self.overriding:
                log.debug("Real mouse moved: gesture movement paused")
            self._override_until = now + self.override_time  # keeps extending while they keep moving
        self._expected = real
        if self.overriding:
            with self._lock:
                self._pos = (float(real[0]), float(real[1]))  # resume from where the user left it

    def _run_scheduled(self, now: float) -> None:
        while self._scheduled and self._scheduled[0][0] <= now:
            _, _, fn = heapq.heappop(self._scheduled)
            fn()

    def _schedule(self, at: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._scheduled, (at, next(self._seq), fn))

    def _run_jobs(self, now: float) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            self._execute(job, now)
            self.metrics.queue_ms.add((time.perf_counter() - job.decided_at) * 1000)
            self.metrics.total_ms.add((time.perf_counter() - job.captured_at) * 1000)

    def _execute(self, job: _Job, now: float) -> None:
        io = self._io
        match job.action:
            case Click(button):
                self._settle(job.target)
                io.click(button)
            case Press():
                self._settle(job.target)
                io.button("left", True)
                self._held_button = True
            case Release():
                io.button("left", False)
                self._held_button = False
            case Scroll(amount):
                io.scroll(amount)
            case HScroll(amount):
                io.hscroll(amount)
            case Zoom(steps):
                io.hold_key("ctrl", True)
                try:
                    io.scroll(120 * steps)
                finally:
                    io.hold_key("ctrl", False)
            case Key(combo):
                io.keys(combo, True)
                self._held_keys.add(combo)
                self._schedule(now + KEY_HOLD, lambda c=combo: self._release_keys(c))
            case Volume(steps):
                name = "volumeup" if steps > 0 else "volumedown"
                for _ in range(abs(steps)):
                    io.keys(name, True)
                    io.keys(name, False)
            case _:
                log.warning("Output ignored unknown action %r", job.action)

    def _release_keys(self, combo: str) -> None:
        if combo in self._held_keys:
            self._io.keys(combo, False)
            self._held_keys.discard(combo)

    def _settle(self, target: tuple[float, float] | None) -> None:
        """Put the cursor exactly on the target before clicking, unless the user is on the real mouse."""
        if target is None or self.overriding:
            return
        self._move_to(target)
        with self._lock:
            self._pos = target

    def _move_to(self, p: tuple[float, float]) -> None:
        x, y = round(p[0]), round(p[1])
        self._io.move(x, y)
        self._expected = self._io.position()  # read back: Windows may adjust it slightly

    def _glide(self, dt: float, now: float) -> None:
        if self.overriding:
            return
        with self._lock:
            target, pos = self._target, self._pos
            if target is None:
                return
            if self._target_new:
                self._target_new = False
                self.metrics.total_ms.add((now - self._target_captured_at) * 1000)
            if pos is None:
                real = self._io.position()
                pos = (float(real[0]), float(real[1]))
            k = 1.0 - math.exp(-dt / self.glide_time)  # exponential approach: fast, never overshoots
            new = (pos[0] + (target[0] - pos[0]) * k, pos[1] + (target[1] - pos[1]) * k)
            self._pos = new
        if (round(new[0]), round(new[1])) != (round(pos[0]), round(pos[1])):
            self._move_to(new)

    def _watchdog(self, now: float) -> None:
        holding = self._held_button or self._held_keys
        if holding and now - self._heartbeat > self.watchdog_timeout:
            log.warning("Tracking stopped responding for %.1f s while input was held: releasing.",
                        now - self._heartbeat)
            self.release_all()


class DryRunIO:
    """Pretends to send input: remembers the cursor position it 'set', sends nothing."""

    def __init__(self, real) -> None:
        self._real = real
        self._pos = real.position()

    def monitors(self) -> list[Rect]:
        return self._real.monitors()

    def position(self) -> tuple[int, int]:
        return self._pos

    def move(self, x: int, y: int) -> None:
        self._pos = (x, y)

    def __getattr__(self, name):  # button, click, scroll, hscroll, keys, hold_key: do nothing
        return lambda *a, **k: None


# Backwards-compatible name used by older modules and tests.
MouseController = OutputController
