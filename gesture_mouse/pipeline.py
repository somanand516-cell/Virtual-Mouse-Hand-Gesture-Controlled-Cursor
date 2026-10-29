"""The tracking thread: newest camera frame -> hand tracking -> gesture engine -> output thread.

Thread layout (see app.py): camera -> [latest-frame slot] -> TRACKING -> [action queue] -> output.
This thread never touches the OS: it only decides. Actions go to the output thread, which sends
them; slow OS work (brightness, screenshots) goes to the system worker. The preview reads the
newest Snapshot from a latest-only slot, so a slow preview can never slow tracking either.

Key presses from the preview arrive through a queue and are handled here, so the engine is only
ever touched by this thread (no locks needed inside it).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from gesture_mouse import overlay
from gesture_mouse.calibration import Calibration
from gesture_mouse.calibration import summary as calibration_summary
from gesture_mouse.core import config as config_file
from gesture_mouse.core.actions import Action, Brightness, Notify, Screenshot
from gesture_mouse.core.config import Config
from gesture_mouse.core.events import LatestSlot, SlotClosed
from gesture_mouse.core.metrics import Metrics
from gesture_mouse.gestures import GestureEngine, Mode
from gesture_mouse.output.controller import OutputController
from gesture_mouse.output.system import SystemActions
from gesture_mouse.vision.camera import Camera, LightMeter, brighten
from gesture_mouse.vision.pose import palm_center
from gesture_mouse.vision.tracker import HandTracker, TrackedHand

log = logging.getLogger("gesture_mouse")

TOAST_SECONDS = 1.5


@dataclass(frozen=True)
class Snapshot:
    frame: np.ndarray
    primary: TrackedHand | None
    second: TrackedHand | None
    hud: overlay.HUD


class Pipeline:
    def __init__(self, cfg: Config, cam: Camera, tracker: HandTracker, engine: GestureEngine,
                 mouse: OutputController, system: SystemActions, metrics: Metrics | None = None) -> None:
        self.cfg, self.cam, self.tracker, self.engine, self.mouse, self.system = (
            cfg, cam, tracker, engine, mouse, system)
        self.metrics = metrics or Metrics()
        self.paused = False
        self.show_guide = True
        self.calibration: Calibration | None = None
        self._calibration_done_at = 0.0
        self._toast = ("", 0.0)
        self._commands: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._snapshots: LatestSlot[Snapshot] = LatestSlot()
        self._error: BaseException | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._prev_palm: tuple[float, float] | None = None
        self.light = LightMeter(cfg.low_light_level, cfg.low_fps_warning)

    # ---- control (called from the main thread) ----

    def start(self, calibrate: bool = False) -> None:
        if calibrate:
            self.calibration = Calibration(time.perf_counter())
        self._running = True
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            log.info("Performance: %s", self.metrics.summary())

    def command(self, key: int) -> None:
        self._commands.put(key)

    def notify(self, text: str) -> None:
        """Thread-safe: show a short message in the preview."""
        self._toast = (text, time.perf_counter() + TOAST_SECONDS)
        log.info("%s", text)

    def check(self) -> None:
        if self._error is not None:
            raise self._error

    def wait_snapshot(self, timeout: float) -> Snapshot | None:
        """The newest snapshot not yet returned, or None after `timeout` seconds."""
        try:
            return self._snapshots.get(timeout)
        except SlotClosed:
            return None

    # ---- tracking thread ----

    def _run(self) -> None:
        start = time.perf_counter()
        try:
            while self._running:
                frame, captured = self.cam.read_timed()
                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)
                if self.cfg.low_light_mode:
                    frame = brighten(frame)
                now = time.perf_counter()
                self.metrics.tracking_fps.tick(now)
                self.light.update(frame, self.metrics.capture_fps.per_second, now)

                hands = self.tracker.detect_hands(frame, now - start)
                primary, second = self._pick_hands(hands)
                self._handle_commands(now)
                self.mouse.check()
                self.mouse.heartbeat()  # tells the output watchdog we're alive
                self._step(primary, second, now, captured)
                self._snapshots.put(Snapshot(frame, primary, second, self._hud(now)))
        except BaseException as e:  # re-raised on the main thread by check()
            log.exception("Tracking thread failed")
            self._error = e
            self._snapshots.close()

    def _pick_hands(self, hands: list[TrackedHand]) -> tuple[TrackedHand | None, TrackedHand | None]:
        """Choose the controlling hand (by left/right mode), sticking with the same hand between
        frames so the cursor never jumps to the other one. The other hand is used for zoom."""
        want = self.cfg.hand
        candidates = [h for h in hands if want == "any" or h.side.lower() == want]
        if not candidates:
            self._prev_palm = None
            return None, None
        if self._prev_palm is not None and len(candidates) > 1:
            primary = min(candidates, key=lambda h: math.dist(palm_center(h.landmarks), self._prev_palm))
        else:
            primary = candidates[0]
        self._prev_palm = palm_center(primary.landmarks)
        others = [h for h in hands if h is not primary]
        return primary, (others[0] if others else None)

    def _step(self, primary: TrackedHand | None, second: TrackedHand | None, now: float, captured: float) -> None:
        engine = self.engine
        lm = primary.landmarks if primary else None
        if self.calibration is not None:
            engine.update(lm, now)  # keeps the live meters current, but no actions are sent
            self._dispatch(engine.release(), captured)
            self.calibration.update(engine.pose if lm is not None else None, now)
            self._finish_calibration(now)
            return

        if engine.mode is Mode.NO_HAND:
            # Direct mode maps the hand onto the monitor the cursor is on right now.
            engine.direct_area = self.mouse.monitor_rect_normalized()
        if engine.pointer_mode == "mouse" and (engine.mode is Mode.NO_HAND or self.mouse.overriding):
            # Hand away ("mouse lifted") or the user is on the real mouse: continue from the real cursor.
            engine.set_cursor(*self.mouse.position_normalized())
        if self.paused:
            self._dispatch(engine.release(), captured)
        else:
            self._dispatch(engine.update(lm, now, second.landmarks if second else None), captured)

    def _dispatch(self, actions: list[Action], captured: float) -> None:
        decided = time.perf_counter()
        self.metrics.track_ms.add((decided - captured) * 1000)
        to_output: list[Action] = []
        for action in actions:
            if isinstance(action, (Brightness, Screenshot)):
                self.system.apply(action)
            elif isinstance(action, Notify):
                self.notify(action.text)
            else:
                to_output.append(action)
        if to_output:
            self.mouse.submit(to_output, captured, decided)

    def _finish_calibration(self, now: float) -> None:
        cal = self.calibration
        if cal is None or not cal.done:
            return
        if not self._calibration_done_at:
            self._calibration_done_at = now
            assert cal.result is not None
            log.info("Calibration measurements: %s", calibration_summary(cal.samples))
            if cal.result.ok:
                self.engine.set_thresholds(**cal.result.thresholds)
                config_file.update(cal.result.thresholds)
                log.info("Calibrated: %s", cal.result.thresholds)
            else:
                log.warning("Calibration failed: %s", cal.result.message)
        if now - self._calibration_done_at > 2.5:  # leave the result on screen briefly
            self.calibration = None
            self._calibration_done_at = 0.0

    def _handle_commands(self, now: float) -> None:
        engine = self.engine
        while True:
            try:
                key = self._commands.get_nowait()
            except queue.Empty:
                return
            if key in (ord("p"), ord(" ")):
                self.paused = not self.paused
                self.notify("Paused" if self.paused else "Resumed")
            elif key == ord("h"):
                self.show_guide = not self.show_guide
            elif key in (ord("+"), ord("=")):
                self.notify(f"DPI {engine.change_dpi(+1)}")
                config_file.update({"dpi": engine.dpi})
            elif key in (ord("-"), ord("_")):
                self.notify(f"DPI {engine.change_dpi(-1)}")
                config_file.update({"dpi": engine.dpi})
            elif key == ord("m"):
                mode = engine.toggle_pointer_mode()
                self.notify("Mouse mode" if mode == "mouse" else "Direct mode")
                config_file.update({"pointer_mode": mode})
            elif key == ord("l"):
                order = ("any", "right", "left")
                hand = order[(order.index(self.cfg.hand) + 1) % len(order)]
                self.cfg = config_from(self.cfg, hand=hand)
                self.notify({"any": "Either hand", "right": "Right hand", "left": "Left hand"}[hand])
                config_file.update({"hand": hand})
            elif key == ord("c"):
                if self.calibration is None:
                    self.calibration = Calibration(now)
                else:
                    self.calibration = None
                    self._calibration_done_at = 0.0
                    self.notify("Calibration skipped (press C to calibrate later)")

    def _hud(self, now: float) -> overlay.HUD:
        e = self.engine
        cal = self.calibration
        text, until = self._toast
        return overlay.HUD(
            mode=e.mode,
            shape=e.shape.value if e.shape else "",
            fingers_up=e.fingers_up,
            hold_progress=e.hold_progress,
            fps=self.metrics.tracking_fps.per_second,
            latency_ms=self.metrics.total_ms.stats().p50,
            paused=self.paused,
            show_guide=self.show_guide,
            dpi=e.dpi,
            pointer_mode=e.pointer_mode,
            hand=self.cfg.hand,
            direct_gain=e.direct_gain,
            pose=e.pose,
            pinch_on=e.cfg.pinch_on,
            pinch_off=e.cfg.pinch_off,
            calibration_text=cal.instruction if cal else "",
            calibration_progress=cal.progress(now) if cal and not cal.done else 0.0,
            toast=text if time.perf_counter() < until else "",
            warning=self.light.warning,
        )


def config_from(cfg: Config, **changes) -> Config:
    from dataclasses import replace

    return replace(cfg, **changes)
