"""Webcam access: picks the fastest backend and captures on a background thread.

Capturing on its own thread means hand detection never waits for the camera, and the app
always works on the newest frame (no stale, laggy backlog).

Duplicate frames: in dim light many webcams slow their sensor down (longer exposure) but some
drivers, notably Windows Media Foundation, keep *reporting* 30 fps by repeating each image. On the
development laptop 2 of every 3 frames were exact copies (10 real fps). Copies carry no new hand
position, so they are dropped here and every FPS figure counts only real, new images.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import cv2
import numpy as np

from gesture_mouse.core.events import LatestSlot, SlotClosed

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 30
GOOD_ENOUGH_FPS = 24      # stop probing once a backend reaches this
PROBE_SECONDS = 0.8
LOW_FPS_WARNING = 20      # below this many real fps, tell the user why and what helps


def brighten(frame: np.ndarray, target: float = 110.0, max_gain: float = 6.0) -> np.ndarray:
    """Low-light mode: a short fixed exposure keeps 30 fps but gives a dark image; lift it in
    software so the hand tracker can see. Costs ~1-2 ms. Noise is amplified too, so this is a
    trade-off, not free: more real light is always better."""
    mean = float(frame[::8, ::8].mean())
    if mean >= target * 0.8:
        return frame
    return cv2.convertScaleAbs(frame, alpha=min(target / max(mean, 1.0), max_gain))


class LightMeter:
    """Low-light warning for the preview. Dim light shows up in one of two ways: the image is dark,
    or (more often) the camera keeps the image bright by exposing longer, so it delivers fewer real
    frames per second. Both are watched. Each check has a margin before the warning clears, so it
    doesn't blink on and off at the border."""

    SETTLE_SECONDS = 2.0   # the camera adjusts its exposure at first; don't judge before that
    SMOOTHING = 1.0        # seconds; brightness is averaged so a hand passing close by doesn't count

    def __init__(self, dark_level: float, low_fps: float) -> None:
        self.dark_level = dark_level
        self.low_fps = low_fps
        self.brightness: float | None = None
        self.dark = self.slow = False
        self._start: float | None = None
        self._last_t = 0.0

    @property
    def warning(self) -> str:
        if self.dark:
            return "Too dark: add light on your hand"
        if self.slow:
            return "Camera slow in this light: more light = smoother"
        return ""

    def update(self, frame: np.ndarray, fps: float, t: float) -> str:
        mean = float(frame[::8, ::8].mean())  # every 8th pixel is plenty; ~0.05 ms
        if self._start is None:
            self._start, self.brightness = t, mean
        else:
            assert self.brightness is not None
            self.brightness += min((t - self._last_t) / self.SMOOTHING, 1.0) * (mean - self.brightness)
        self._last_t = t
        if t - self._start < self.SETTLE_SECONDS:
            return ""
        before = self.warning
        if self.brightness < self.dark_level:
            self.dark = True
        elif self.brightness > self.dark_level + 10:
            self.dark = False
        if 0 < fps < self.low_fps:
            self.slow = True
        elif fps > self.low_fps + 3:
            self.slow = False
        if self.warning != before:
            if self.warning:
                log.warning("%s (brightness %.0f/255, camera %.0f fps)", self.warning, self.brightness, fps)
            else:
                log.info("Light OK again (brightness %.0f/255, camera %.0f fps)", self.brightness, fps)
        return self.warning


def _signature(frame: np.ndarray) -> np.ndarray:
    """A cheap fingerprint (every 16th pixel) to spot repeated frames. ~0.05 ms."""
    return frame[::16, ::16]


class CameraError(RuntimeError):
    pass


def _backends() -> list[tuple[str, int]]:
    if sys.platform == "win32":
        # Media Foundation is usually faster than DirectShow on modern laptops, but not always,
        # so both are measured.
        return [("Media Foundation", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW), ("auto", cv2.CAP_ANY)]
    return [("auto", cv2.CAP_ANY)]


def _measure_fps(cap: cv2.VideoCapture) -> float:
    """Real frames per second: repeated images don't count."""
    for _ in range(3):  # first frames are often slow while the camera warms up
        cap.read()
    start, frames, prev = time.perf_counter(), 0, None
    while time.perf_counter() - start < PROBE_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        sig = _signature(frame)
        if prev is None or not np.array_equal(sig, prev):
            frames += 1
        prev = sig
    return frames / PROBE_SECONDS


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 exposure: float | None = None, metrics=None) -> None:
        self.index = index
        self.exposure = exposure  # None = automatic; a fixed value keeps 30 fps in dim light
        self.metrics = metrics
        self.width = width
        self.height = height
        self.fps = 0.0
        self.backend = ""
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._slot: LatestSlot[tuple[np.ndarray, float]] = LatestSlot()
        self._error: CameraError | None = None

    # ---- opening ----

    def _open_backend(self, backend: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.exposure is not None:
            # Auto exposure lengthens the shutter in dim light, which lowers the frame rate.
            # A fixed (short) exposure keeps 30 fps; the image gets darker, but tracking copes.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on most Windows drivers
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        return cap

    def open(self) -> None:
        # Each backend is measured and CLOSED before the next is tried: opening the same webcam
        # through two drivers at once breaks the first stream.
        results: list[tuple[float, str, int]] = []
        for name, backend in _backends():
            cap = self._open_backend(backend)
            if cap is None:
                continue
            fps = _measure_fps(cap)
            cap.release()
            log.debug("Camera %d via %s: %.1f real fps", self.index, name, fps)
            if fps > 0:
                results.append((fps, name, backend))
            if fps >= GOOD_ENOUGH_FPS:
                break

        self._cap = None
        for fps, name, backend in sorted(results, reverse=True):  # best first; fall back if reopening fails
            cap = self._open_backend(backend)
            if cap is not None and cap.read()[0]:
                self._cap, self.fps, self.backend = cap, fps, name
                break
            if cap is not None:
                cap.release()
        if self._cap is None:
            raise CameraError(
                f"Could not read from camera {self.index}. Close other apps using the webcam "
                "(Zoom, Teams, the Camera app), check Windows Settings > Privacy > Camera, "
                "or try another index with --camera 1."
            )
        log.info("Camera %d: %s, %.0f real fps", self.index, self.backend, self.fps)
        if self.fps < LOW_FPS_WARNING:
            log.warning("The camera gives only %.0f real frames per second in this light (it slows its sensor "
                        "down when it is dark). More light on your hand gives up to 30 fps and a smoother cursor. "
                        "Or try \"low_light_mode\": true in config.json.", self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="camera", daemon=True)
        self._thread.start()

    # ---- capture thread ----

    def _capture_loop(self) -> None:
        assert self._cap is not None
        failures = 0
        prev_sig: np.ndarray | None = None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    self._error = CameraError(
                        "Camera stopped sending frames (was it unplugged or taken by another app?)")
                    self._slot.close()
                    return
                continue
            failures = 0
            sig = _signature(frame)
            if prev_sig is not None and np.array_equal(sig, prev_sig):
                if self.metrics is not None:
                    self.metrics.duplicate_frames += 1
                continue  # the driver repeated the last image: nothing new to track
            prev_sig = sig
            now = time.perf_counter()
            if self.metrics is not None:
                self.metrics.capture_fps.tick(now)
            self._slot.put((frame, now))  # overwrites an unread frame: tracking always gets the newest
            if self.metrics is not None:
                self.metrics.dropped_frames = self._slot.dropped

    # ---- public ----

    def read(self, timeout: float = 3.0) -> np.ndarray:
        """Return the newest frame, waiting for one we haven't returned yet."""
        return self.read_timed(timeout)[0]

    def read_timed(self, timeout: float = 3.0) -> tuple[np.ndarray, float]:
        """Like read(), plus the time.perf_counter() moment the frame arrived (for latency)."""
        try:
            item = self._slot.get(timeout)
        except SlotClosed:
            raise self._error or CameraError("Camera closed") from None
        if item is None:
            raise CameraError("Camera stopped responding")
        return item

    def close(self) -> None:
        self._running = False
        self._slot.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
