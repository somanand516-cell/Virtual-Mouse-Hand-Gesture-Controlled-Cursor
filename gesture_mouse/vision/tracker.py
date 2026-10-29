"""MediaPipe hand landmark detection."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Silence MediaPipe/TFLite C++ log spam; must be set before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from gesture_mouse.vision.pose import Landmarks  # noqa: E402

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@contextlib.contextmanager
def _quiet_native_stderr() -> Iterator[None]:
    """Hide MediaPipe's C++ start-up warnings, which ignore the env vars above."""
    sys.stderr.flush()
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def ensure_model(path: Path) -> Path:
    """Download the hand model on first run."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading hand model to %s ...", path)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the hand model ({e}). Check your internet connection.") from e
    return path


_KEEP_ALIVE: list = []


@dataclass(frozen=True)
class TrackedHand:
    landmarks: Landmarks
    side: str     # "Left" or "Right" (the user's own hand; correct because we mirror the image)
    score: float


class HandTracker:
    def __init__(self, model_path: Path, detection_confidence: float, tracking_confidence: float,
                 num_hands: int = 2) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(Path(model_path)))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        with _quiet_native_stderr():
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self._warmup_frames = 3  # MediaPipe also warns on the first detections

    def detect(self, frame_bgr: np.ndarray, t: float) -> Landmarks | None:
        """Return the 21 landmarks of the most prominent hand, or None."""
        hands = self.detect_hands(frame_bgr, t)
        return hands[0].landmarks if hands else None

    def detect_hands(self, frame_bgr: np.ndarray, t: float) -> list[TrackedHand]:
        """All visible hands (up to num_hands), most confident first."""
        ts = max(int(t * 1000), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self._warmup_frames:
            self._warmup_frames -= 1
            with _quiet_native_stderr():
                result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect_for_video(image, ts)
        hands = []
        for lm, handedness in zip(result.hand_landmarks, result.handedness):
            best = handedness[0] if handedness else None
            hands.append(TrackedHand(lm, best.category_name if best else "Unknown", best.score if best else 0.0))
        return sorted(hands, key=lambda h: -h.score)

    def close(self) -> None:
        """Stop using the tracker WITHOUT MediaPipe's own shutdown.

        MediaPipe's close() (also called by its __del__) waits for a usage-statistics upload to
        Google ("Clearcut"). When that network call is blocked or slow it hangs for ~42 s, which
        made quitting the app take that long. The tracker is only closed when the app exits, and
        the OS frees everything then, so we detach the native handle instead of closing it.
        """
        lm = getattr(self, "_landmarker", None)
        if lm is None:
            return
        if hasattr(lm, "_handle"):
            lm._handle = None  # makes MediaPipe's close()/__del__ a no-op
        else:  # unknown MediaPipe version: keep it alive so __del__ never runs before exit
            _KEEP_ALIVE.append(lm)
        self._landmarker = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
