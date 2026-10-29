"""Draws the preview: hands, current gesture and mode, FPS/latency, live meters, guide, calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gesture_mouse.gestures import Mode
from gesture_mouse.vision.pose import FINGER_NAMES, INDEX_TIP, THUMB_TIP, HandPose, Landmarks, palm_center

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
RED = (60, 60, 255)
ORANGE = (0, 165, 255)

# BGR
MODE_COLORS = {
    Mode.NO_HAND: (160, 160, 160),
    Mode.IDLE: (200, 200, 200),
    Mode.MOVE: (80, 220, 80),
    Mode.PRECISION: (180, 255, 120),
    Mode.CLICK: (60, 60, 255),
    Mode.DRAG: (255, 140, 0),
    Mode.RIGHT: (255, 120, 255),
    Mode.SCROLL: (0, 200, 255),
    Mode.VOLUME: (255, 200, 0),
    Mode.BRIGHTNESS: (0, 255, 255),
    Mode.ZOOM: (255, 255, 0),
    Mode.FREEZE: (200, 200, 200),
    Mode.PALM: (230, 230, 230),
    Mode.HOLD: (255, 80, 255),
}

GUIDE = [
    (Mode.MOVE, "Point (index)", "move"),
    (Mode.PRECISION, "Point, thumb tucked", "precise move"),
    (Mode.CLICK, "Pinch thumb+index", "click / x2 double"),
    (Mode.DRAG, "Pinch + hold", "drag"),
    (Mode.RIGHT, "Thumb + middle", "right click"),
    (Mode.SCROLL, "Two fingers", "scroll"),
    (Mode.VOLUME, "Three fingers", "volume"),
    (Mode.BRIGHTNESS, "Rock (index+pinky)", "brightness"),
    (Mode.FREEZE, "Fist", "freeze"),
    (Mode.PALM, "Palm swipe", "Alt+Tab"),
    (Mode.HOLD, "Shaka (pinky) hold", "screenshot"),
    (Mode.ZOOM, "Pinch both hands", "zoom"),
]


@dataclass(frozen=True)
class HUD:
    mode: Mode
    shape: str
    fingers_up: tuple[bool, ...]
    hold_progress: float
    fps: float
    latency_ms: float
    paused: bool
    show_guide: bool
    dpi: int
    pointer_mode: str
    hand: str
    direct_gain: float
    pose: HandPose | None
    pinch_on: float
    pinch_off: float
    calibration_text: str = ""
    calibration_progress: float = 0.0
    toast: str = ""
    warning: str = ""  # e.g. low light


def _darken(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, factor: float = 0.35) -> None:
    h, w = img.shape[:2]
    region = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    region[:] = (region * factor).astype(np.uint8)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int],
          thickness: int = 1) -> None:
    if scale < 0.6:  # outlines smudge small text; darken a strip behind it instead
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = org
        _darken(img, x - 4, y - th - 4, x + tw + 4, y + base + 2)
    else:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _right(img: np.ndarray, text: str, y: int, scale: float, color=WHITE, margin: int = 10) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    _text(img, text, (img.shape[1] - tw - margin, y), scale, color)


def _centered(img: np.ndarray, text: str, y: int, scale: float, color: tuple[int, int, int], thickness: int) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _text(img, text, ((img.shape[1] - tw) // 2, y), scale, color, thickness)


def _hand(img: np.ndarray, lm: Landmarks, color: tuple[int, int, int], s: float, faint: bool = False) -> None:
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    line = (120, 120, 120) if faint else (230, 230, 230)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], line, max(1, round(2 * s)), cv2.LINE_AA)
    if faint:
        return
    for i in (THUMB_TIP, INDEX_TIP):
        cv2.circle(img, pts[i], round(6 * s), color, -1, cv2.LINE_AA)


def _status_panel(img: np.ndarray, hud: HUD, s: float) -> None:
    """Top-right: pinch meter and which fingers are up."""
    w = img.shape[1]
    x, y, bw = w - round(150 * s), round(58 * s), round(138 * s)
    _darken(img, x - 6, y - round(14 * s), x + bw + 6, y + round(40 * s))
    pose = hud.pose
    # Pinch meter: fill = how close thumb and index are; white marks = click thresholds.
    fill = lambda r: 1 - min(r, 1.0)  # noqa: E731
    active = hud.mode in (Mode.CLICK, Mode.DRAG)
    cv2.putText(img, "Pinch", (x, y - round(3 * s)), FONT, 0.38 * s, MODE_COLORS[Mode.CLICK] if active else WHITE,
                1, cv2.LINE_AA)
    bar_y, bar_h = y, round(7 * s)
    cv2.rectangle(img, (x, bar_y), (x + bw, bar_y + bar_h), GREY, 1)
    cv2.rectangle(img, (x, bar_y), (x + int(bw * (fill(pose.pinch_index) if pose else 0)), bar_y + bar_h),
                  MODE_COLORS[Mode.CLICK] if active else GREY, -1)
    for m in (fill(hud.pinch_off), fill(hud.pinch_on)):
        cv2.line(img, (x + int(bw * m), bar_y - 2), (x + int(bw * m), bar_y + bar_h + 2), WHITE, 1)
    # Finger states: lit = finger up.
    fy = y + round(28 * s)
    cell = bw // 4
    for i, name in enumerate(FINGER_NAMES):
        up = pose is not None and i < len(hud.fingers_up) and hud.fingers_up[i]
        cx = x + i * cell
        cv2.rectangle(img, (cx + 2, fy - round(11 * s)), (cx + cell - 2, fy + round(4 * s)),
                      (80, 220, 80) if up else (70, 70, 70), -1)
        cv2.putText(img, name[0].upper(), (cx + cell // 2 - round(4 * s), fy), FONT, 0.4 * s, WHITE, 1, cv2.LINE_AA)


def _guide(img: np.ndarray, hud: HUD, s: float) -> None:
    h = img.shape[0]
    line_h = round(16 * s)
    top = h - round(10 * s) - line_h * len(GUIDE)
    _darken(img, 4, top - round(10 * s), round(318 * s), h - 4, 0.4)
    for i, (mode, gesture, action) in enumerate(GUIDE):
        lit = mode is hud.mode and mode not in (Mode.NO_HAND, Mode.IDLE)
        color = MODE_COLORS[mode] if lit else (235, 235, 235)
        y = top + i * line_h
        cv2.putText(img, gesture, (round(10 * s), y), FONT, 0.42 * s, color, 2 if lit else 1, cv2.LINE_AA)
        cv2.putText(img, action, (round(170 * s), y), FONT, 0.42 * s, color, 2 if lit else 1, cv2.LINE_AA)


def _calibration(img: np.ndarray, hud: HUD, s: float) -> None:
    h, w = img.shape[:2]
    y0, y1 = h // 2 - round(50 * s), h // 2 + round(40 * s)
    _darken(img, 0, y0, w, y1, 0.25)
    _centered(img, "CALIBRATION", y0 + round(26 * s), 0.55 * s, (0, 200, 255), 1)
    _centered(img, hud.calibration_text, y0 + round(56 * s), 0.55 * s, WHITE, 1)
    if hud.calibration_progress > 0:
        bx0, bx1, by = round(w * 0.2), round(w * 0.8), y1 - round(14 * s)
        cv2.rectangle(img, (bx0, by), (bx1, by + round(6 * s)), GREY, 1)
        cv2.rectangle(img, (bx0, by), (bx0 + int((bx1 - bx0) * hud.calibration_progress), by + round(6 * s)),
                      (0, 200, 255), -1)
    _right(img, "C skip calibration  Q quit", h - round(8 * s), 0.36 * s)


def draw(frame: np.ndarray, lm: Landmarks | None, hud: HUD, second: Landmarks | None = None) -> None:
    """Draw onto `frame` (any size; everything scales with its width)."""
    h, w = frame.shape[:2]
    s = w / 640
    color = MODE_COLORS[hud.mode]

    if hud.pointer_mode == "direct":  # active area: this box maps to the whole screen
        half = 0.5 / hud.direct_gain
        cv2.rectangle(frame, (int((0.5 - half) * w), int((0.5 - half) * h)),
                      (int((0.5 + half) * w), int((0.5 + half) * h)), (255, 0, 255), 1, cv2.LINE_AA)

    if second is not None:
        _hand(frame, second, color, s, faint=hud.mode is not Mode.ZOOM)
    if lm is not None:
        _hand(frame, lm, color, s)
        px, py = palm_center(lm)  # the cursor follows this point
        centre = (int(px * w), int(py * h))
        r = round(11 * s)
        cv2.circle(frame, centre, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, centre, r, WHITE, 2, cv2.LINE_AA)
        if hud.hold_progress > 0:
            ring = round(22 * s)
            cv2.ellipse(frame, centre, (ring, ring), -90, 0, 360 * min(hud.hold_progress, 1), (255, 80, 255),
                        max(2, round(4 * s)), cv2.LINE_AA)

    # Top-left: what's happening.
    status = "PAUSED (P / F9)" if hud.paused else hud.mode.value
    _text(frame, status, (round(10 * s), round(32 * s)), 0.8 * s, (0, 200, 255) if hud.paused else color, 2)
    if hud.shape and lm is not None:
        _text(frame, f"Hand: {hud.shape}", (round(12 * s), round(54 * s)), 0.42 * s, WHITE)
    if hud.warning:
        _text(frame, hud.warning, (round(12 * s), round(76 * s)), 0.45 * s, ORANGE)

    # Top-right: settings and performance.
    hand = {"any": "any hand", "right": "right hand", "left": "left hand"}[hud.hand]
    _right(frame, f"{hud.pointer_mode.upper()} {hud.dpi} DPI  {hand}", round(20 * s), 0.42 * s)
    _right(frame, f"{hud.fps:3.0f} FPS  {hud.latency_ms:3.0f} ms", round(38 * s), 0.42 * s)

    if hud.calibration_text:
        _calibration(frame, hud, s)
    else:
        if hud.show_guide:
            _status_panel(frame, hud, s)
            _guide(frame, hud, s)
        _right(frame, "+/- DPI  M mode  L hand  C calibrate  H help  P/F9 pause  Q quit", h - round(8 * s), 0.36 * s)

    if hud.toast:
        _centered(frame, hud.toast, h // 2 - round(60 * s), 0.9 * s, WHITE, 2)
