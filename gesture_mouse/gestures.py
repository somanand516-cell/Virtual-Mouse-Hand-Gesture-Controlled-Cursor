"""Gesture engine: turns hand landmarks into actions.

Pure logic (no camera, MediaPipe or OS calls) so every gesture can be unit-tested.

Gestures (defaults; one-shot and continuous ones are remappable in config.json):
  point (index up)                -> move the cursor
  point with the thumb tucked     -> precision mode: slower cursor for small targets
  pinch thumb + index             -> left click; two quick pinches = double click
  pinch and hold (or move)        -> drag and drop
  pinch thumb + middle            -> right click
  two fingers up, move up/down    -> scroll (faster movement = faster scroll)
  three fingers up, move up/down  -> volume
  rock (index + pinky), up/down   -> brightness
  fist                            -> freeze the cursor
  open palm swiped left/right     -> Alt+Shift+Tab / Alt+Tab
  shaka (pinky up) held ~0.8 s    -> screenshot
  pinch with both hands           -> zoom (move hands apart / together)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from enum import Enum

from gesture_mouse.core.actions import (
    Action, Brightness, Click, Key, Move, Press, Release, Screenshot, Scroll, Volume, Zoom,
)
from gesture_mouse.core.config import DPI_LEVELS, Config
from gesture_mouse.filters import OneEuroFilter
from gesture_mouse.vision.pose import FingerTracker, HandPose, Landmarks, Shape, analyze, shape_of

# Re-exported for older imports (overlay, tools).
from gesture_mouse.vision.pose import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, palm_center  # noqa: F401


class Mode(Enum):
    NO_HAND = "Show your hand"
    IDLE = "Idle"
    MOVE = "Move"
    PRECISION = "Precision (thumb tucked)"
    CLICK = "Pinch"
    DRAG = "Drag"
    RIGHT = "Right click"
    SCROLL = "Scroll"
    VOLUME = "Volume"
    BRIGHTNESS = "Brightness"
    ZOOM = "Zoom"
    FREEZE = "Frozen (fist)"
    PALM = "Open palm"
    HOLD = "Hold..."


CONTINUOUS = {Shape.TWO: "two_fingers", Shape.THREE: "three_fingers", Shape.ROCK: "rock"}
PINCH_CERTAIN = 0.6  # a pinch this much tighter than pinch_on needs no debounce
CONTINUOUS_MODES = {"scroll": Mode.SCROLL, "volume": Mode.VOLUME, "brightness": Mode.BRIGHTNESS}


def direct_map(value: float, gain: float) -> float:
    """Direct mode: map a normalized camera coordinate around the centre with `gain`, clamped."""
    return min(max(0.5 + (value - 0.5) * gain, 0.0), 1.0)


class GestureEngine:
    """Stateful gesture recognizer. Call `update` once per frame.

    aspect: camera width / height.  screen: screen size in pixels (for mouse-mode DPI).
    """

    def __init__(self, cfg: Config, aspect: float = 1.0, screen: tuple[int, int] = (1920, 1080)) -> None:
        self.cfg = cfg
        self.aspect = aspect
        self.screen = screen
        self.dpi = cfg.dpi
        self.pointer_mode = cfg.pointer_mode
        self.fingers = FingerTracker(cfg.finger_extend, cfg.finger_fold)
        self._fx = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self._fy = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self.mode = Mode.NO_HAND
        self.pose: HandPose | None = None   # latest pose, for the on-screen meters
        self.fingers_up: tuple[bool, ...] = (False,) * 4
        self.cursor = (0.5, 0.5)            # normalized cursor position (0..1 over all monitors)
        self.direct_area = (0.0, 0.0, 1.0, 1.0)  # direct mode maps the hand onto this part of the desktop
        self._last_click = -math.inf
        self._last_oneshot = -math.inf
        self._lost_since: float | None = None
        self._last_t: float | None = None
        self._frame_dt = 0.0
        self._reset_hand_state()

    # ---- runtime settings ----

    def change_dpi(self, step: int) -> int:
        levels = sorted(set(DPI_LEVELS) | {self.dpi})
        self.dpi = levels[min(max(levels.index(self.dpi) + step, 0), len(levels) - 1)]
        return self.dpi

    def toggle_pointer_mode(self) -> str:
        self.pointer_mode = "direct" if self.pointer_mode == "mouse" else "mouse"
        self._anchor = None
        return self.pointer_mode

    def set_thresholds(self, **values: float) -> None:
        self.cfg = replace(self.cfg, **values)
        self.fingers.extend, self.fingers.fold = self.cfg.finger_extend, self.cfg.finger_fold

    def set_cursor(self, x: float, y: float) -> None:
        """Mouse mode: continue from where the real cursor is (e.g. after the user moved the mouse)."""
        self.cursor = (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    @property
    def direct_gain(self) -> float:
        return self.cfg.direct_gain_at_800 * self.dpi / 800

    # ---- state ----

    def _reset_hand_state(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self.fingers.reset()
        self.shape: Shape | None = None
        self.thumb_tucked = False
        self._shape_cand: Shape | None = None
        self._shape_cand_t = 0.0
        self._shape_since = 0.0
        self._anchor: tuple[float, float] | None = None
        self._speed = 0.0
        self._speed_prev: tuple[float, float, float] | None = None
        self._history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=60)  # (t, cursor)
        # pinches
        self._left_cand: float | None = None
        self._right_cand: float | None = None
        self._pinching = False
        self._pinch_t = 0.0
        self._pinch_xy = (0.0, 0.0)
        self._dragging = False
        self._right_down = False
        # continuous gestures
        self._cont_name: str | None = None
        self._cont_last_y: float | None = None
        self._cont_acc = 0.0
        # one-shot gestures
        self._hold_fired = False
        self._trail: deque[tuple[float, float, float]] = deque()
        # two-hand zoom
        self._zooming = False
        self._zoom_prev: float | None = None
        self._zoom_acc = 0.0
        self.hold_progress = 0.0  # 0..1, drawn as a ring (drag hold, screenshot)

    def release(self) -> list[Action]:
        """Drop all hand state; releases the button if a drag was in progress."""
        actions: list[Action] = [Release()] if self._dragging else []
        self._reset_hand_state()
        self.mode = Mode.NO_HAND
        return actions

    # ---- per frame ----

    def update(self, lm: Landmarks | None, t: float, second: Landmarks | None = None) -> list[Action]:
        """Process one frame. `lm`: the controlling hand (None if not visible); `second`: the other
        hand if visible (for two-hand zoom); `t`: time in seconds."""
        actions = self._update(lm, second, t)
        self._history.append((t, self.cursor))
        return actions

    def _update(self, lm: Landmarks | None, second: Landmarks | None, t: float) -> list[Action]:
        cfg = self.cfg
        self._frame_dt = t - self._last_t if self._last_t is not None else 0.0
        self._last_t = t
        if lm is None:
            # Trackers often miss a frame or two; hold state briefly so drags and clicks survive.
            if self._lost_since is None:
                self._lost_since = t
            if self.mode is not Mode.NO_HAND and t - self._lost_since < cfg.hand_lost_grace:
                return []
            self.pose = None
            return self.release()
        self._lost_since = None

        pose = self.pose = analyze(lm, self.aspect, cfg.depth_weight)
        hx, hy = self._fx(pose.palm_x, t), self._fy(pose.palm_y, t)  # smoothed palm position
        self._measure_speed(hx, hy, t)
        self.fingers_up = self.fingers.update(pose.finger_ratios)
        raw = shape_of(self.fingers_up)
        self._update_tuck(pose)
        self._update_shape(raw, t)
        self.hold_progress = 0.0

        actions: list[Action] = []

        # 1. Both hands pinching: zoom.
        if self._zoom(second, pose, actions):
            self._track_hand(hx, hy)
            return actions

        # 2. Pinches take priority over hand shapes once started.
        pinch = self._pinches(pose, raw, hx, hy, t)
        if pinch is not None:
            return actions + pinch

        # 4. Hand shapes.
        shape = self.shape
        if shape is Shape.POINT:
            precise = self.thumb_tucked and self.pointer_mode == "mouse"
            self.mode = Mode.PRECISION if precise else Mode.MOVE
            if raw is not Shape.POINT:  # changing shape: hold still so e.g. starting a scroll doesn't drift
                self._track_hand(hx, hy)
                return actions
            return actions + [self._pointer(hx, hy, t, pose.pinch_index, precise)]
        self._track_hand(hx, hy)
        if shape in CONTINUOUS:
            return actions + self._continuous(CONTINUOUS[shape], hy)
        if shape is Shape.FIST:
            self.mode = Mode.FREEZE if cfg.gesture_map.get("fist") == "freeze" else Mode.IDLE
        elif shape is Shape.SHAKA:
            actions += self._hold_gesture("shaka_hold", cfg.screenshot_hold, t)
        elif shape is Shape.PALM:
            self.mode = Mode.PALM
            actions += self._swipe(hx, hy, t)
        else:
            self.mode = Mode.IDLE
        return actions

    def _measure_speed(self, hx: float, hy: float, t: float) -> None:
        prev = self._speed_prev
        self._speed_prev = (hx, hy, t)
        if prev is None:
            return
        dt = max(t - prev[2], 1e-3)
        speed = math.hypot(hx - prev[0], hy - prev[1]) / dt
        self._speed += (1 - math.exp(-dt / self.cfg.speed_smoothing)) * (speed - self._speed)

    def _update_tuck(self, pose: HandPose) -> None:
        """Thumb tucked against the palm, with hysteresis so it doesn't flicker at the border."""
        if pose.thumb_tuck < self.cfg.thumb_tuck_on:
            self.thumb_tucked = True
        elif pose.thumb_tuck > self.cfg.thumb_tuck_off:
            self.thumb_tucked = False

    def _update_shape(self, raw: Shape, t: float) -> None:
        """Debounce: a hand shape counts only after it has been stable for `pose_confirm` seconds."""
        if raw != self._shape_cand:
            self._shape_cand, self._shape_cand_t = raw, t
        if raw != self.shape and t - self._shape_cand_t >= self.cfg.pose_confirm:
            self.shape, self._shape_since = raw, t
            self._hold_fired = False
            self._cont_name = None
            self._trail.clear()

    # ---- pointer ----

    def _track_hand(self, hx: float, hy: float) -> None:
        """Keep the cursor still but remember where the hand is, so it doesn't jump later."""
        self._anchor = (hx, hy)

    def _rewind(self, t: float) -> bool:
        """Put the cursor back where it was `click_rewind` seconds ago. Returns True if it moved."""
        target = t - self.cfg.click_rewind
        for ht, cursor in reversed(self._history):
            if ht <= target:
                if cursor == self.cursor:
                    return False
                self.cursor = cursor
                return True
        return False

    def _pointer(self, hx: float, hy: float, t: float, pinch: float = math.inf, precise: bool = False) -> Move:
        cfg = self.cfg
        if self.pointer_mode == "direct":
            self._track_hand(hx, hy)
            gain = self.direct_gain
            left, top, right, bottom = self.direct_area
            self.cursor = (left + direct_map(hx / self.aspect, gain) * (right - left),
                           top + direct_map(hy, gain) * (bottom - top))
            return Move(*self.cursor)

        if self._anchor is None:
            self._anchor = (hx, hy)
            return Move(*self.cursor)

        # Rubber-band dead zone: tremor inside the radius is ignored, but once the hand moves
        # beyond it the anchor is dragged along, so every deliberate movement registers in full.
        ax, ay = self._anchor
        vx, vy = hx - ax, hy - ay
        dist = math.hypot(vx, vy)
        if dist <= cfg.precision_radius:
            return Move(*self.cursor)
        k = (dist - cfg.precision_radius) / dist
        mx, my = vx * k, vy * k
        self._anchor = (ax + mx, ay + my)

        # Acceleration, like a trackpad: slow = precise, fast = covers the screen.
        ramp = min(self._speed / cfg.accel_speed, 1.0) ** cfg.accel_curve
        gain = cfg.accel_min + (cfg.accel_max - cfg.accel_min) * ramp
        if pinch < cfg.pinch_off * cfg.aim_brake_zone:
            gain *= cfg.aim_brake  # fingers are closing for a click: slow down to aim
        if precise:
            gain *= cfg.precision_gain  # thumb tucked: precision mode for small targets
        px = cfg.pixels_per_unit_at_800 * self.dpi / 800 * gain
        sw, sh = self.screen
        self.set_cursor(self.cursor[0] + mx * px / sw, self.cursor[1] + my * px / sh)
        return Move(*self.cursor)

    # ---- pinches: left click / double click / drag, right click ----

    def _pinches(self, pose: HandPose, raw: Shape, hx: float, hy: float, t: float) -> list[Action] | None:
        """Returns actions if a pinch is starting, held or ending; None if no pinch is involved."""
        cfg = self.cfg
        li, mi = pose.pinch_index, pose.pinch_middle

        if self._right_down:
            if mi <= cfg.pinch_off:
                self.mode = Mode.RIGHT
                self._track_hand(hx, hy)
                return []
            self._right_down = False

        if self._pinching:
            if li > cfg.pinch_off:
                return self._end_left(t, hx, hy)
            return self._hold_left(hx, hy, t)

        allowed = raw is not Shape.FIST and self.shape is not Shape.FIST  # a fist isn't a pinch
        index_closer = li <= mi  # a thumb-middle pinch also brings the thumb near the index
        # Left pinch (thumb + index). Debounced: a borderline pinch must persist for pinch_confirm,
        # but a clearly closed one counts at once (quick clicks on slow cameras last only 1-2 frames).
        if allowed and index_closer and li < cfg.pinch_on:
            if self._left_cand is None:
                self._left_cand = t
            if self._pinch_confirmed(self._left_cand, li, t):
                return self._start_left(hx, hy, t)
            self.mode = Mode.CLICK
            self._track_hand(hx, hy)
            return []
        self._left_cand = None

        # Right click (thumb + middle), debounced; fires as soon as it's confirmed. A tucked thumb
        # rests next to the curled middle finger: that is precision mode, not a right click.
        if allowed and not index_closer and mi < cfg.pinch_on and not self.thumb_tucked:
            if self._right_cand is None:
                self._right_cand = t
            self._track_hand(hx, hy)
            if self._pinch_confirmed(self._right_cand, mi, t):
                self._right_cand = None
                self._right_down = True
                self.mode = Mode.RIGHT
                actions: list[Action] = [Move(*self.cursor)] if self._rewind(t) else []
                return actions + ([Click("right")] if self._click_ready(t) else [])
            self.mode = Mode.CLICK
            return []
        self._right_cand = None
        return None

    def _pinch_confirmed(self, since: float, distance: float, t: float) -> bool:
        """Debounce: the pinch must have lasted pinch_confirm. On a slow camera one frame already
        spans that long, and a clearly closed pinch is never accidental, so both count at once."""
        cfg = self.cfg
        return (t - since >= cfg.pinch_confirm
                or self._frame_dt >= cfg.pinch_confirm
                or distance < cfg.pinch_on * PINCH_CERTAIN)

    def _start_left(self, hx: float, hy: float, t: float) -> list[Action]:
        self._pinching = True
        self._pinch_t = self._left_cand if self._left_cand is not None else t
        self._left_cand = None
        self._pinch_xy = (hx, hy)
        self.mode = Mode.CLICK
        self._track_hand(hx, hy)
        # Click where you were aiming, before the fingers started closing.
        return [Move(*self.cursor)] if self._rewind(t) else []

    def _hold_left(self, hx: float, hy: float, t: float) -> list[Action]:
        cfg = self.cfg
        if self._dragging:
            self.mode = Mode.DRAG
            return [self._pointer(hx, hy, t)]
        held = t - self._pinch_t
        moved = math.dist((hx, hy), self._pinch_xy) > cfg.drag_start_distance
        if held >= cfg.drag_hold or moved:
            self._dragging = True
            self.mode = Mode.DRAG
            return [Press(), self._pointer(hx, hy, t)]
        self.mode = Mode.CLICK  # cursor holds still so the click lands where you aimed
        self.hold_progress = held / cfg.drag_hold
        self._track_hand(hx, hy)
        return []

    def _end_left(self, t: float, hx: float, hy: float) -> list[Action]:
        actions: list[Action] = []
        if self._dragging:
            actions.append(Release())
        elif self._click_ready(t):
            actions.append(Click("left"))
        self._pinching = self._dragging = False
        self.mode = Mode.MOVE
        self._track_hand(hx, hy)
        return actions

    def _cancel_pinch(self) -> list[Action]:
        """Abandon a pinch without clicking (e.g. it turned into a two-hand zoom)."""
        actions: list[Action] = [Release()] if self._dragging else []
        self._pinching = self._dragging = self._right_down = False
        self._left_cand = self._right_cand = None
        return actions

    def _click_ready(self, t: float) -> bool:
        if t - self._last_click < self.cfg.click_cooldown:
            return False
        self._last_click = t
        return True

    # ---- continuous gestures: scroll / volume / brightness ----

    def _continuous(self, name: str, y: float) -> list[Action]:
        cfg = self.cfg
        target = cfg.gesture_map.get(name, "none")
        self.mode = CONTINUOUS_MODES.get(target, Mode.IDLE)
        if self._cont_name != name:
            self._cont_name, self._cont_last_y, self._cont_acc = name, y, 0.0
            return []
        assert self._cont_last_y is not None
        dy = self._cont_last_y - y  # hand moved up -> positive
        self._cont_last_y = y

        if target == "scroll":
            boost = 1 + cfg.scroll_accel * min(self._speed / cfg.accel_speed, 1.0)  # faster = further
            self._cont_acc += dy * cfg.scroll_speed * boost
            step = cfg.scroll_step
            amount = int(self._cont_acc / step) * step  # whole steps only, so tremor doesn't scroll
            if amount:
                self._cont_acc -= amount
                return [Scroll(amount)]
        elif target == "volume":
            self._cont_acc += dy * cfg.volume_speed
            steps = int(self._cont_acc)
            if steps:
                self._cont_acc -= steps
                return [Volume(steps)]
        elif target == "brightness":
            self._cont_acc += dy * cfg.brightness_speed
            delta = int(self._cont_acc / 2) * 2  # 2% steps
            if delta:
                self._cont_acc -= delta
                return [Brightness(delta)]
        return []

    # ---- one-shot gestures ----

    def _fire(self, name: str, t: float) -> list[Action]:
        target = self.cfg.gesture_map.get(name, "none")
        if target == "none" or t - self._last_oneshot < self.cfg.gesture_cooldown:
            return []
        self._last_oneshot = t
        if target == "screenshot":
            return [Screenshot()]
        if target.startswith("key:"):
            return [Key(target[4:])]
        return []

    def _hold_gesture(self, name: str, hold: float, t: float) -> list[Action]:
        if self._hold_fired:
            self.mode = Mode.IDLE
            return []
        held = t - self._shape_since
        self.hold_progress = min(held / hold, 1.0)
        self.mode = Mode.HOLD
        if held >= hold:
            self._hold_fired = True
            return self._fire(name, t)
        return []

    def _swipe(self, hx: float, hy: float, t: float) -> list[Action]:
        cfg = self.cfg
        self._trail.append((t, hx, hy))
        while self._trail and t - self._trail[0][0] > cfg.swipe_time:
            self._trail.popleft()
        dx = self._trail[-1][1] - self._trail[0][1]
        dy = self._trail[-1][2] - self._trail[0][2]
        if abs(dx) >= cfg.swipe_distance and abs(dy) < 0.6 * abs(dx):
            self._trail.clear()
            # The image is mirrored, so +x is the user's right.
            return self._fire("swipe_right" if dx > 0 else "swipe_left", t)
        return []

    # ---- two-hand zoom ----

    def _zoom(self, second: Landmarks | None, pose: HandPose, actions: list[Action]) -> bool:
        cfg = self.cfg
        if second is None or cfg.gesture_map.get("two_hand_pinch") != "zoom":
            self._zooming = False
            return False
        other = analyze(second, self.aspect, cfg.depth_weight)
        limit = cfg.pinch_off if self._zooming else cfg.pinch_on
        if not (pose.pinch_index < limit and other.pinch_index < limit):
            self._zooming = False
            return False
        if not self._zooming:
            self._zooming, self._zoom_prev, self._zoom_acc = True, None, 0.0
            actions += self._cancel_pinch()  # it's a zoom, not a click
        d = math.dist((pose.pinch_x, pose.pinch_y), (other.pinch_x, other.pinch_y))
        if self._zoom_prev is not None:
            self._zoom_acc += (d - self._zoom_prev) * cfg.zoom_speed
            steps = int(self._zoom_acc)
            if steps:
                self._zoom_acc -= steps
                actions.append(Zoom(steps))
        self._zoom_prev = d
        self.mode = Mode.ZOOM
        return True
