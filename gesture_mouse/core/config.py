"""All tunable settings in one place.

Distances on the hand are measured in "camera units": 1.0 = the height of the camera image.

Files next to the project:
  config.json           YOUR changes only (calibration, DPI, anything you edit). Settings not in it
                        use the defaults below, so improved defaults in new versions reach you.
  config.defaults.json  every setting with its default, rewritten on each start: a reference to
                        copy from. Editing it does nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"

# Settings that existed in older versions. Found in a config.json, they are removed quietly.
REMOVED_SETTINGS = {"palm_hold", "palm_spread", "palm_still_speed", "toggle_hotkey"}
REMOVED_GESTURES = {"palm_hold"}

DPI_LEVELS = (400, 600, 800, 1200, 1600, 2400, 3200)
POINTER_MODES = ("mouse", "direct")
HAND_MODES = ("any", "right", "left")

# What each gesture does. Continuous gestures (moved up/down while held) accept:
#   "scroll", "volume", "brightness", "none".
# One-shot gestures accept: "screenshot", "none", or "key:<combo>" e.g. "key:alt+tab".
DEFAULT_GESTURE_MAP = {
    "two_fingers": "scroll",          # index + middle up, move up/down
    "three_fingers": "volume",        # index + middle + ring up, move up/down
    "rock": "brightness",             # index + pinky up, move up/down
    "fist": "freeze",                 # "freeze" (hold cursor still) or "none"
    "swipe_left": "key:alt+shift+tab",    # open palm swiped left / right
    "swipe_right": "key:alt+tab",
    "shaka_hold": "screenshot",       # only the pinky up (or thumb + pinky), held ~0.8 s
    "two_hand_pinch": "zoom",         # pinch with both hands, move them apart / together
}


@dataclass(frozen=True)
class Config:
    # ---- Camera ----
    camera_index: int = 0
    frame_width: int = 960            # 960x540: fast, yet enough pixels for steady landmarks
    frame_height: int = 540
    mirror: bool = True
    camera_exposure: float | None = None  # None = automatic. Fixed values are camera-specific (e.g. -5)
    low_light_mode: bool = False      # EXPERIMENTAL: fixed short exposure (30 fps) + software brightening.
                                      # Many webcams drop to ~10 fps in dim light; this trades image noise
                                      # for frame rate. Not yet verified for tracking quality on real hands.

    # ---- Hand tracking ----
    model_path: str = str(PROJECT_ROOT / "models" / "hand_landmarker.task")
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    hand: str = "any"                 # "any", "right" or "left": which hand controls the mouse
    depth_weight: float = 0.5         # how much MediaPipe's (noisy) depth estimate counts in distances

    # ---- Pointer ----
    # "mouse":  relative + accelerated, like a trackpad. Hide your hand to "lift" and reposition.
    # "direct": the cursor goes where your hand is inside the active-area box.
    pointer_mode: str = "mouse"
    dpi: int = 800                    # sensitivity; change live with + / -
    pixels_per_unit_at_800: float = 4000.0  # mouse mode: px per camera unit of hand travel at 800 DPI
    accel_min: float = 0.5            # mouse mode gain for very slow, precise movement
    accel_max: float = 2.4            # mouse mode gain for fast flicks
    accel_speed: float = 0.8          # hand speed (camera units/s) where gain reaches accel_max
    accel_curve: float = 1.0          # >1 keeps gain low longer (more precision, less reach)
    speed_smoothing: float = 0.05     # seconds; hand speed is averaged so the gain doesn't flicker
    precision_radius: float = 0.0025  # dead zone: tremor smaller than this (camera units) is ignored
    direct_gain_at_800: float = 1.8   # direct mode: the active box is 1/gain of the frame at 800 DPI

    # ---- Precision mode: point with the thumb tucked against the palm (mouse mode) ----
    precision_gain: float = 0.35      # cursor speed factor while the thumb is tucked
    thumb_tuck_on: float = 0.35       # thumb tip to palm centre / hand size below this = tucked
    thumb_tuck_off: float = 0.45      # above this = not tucked any more (hysteresis)

    # ---- Smoothing (One Euro filter) ----
    filter_min_cutoff: float = 1.0    # lower = steadier when still
    filter_beta: float = 10.0         # higher = less lag when moving fast
    cursor_glide: float = 0.025       # seconds; the output thread glides the cursor between camera frames
    output_rate_hz: float = 240.0     # how often the output thread updates the cursor

    # ---- Finger detection (ratios of fingertip-to-wrist / knuckle-to-wrist; calibration tunes) ----
    finger_extend: float = 1.12       # above = finger up
    finger_fold: float = 0.95         # below = finger down (in between keeps the previous state)
    pose_confirm: float = 0.10        # seconds a hand shape must be stable before it counts

    # ---- Pinch / click (thumb-to-fingertip distance / hand size; calibration tunes) ----
    pinch_on: float = 0.32            # closer than this = pinched
    pinch_off: float = 0.47           # further than this = released (hysteresis)
    pinch_confirm: float = 0.05       # seconds a pinch must hold before it counts (debounce)
    click_cooldown: float = 0.15      # seconds between clicks (short enough for double clicks)
    drag_hold: float = 0.35           # pinch held this long = start dragging
    drag_start_distance: float = 0.02 # ...or moved this far while pinched
    aim_brake: float = 0.3            # cursor speed factor while fingers are closing for a pinch
    aim_brake_zone: float = 1.6       # braking starts when pinch distance < pinch_off * this
    click_rewind: float = 0.15        # a pinch clicks where the cursor was this long before

    # ---- Scroll / volume / brightness / zoom (hand moved up/down while holding the shape) ----
    scroll_speed: float = 4500.0      # wheel units per camera unit (120 = one notch)
    scroll_accel: float = 1.5         # extra scroll speed for fast movement
    scroll_step: int = 40
    volume_speed: float = 60.0        # volume key presses (2% each) per camera unit
    brightness_speed: float = 150.0   # brightness % per camera unit
    zoom_speed: float = 25.0          # zoom steps per camera unit change in distance between hands

    # ---- One-shot gestures ----
    screenshot_hold: float = 0.8
    swipe_distance: float = 0.18      # camera units of horizontal travel...
    swipe_time: float = 0.35          # ...within this many seconds
    gesture_cooldown: float = 0.8     # seconds between one-shot gestures

    # ---- Low-light warning (shown in the preview) ----
    low_light_level: float = 60.0     # average image brightness (0-255) below which the preview warns
    low_fps_warning: float = 15.0     # ...or real camera fps below this (webcams slow down in the dark)

    # ---- Safety / robustness ----
    hand_lost_grace: float = 0.3      # seconds the hand may vanish before we let go of everything
    override_time: float = 1.0        # after you touch the real mouse, gestures wait this long (seconds)
    watchdog_timeout: float = 1.0     # release held buttons if tracking freezes this long (seconds)
    quit_hotkey: str = "ctrl+alt+q"   # works even when the preview window isn't focused
    pause_hotkey: str = "f9"          # gesture control off / on from anywhere (same as P in the preview)
    screenshot_dir: str = str(Path.home() / "Pictures" / "Gesture Mouse")

    gesture_map: dict = field(default_factory=lambda: dict(DEFAULT_GESTURE_MAP))

    def __post_init__(self) -> None:
        if not 0 < self.pinch_on < self.pinch_off:
            raise ValueError("expected 0 < pinch_on < pinch_off")
        if not 0 < self.finger_fold < self.finger_extend:
            raise ValueError("expected 0 < finger_fold < finger_extend")
        if not 0 < self.thumb_tuck_on < self.thumb_tuck_off:
            raise ValueError("expected 0 < thumb_tuck_on < thumb_tuck_off")
        if self.scroll_step <= 0:
            raise ValueError("scroll_step must be positive")
        if self.pointer_mode not in POINTER_MODES:
            raise ValueError(f"pointer_mode must be one of {POINTER_MODES}")
        if self.hand not in HAND_MODES:
            raise ValueError(f"hand must be one of {HAND_MODES}")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        unknown = set(self.gesture_map) - set(DEFAULT_GESTURE_MAP)
        if unknown:
            raise ValueError(f"unknown gestures in gesture_map: {sorted(unknown)}")
        bad = [g for g, a in self.gesture_map.items() if a == "toggle"]
        if bad:
            raise ValueError(f"\"toggle\" (mouse on/off) was removed; used by {bad}")


# ---- config.json ----------------------------------------------------------

def _defaults_path(path: Path) -> Path:
    return path.with_name("config.defaults.json")


def write_defaults_reference(path: Path | None = None) -> None:
    """config.defaults.json: every setting and its default, for reference."""
    target = _defaults_path(path or CONFIG_PATH)
    try:
        target.write_text(json.dumps(asdict(Config()), indent=2), encoding="utf-8")
    except OSError as e:
        log.debug("Could not write %s (%s)", target, e)


def _only_changes(data: dict) -> dict:
    """Drop removed settings and anything equal to the default, so defaults can improve later."""
    defaults = asdict(Config())
    kept: dict = {}
    for key, value in data.items():
        if key in REMOVED_SETTINGS:
            continue
        if key == "gesture_map" and isinstance(value, dict):
            changed = {g: a for g, a in value.items()
                       if g not in REMOVED_GESTURES and DEFAULT_GESTURE_MAP.get(g) != a}
            if changed:
                kept[key] = changed
        elif key not in defaults or defaults[key] != value:
            kept[key] = value
    return kept


def load(path: Path | None = None, base: Config | None = None) -> Config:
    """Read config.json (your changes) over the defaults. A missing file is created empty."""
    path = path or CONFIG_PATH
    cfg = base or Config()
    write_defaults_reference(path)
    if not path.exists():
        _write(path, {})
        log.info("Created %s for your settings (all options: %s).", path.name, _defaults_path(path).name)
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        log.warning("Ignoring unreadable %s (%s); using defaults.", path, e)
        return cfg
    if not isinstance(data, dict):
        log.warning("Ignoring %s: expected {...}; using defaults.", path)
        return cfg

    changes = _only_changes(data)
    if changes != data:  # older versions saved every default, and removed settings may linger
        log.info("Tidied %s: kept your %d own setting(s), removed copies of defaults.", path.name, len(changes))
        _write(path, changes)

    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in changes.items() if k in known}
    ignored = sorted(set(changes) - known)
    if ignored:
        log.warning("Ignoring unknown settings in %s: %s", path.name, ", ".join(ignored))
    if isinstance(values.get("gesture_map"), dict):
        unknown = sorted(set(values["gesture_map"]) - set(DEFAULT_GESTURE_MAP))
        if unknown:
            log.warning("Ignoring unknown gestures in %s: %s", path.name, ", ".join(unknown))
        values["gesture_map"] = {**DEFAULT_GESTURE_MAP,
                                 **{g: a for g, a in values["gesture_map"].items() if g in DEFAULT_GESTURE_MAP}}
    try:
        return replace(cfg, **values)
    except (TypeError, ValueError) as e:
        log.warning("Invalid settings in %s (%s); using defaults.", path.name, e)
        return cfg


CALIBRATED_SETTINGS = ("pinch_on", "pinch_off", "finger_extend", "finger_fold")


def has_calibration(path: Path | None = None) -> bool:
    """True once a calibration result is saved in config.json. False on first launch."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and any(k in data for k in CALIBRATED_SETTINGS)


def _write(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("Could not write %s (%s)", path, e)


def update(values: dict, path: Path | None = None) -> None:
    """Save a few changed settings (DPI, mode, calibration) without touching the rest."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    data.update(values)
    _write(path, _only_changes(data))