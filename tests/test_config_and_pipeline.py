import json
import time

from gesture_mouse.core import config as config_file
from gesture_mouse.core.actions import Brightness, Screenshot
from gesture_mouse.core.config import DEFAULT_GESTURE_MAP, Config
from gesture_mouse.pipeline import Pipeline
from gesture_mouse.output.system import SystemActions
from gesture_mouse.vision.tracker import TrackedHand
from hands import make_hand


# ---- config.json ----

def test_missing_config_is_created_empty_with_reference(tmp_path):
    path = tmp_path / "config.json"
    cfg = config_file.load(path)
    assert cfg == Config() and json.loads(path.read_text()) == {}
    assert json.loads((tmp_path / "config.defaults.json").read_text())["dpi"] == 800


def test_old_full_config_is_tidied_keeping_calibration(tmp_path):
    # Exactly the shape of the real config.json written by the earlier version: every default
    # copied in, the removed on/off settings, plus four calibrated values.
    from dataclasses import asdict

    old = asdict(Config())
    old.update({"palm_hold": 1.0, "palm_still_speed": 0.08, "palm_spread": 0.61,
                "finger_extend": 1.054, "finger_fold": 0.938, "pinch_on": 0.394, "pinch_off": 0.542})
    old["gesture_map"] = dict(DEFAULT_GESTURE_MAP, palm_hold="toggle")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(old))
    cfg = config_file.load(path)
    assert json.loads(path.read_text()) == {
        "finger_extend": 1.054, "finger_fold": 0.938, "pinch_on": 0.394, "pinch_off": 0.542}
    assert (cfg.finger_extend, cfg.pinch_on) == (1.054, 0.394)
    assert "palm_hold" not in cfg.gesture_map


def test_unknown_gesture_in_config_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dpi": 1600, "gesture_map": {"wave": "key:a", "rock": "none"}}))
    cfg = config_file.load(path)
    assert cfg.dpi == 1600 and cfg.gesture_map["rock"] == "none" and "wave" not in cfg.gesture_map


def test_config_overrides_and_merges_gesture_map(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dpi": 1600, "hand": "left", "gesture_map": {"rock": "none"}, "junk": 1}))
    cfg = config_file.load(path)
    assert (cfg.dpi, cfg.hand) == (1600, "left")
    assert cfg.gesture_map == {**DEFAULT_GESTURE_MAP, "rock": "none"}


def test_bad_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    assert config_file.load(path) == Config()
    path.write_text(json.dumps({"pinch_on": 0.9, "pinch_off": 0.1}))
    assert config_file.load(path) == Config()


def test_update_keeps_other_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dpi": 1600, "hand": "left"}))
    config_file.update({"dpi": 2400}, path)
    assert json.loads(path.read_text()) == {"dpi": 2400, "hand": "left"}
    config_file.update({"dpi": 800}, path)  # back to the default: no need to store it
    assert json.loads(path.read_text()) == {"hand": "left"}


# ---- choosing the controlling hand ----

def hand(side, x0, score=0.9):
    return TrackedHand(make_hand(x0=x0), side, score)


def pick(cfg, *frames):
    p = Pipeline(cfg, None, None, None, None, None)
    return [p._pick_hands(list(hands)) for hands in frames]


def test_right_hand_mode_ignores_left_hand():
    (primary, second), = pick(Config(hand="right"), [hand("Left", 0.3), hand("Right", 0.7)])
    assert primary.side == "Right" and second.side == "Left"
    (primary, _), = pick(Config(hand="right"), [hand("Left", 0.3)])
    assert primary is None


def test_any_hand_sticks_with_the_same_hand():
    a1, b1 = hand("Right", 0.3, score=0.9), hand("Left", 0.7, score=0.8)
    a2, b2 = hand("Right", 0.31, score=0.7), hand("Left", 0.71, score=0.95)  # confidence order flips
    first, second = pick(Config(), [a1, b1], [b2, a2])
    assert first[0] is a1 and second[0] is a2


# ---- system actions (dry run) ----

def test_screenshot_and_brightness_dry_run(tmp_path):
    notes = []
    system = SystemActions(str(tmp_path / "shots"), notes.append, dry_run=True)
    system.apply(Screenshot())
    system.apply(Brightness(4))
    system.apply(Brightness(-2))
    time.sleep(0.3)
    system.close()
    assert any(n.startswith("Screenshot saved") for n in notes)
    assert (tmp_path / "shots").is_dir()


def test_pause_hotkey_toggles_pause_through_the_key_queue():
    from gesture_mouse.infra.hotkeys import _split

    cfg = Config()
    assert _split(cfg.pause_hotkey)  # F9 is a valid global hotkey
    p = Pipeline(cfg, None, None, None, None, None)
    p.command(ord("p"))  # what the F9 callback does, from the hotkey thread
    p._handle_commands(0.0)
    assert p.paused
    p.command(ord("p"))
    p._handle_commands(0.0)
    assert not p.paused


def test_config_saved_with_bom_still_loads(tmp_path):
    # Notepad and PowerShell often save UTF-8 with a byte-order mark; that used to be ignored silently.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dpi": 1600}), encoding="utf-8-sig")
    assert config_file.load(path).dpi == 1600


def test_has_calibration_only_after_one_is_saved(tmp_path):
    path = tmp_path / "config.json"
    assert not config_file.has_calibration(path)  # first launch: no file
    config_file.load(path)
    config_file.update({"dpi": 1600}, path)
    assert not config_file.has_calibration(path)
    config_file.update({"pinch_on": 0.3, "pinch_off": 0.45}, path)
    assert config_file.has_calibration(path)


def test_c_starts_and_skips_calibration():
    p = Pipeline(Config(), None, None, None, None, None)
    p.command(ord("c"))
    p._handle_commands(0.0)
    assert p.calibration is not None
    p.command(ord("c"))
    p._handle_commands(1.0)
    assert p.calibration is None and "skipped" in p._toast[0]
