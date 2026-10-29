from dataclasses import replace

import pytest

from gesture_mouse.core.actions import (
    Brightness, Click, Key, Move, Notify, Press, Release, Screenshot, Scroll, Volume, Zoom,
)
from gesture_mouse.core.config import DEFAULT_GESTURE_MAP, Config
from gesture_mouse.gestures import GestureEngine, Mode, direct_map
from hands import FIST, PALM, POINT, ROCK, SHAKA, THREE, TWO, make_hand, run, seconds


def kinds(actions, *types):
    return [a for a in actions if isinstance(a, types)]


def non_moves(actions):
    return [a for a in actions if not isinstance(a, Move)]


@pytest.fixture
def engine():
    return GestureEngine(Config())


def settle(engine, fingers=POINT, n=6, t0=0.0, **kw):
    """Show a hand shape long enough for it to be confirmed."""
    return run(engine, [make_hand(fingers, **kw)] * n, t0=t0)


# ---- moving ----

def test_point_moves_cursor_after_shape_is_confirmed(engine):
    first = engine.update(make_hand(), 0.0)
    assert first == [] and engine.mode is not Mode.MOVE  # not confirmed yet (debounce)
    actions = settle(engine, t0=0.04)
    assert engine.mode is Mode.MOVE and kinds(actions, Move)


def test_shape_debounce_ignores_one_frame_flicker(engine):
    settle(engine)
    run(engine, [make_hand(TWO)], t0=1.0)  # a single frame of "two fingers"
    actions = run(engine, [make_hand(POINT, dy=-0.05)] * 3, t0=1.04)
    assert engine.mode is Mode.MOVE and not kinds(actions, Scroll)


def test_mouse_mode_moves_with_hand(engine):
    settle(engine)
    out = kinds(run(engine, [make_hand(dx=i * 0.01) for i in range(20)], t0=1.0), Move)
    assert out[-1].x > out[0].x + 0.05 and out[-1].y == pytest.approx(out[0].y)


def test_still_hand_holds_cursor(engine):
    settle(engine)
    out = kinds(run(engine, [make_hand(dx=0.0005 * (i % 2)) for i in range(30)], t0=1.0), Move)
    assert len({(m.x, m.y) for m in out}) == 1


def test_slow_precise_move_registers_fully(engine):
    settle(engine)
    start = engine.cursor[0]
    run(engine, [make_hand(dx=0.012 * i / 60) for i in range(61)], t0=1.0)
    cfg = Config()
    expected_px = (0.012 - cfg.precision_radius) * cfg.pixels_per_unit_at_800 * cfg.accel_min
    assert (engine.cursor[0] - start) * 1920 == pytest.approx(expected_px, rel=0.25)


def test_higher_dpi_moves_further():
    def travel(dpi):
        e = GestureEngine(Config(dpi=dpi))
        settle(e)
        out = kinds(run(e, [make_hand(dx=i * 0.005) for i in range(20)], t0=1.0), Move)
        return out[-1].x - out[0].x

    assert travel(1600) > travel(800) * 1.8


def test_acceleration_fast_goes_further_than_slow():
    def travel(frames):
        e = GestureEngine(Config())
        settle(e)
        out = kinds(run(e, [make_hand(dx=0.1 * i / (frames - 1)) for i in range(frames)], t0=1.0), Move)
        return out[-1].x - out[0].x

    assert travel(5) > travel(60) * 1.5


def test_aim_brake_slows_cursor_while_pinch_closes():
    def travel(thumb_near):
        e = GestureEngine(Config())
        settle(e)
        frames = []
        for i in range(20):
            lm = make_hand(dx=i * 0.004)
            if thumb_near:  # close to the index tip, but not a pinch yet
                lm[4] = type(lm[8])(lm[8].x - 0.065, lm[8].y)
            frames.append(lm)
        out = kinds(run(e, frames, t0=1.0), Move)
        return out[-1].x - out[0].x

    assert travel(True) < travel(False) * 0.5


def test_direct_mode_follows_hand():
    e = GestureEngine(Config(pointer_mode="direct"))
    out = kinds(run(e, [make_hand(dx=-0.2)] * 10 + [make_hand(dx=0.2)] * 30), Move)
    assert out[-1].x > out[0].x + 0.4


@pytest.mark.parametrize("value,gain,expected", [(0.5, 2, 0.5), (0.25, 2, 0.0), (0.6, 2, 0.7)])
def test_direct_map(value, gain, expected):
    assert direct_map(value, gain) == pytest.approx(expected)


def test_dpi_and_mode_changes(engine):
    assert engine.change_dpi(+1) == 1200
    for _ in range(20):
        engine.change_dpi(-1)
    assert engine.dpi == 400
    assert engine.toggle_pointer_mode() == "direct"


# ---- clicks ----

def test_quick_pinch_left_clicks(engine):
    settle(engine)
    actions = run(engine, [make_hand(pinch="index")] * 4 + [make_hand()], t0=1.0)
    assert non_moves(actions) == [Click("left")]


def borderline_pinch():
    """Thumb near the index tip: below pinch_on, but not clearly closed."""
    lm = make_hand()
    lm[4] = type(lm[8])(lm[8].x - 0.042, lm[8].y)
    return lm


def test_one_frame_borderline_pinch_is_ignored(engine):
    settle(engine)  # 6 frames at 30 fps; continue at the same frame rate
    assert non_moves(run(engine, [borderline_pinch(), make_hand(), make_hand()], t0=6 / 30)) == []


def test_borderline_pinch_that_persists_clicks(engine):
    settle(engine)
    assert non_moves(run(engine, [borderline_pinch()] * 4 + [make_hand()], t0=1.0)) == [Click("left")]


def test_one_frame_borderline_pinch_counts_on_slow_camera(engine):
    # at 12 fps one frame lasts ~83 ms, longer than the 50 ms debounce
    actions, t = [], 0.0
    for lm in [make_hand()] * 6 + [borderline_pinch(), make_hand()]:
        actions += engine.update(lm, t)
        t += 1 / 12
    assert non_moves(actions) == [Click("left")]


def test_cursor_still_while_pinched(engine):
    settle(engine)
    assert kinds(run(engine, [make_hand(pinch="index", dx=0.005)] * 5, t0=1.0), Move) == []


def test_double_pinch_double_clicks(engine):
    settle(engine)
    cycle = [make_hand(pinch="index")] * 4 + [make_hand()] * 3
    assert non_moves(run(engine, cycle * 2, t0=1.0)) == [Click("left"), Click("left")]


def test_click_rewinds_to_where_you_aimed(engine):
    settle(engine, n=10)
    aimed = engine.cursor
    actions = run(engine, [make_hand(dx=i * 0.003) for i in range(1, 4)]
                  + [make_hand(pinch="index", dx=0.009)] * 4 + [make_hand(dx=0.009)], t0=1.0)
    click_at = actions.index(Click("left"))
    landed = kinds(actions[:click_at], Move)[-1]
    assert (landed.x, landed.y) == pytest.approx(aimed, abs=1 / 1920)


def test_right_click_thumb_middle(engine):
    settle(engine)
    actions = run(engine, [make_hand(pinch="middle")] * 6 + [make_hand()] * 2, t0=1.0)
    assert non_moves(actions) == [Click("right")]


def test_fist_never_clicks(engine):
    actions = run(engine, [make_hand(FIST)] * 20)
    assert not kinds(actions, Click, Press)


# ---- drag ----

def test_pinch_and_hold_drags(engine):
    settle(engine)
    hold = seconds(Config().drag_hold) + 3
    actions = run(engine, [make_hand(pinch="index")] * hold + [make_hand(pinch="index", dx=i * 0.01) for i in range(10)]
                  + [make_hand(dx=0.1)], t0=1.0)
    assert non_moves(actions) == [Press(), Release()]
    after = kinds(actions[actions.index(Press()):], Move)
    assert after[-1].x > after[0].x


def test_pinch_and_move_drags_immediately(engine):
    settle(engine)
    actions = run(engine, [make_hand(pinch="index", dx=i * 0.01) for i in range(8)] + [make_hand(dx=0.08)], t0=1.0)
    assert non_moves(actions) == [Press(), Release()]


def test_losing_hand_mid_drag_releases(engine):
    settle(engine)
    lost = seconds(Config().hand_lost_grace) + 2
    actions = run(engine, [make_hand(pinch="index", dx=i * 0.01) for i in range(10)] + [None] * lost, t0=1.0)
    assert actions[-1] == Release() and engine.mode is Mode.NO_HAND


def test_brief_dropout_keeps_drag(engine):
    settle(engine)
    frames = ([make_hand(pinch="index", dx=i * 0.01) for i in range(10)] + [None] * 3
              + [make_hand(pinch="index", dx=0.1 + i * 0.01) for i in range(5)] + [make_hand(dx=0.15)])
    assert non_moves(run(engine, frames, t0=1.0)) == [Press(), Release()]


# ---- continuous: scroll / volume / brightness ----

def test_two_fingers_scroll_up_and_down(engine):
    settle(engine, TWO)
    up = kinds(run(engine, [make_hand(TWO, dy=-i * 0.01) for i in range(15)], t0=1.0), Scroll)
    assert up and all(s.amount > 0 for s in up) and engine.mode is Mode.SCROLL
    e = GestureEngine(Config())
    settle(e, TWO)
    down = kinds(run(e, [make_hand(TWO, dy=i * 0.01) for i in range(15)], t0=1.0), Scroll)
    assert down and all(s.amount < 0 for s in down)


def test_faster_scroll_goes_further(engine):
    def total(frames):
        e = GestureEngine(Config())
        settle(e, TWO)
        return sum(s.amount for s in kinds(run(e, [make_hand(TWO, dy=-0.1 * i / (frames - 1))
                                                    for i in range(frames)], t0=1.0), Scroll))

    assert total(4) > total(40) * 1.3


def test_two_fingers_do_not_move_cursor(engine):
    settle(engine)
    before = engine.cursor
    run(engine, [make_hand(TWO, dy=-i * 0.01) for i in range(15)], t0=1.0)
    assert engine.cursor == before


def test_three_fingers_volume(engine):
    settle(engine, THREE)
    vol = kinds(run(engine, [make_hand(THREE, dy=-i * 0.01) for i in range(15)], t0=1.0), Volume)
    assert vol and sum(v.steps for v in vol) > 0 and engine.mode is Mode.VOLUME


def test_rock_brightness_down(engine):
    settle(engine, ROCK)
    b = kinds(run(engine, [make_hand(ROCK, dy=i * 0.01) for i in range(15)], t0=1.0), Brightness)
    assert b and sum(x.delta for x in b) < 0 and engine.mode is Mode.BRIGHTNESS


# ---- fist / palm / shaka / swipe ----

def test_fist_freezes_cursor(engine):
    settle(engine)
    before = engine.cursor
    actions = run(engine, [make_hand(FIST, dx=i * 0.01) for i in range(20)], t0=1.0)
    assert engine.mode is Mode.FREEZE and engine.cursor == before and not kinds(actions, Move)


def test_mouse_never_turns_itself_off(engine):
    # The on/off feature was removed: no hand pose, held however long, can disable the mouse.
    for spread in (1, 2):
        actions = run(engine, [make_hand(PALM, spread=spread)] * seconds(5))
        assert not kinds(actions, Notify)
    settle(engine, t0=20.0)
    assert kinds(run(engine, [make_hand(dx=i * 0.01) for i in range(10)], t0=21.0), Move)
    assert not hasattr(engine, "toggle") and not hasattr(engine, "enabled")


def test_toggle_mapping_is_rejected():
    with pytest.raises(ValueError):
        Config(gesture_map=dict(DEFAULT_GESTURE_MAP, shaka_hold="toggle"))

def test_shaka_hold_takes_one_screenshot(engine):
    actions = run(engine, [make_hand(SHAKA)] * seconds(Config().screenshot_hold + 1.5))
    assert kinds(actions, Screenshot) == [Screenshot()]


def test_short_shaka_does_nothing(engine):
    assert not kinds(run(engine, [make_hand(SHAKA)] * seconds(0.4)), Screenshot)


def test_swipe_right_and_left(engine):
    settle(engine, PALM)
    right = run(engine, [make_hand(PALM, dx=i * 0.05) for i in range(6)], t0=1.0)
    assert kinds(right, Key) == [Key("alt+tab")]
    left = run(engine, [make_hand(PALM, dx=0.25 - i * 0.05) for i in range(6)], t0=3.0)
    assert kinds(left, Key) == [Key("alt+shift+tab")]


def test_slow_palm_movement_is_not_a_swipe(engine):
    settle(engine, PALM)
    assert not kinds(run(engine, [make_hand(PALM, dx=i * 0.003) for i in range(60)], t0=1.0), Key)


def test_swipe_cooldown(engine):
    settle(engine, PALM)
    frames = [make_hand(PALM, dx=i * 0.05) for i in range(6)] + [make_hand(PALM, dx=0.25 + i * 0.05) for i in range(6)]
    assert len(kinds(run(engine, frames, t0=1.0), Key)) == 1


# ---- two-hand zoom ----

def test_two_hand_pinch_zooms_without_clicking(engine):
    settle(engine)
    primary = [make_hand(pinch="index", x0=0.35 - i * 0.01) for i in range(15)]
    other = [make_hand(pinch="index", x0=0.65 + i * 0.01) for i in range(15)]
    actions = run(engine, primary + [make_hand(x0=0.2)], t0=1.0, second=other + [None])
    zoom = kinds(actions, Zoom)
    assert zoom and sum(z.steps for z in zoom) > 0
    assert not kinds(actions, Click, Press)


def test_hands_together_zoom_out(engine):
    settle(engine)
    primary = [make_hand(pinch="index", x0=0.3 + i * 0.01) for i in range(15)]
    other = [make_hand(pinch="index", x0=0.7 - i * 0.01) for i in range(15)]
    zoom = kinds(run(engine, primary, t0=1.0, second=other), Zoom)
    assert zoom and sum(z.steps for z in zoom) < 0


# ---- gesture mapping ----

def test_gesture_map_remaps_and_disables():
    gm = dict(DEFAULT_GESTURE_MAP, swipe_right="key:ctrl+tab", rock="none", two_fingers="volume")
    e = GestureEngine(Config(gesture_map=gm))
    settle(e, PALM)
    assert kinds(run(e, [make_hand(PALM, dx=i * 0.05) for i in range(6)], t0=1.0), Key) == [Key("ctrl+tab")]
    settle(e, ROCK, t0=3.0)
    assert not run(e, [make_hand(ROCK, dy=-i * 0.01) for i in range(15)], t0=4.0)
    settle(e, TWO, t0=6.0)
    assert kinds(run(e, [make_hand(TWO, dy=-i * 0.01) for i in range(15)], t0=7.0), Volume)


def test_calibrated_thresholds_apply(engine):
    engine.set_thresholds(pinch_on=0.01, pinch_off=0.02)  # synthetic pinch is now too loose
    settle(engine)
    assert non_moves(run(engine, [make_hand(pinch="index")] * 5 + [make_hand()], t0=1.0)) == []


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Config(pinch_on=0.5, pinch_off=0.4)
    with pytest.raises(ValueError):
        Config(hand="both")
    with pytest.raises(ValueError):
        Config(gesture_map={"wave": "toggle"})
    assert replace(Config(), dpi=1600).dpi == 1600


# ---- precision mode (thumb tucked) ----

def travel(engine, **kw):
    settle(engine, **kw)
    out = kinds(run(engine, [make_hand(dx=i * 0.005, **kw) for i in range(20)], t0=1.0), Move)
    return out[-1].x - out[0].x


def test_tucked_thumb_is_precision_mode_and_moves_slower():
    normal = travel(GestureEngine(Config()))
    precise_engine = GestureEngine(Config())
    precise = travel(precise_engine, tucked=True)
    assert precise_engine.mode is Mode.PRECISION
    assert 0 < precise < normal * 0.5


def test_tucked_thumb_near_curled_middle_finger_is_not_a_right_click(engine):
    settle(engine, tucked=True)
    actions = run(engine, [make_hand(tucked=True)] * 20, t0=1.0)
    assert not kinds(actions, Click) and engine.mode is Mode.PRECISION


def test_untucking_ends_precision_mode(engine):
    settle(engine, tucked=True)
    run(engine, [make_hand()] * 3, t0=1.0)
    assert engine.mode is Mode.MOVE and not engine.thumb_tucked


def test_precision_mode_is_mouse_mode_only():
    engine = GestureEngine(Config(pointer_mode="direct"))
    settle(engine, tucked=True)
    assert engine.mode is Mode.MOVE
