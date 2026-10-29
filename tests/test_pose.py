import pytest

from gesture_mouse.core.config import Config
from gesture_mouse.vision.pose import FingerTracker, Shape, analyze, shape_of
from hands import FIST, PALM, POINT, ROCK, SHAKA, THREE, TWO, P, make_hand


def classify(fingers):
    cfg = Config()
    ft = FingerTracker(cfg.finger_extend, cfg.finger_fold)
    return shape_of(ft.update(analyze(make_hand(fingers)).finger_ratios))


@pytest.mark.parametrize("fingers,shape", [
    (POINT, Shape.POINT), (TWO, Shape.TWO), (THREE, Shape.THREE), (ROCK, Shape.ROCK),
    (SHAKA, Shape.SHAKA), (PALM, Shape.PALM), (FIST, Shape.FIST), ((0, 1, 0, 0), Shape.OTHER),
])
def test_shapes(fingers, shape):
    assert classify(fingers) is shape


def test_finger_hysteresis_keeps_state_when_half_bent():
    ft = FingerTracker(extend=1.12, fold=0.95)
    assert ft.update([1.4, 0.6, 0.6, 0.6]) == (True, False, False, False)
    assert ft.update([1.0, 1.0, 1.0, 1.0]) == (True, False, False, False)  # in between: unchanged
    assert ft.update([0.9, 1.2, 0.6, 0.6]) == (False, True, False, False)


def test_pinch_distances():
    cfg = Config()
    assert analyze(make_hand(pinch="index")).pinch_index < cfg.pinch_on
    assert analyze(make_hand(pinch="middle")).pinch_middle < cfg.pinch_on
    open_ = analyze(make_hand(PALM))
    assert open_.pinch_index > cfg.pinch_off and open_.pinch_middle > cfg.pinch_off


def test_palm_does_not_move_when_pinching():
    a, b = analyze(make_hand()), analyze(make_hand(pinch="index"))
    assert (a.palm_x, a.palm_y) == (b.palm_x, b.palm_y)


def test_aspect_ratio_corrects_horizontal_distances():
    assert analyze(make_hand(pinch="index"), aspect=16 / 9).pinch_index > analyze(make_hand(pinch="index")).pinch_index


def test_depth_is_used():
    flat = make_hand(pinch="index")
    deep = [P(p.x, p.y, 0.1 if i == 4 else 0.0) for i, p in enumerate(flat)]  # thumb in front of index
    assert analyze(deep).pinch_index > analyze(flat).pinch_index


def test_same_result_near_and_far():
    near, far = make_hand(PALM), [P(0.5 + (p.x - 0.5) * 0.5, 0.5 + (p.y - 0.5) * 0.5) for p in make_hand(PALM)]
    a, b = analyze(near), analyze(far)
    assert a.pinch_index == pytest.approx(b.pinch_index)
    assert a.finger_ratios == pytest.approx(b.finger_ratios)


def test_thumb_tuck():
    cfg = Config()
    assert analyze(make_hand(tucked=True)).thumb_tuck < cfg.thumb_tuck_on
    for hand in (make_hand(), make_hand(PALM), make_hand(pinch="index"), make_hand(pinch="middle")):
        assert analyze(hand).thumb_tuck > cfg.thumb_tuck_off
