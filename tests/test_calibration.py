from gesture_mouse.calibration import (
    MIN_PINCHES, PINCH_STEP_TIMEOUT, PINCHES, STEP_SECONDS, STEPS, Calibration, compute, summary,
)
from gesture_mouse.vision.pose import HandPose


def poses(pinch, finger, n=30):
    return [HandPose(pinch_index=pinch + (i % 5) * 0.01, pinch_middle=1.0,
                     finger_ratios=(finger + (i % 5) * 0.01,) * 4, palm_x=0.5, palm_y=0.5, pinch_x=0, pinch_y=0)
            for i in range(n)]


def test_compute_places_thresholds_between_poses():
    r = compute({"open": poses(0.9, 1.5), "pinch": poses(0.1, 1.3), "fist": poses(0.7, 0.6)})
    assert r.ok
    t = r.thresholds
    assert 0.14 < t["pinch_on"] < t["pinch_off"] < 0.9
    assert 0.64 < t["finger_fold"] < t["finger_extend"] < 1.5


def test_compute_tolerates_transition_frames():
    pinch = poses(0.1, 1.3, 20) + poses(0.9, 1.5, 10)  # a third of the step was still an open hand
    fist = poses(0.7, 0.6, 20) + poses(0.9, 1.5, 10)
    assert compute({"open": poses(0.9, 1.5), "pinch": pinch, "fist": fist}).ok


def test_unclear_pinch_still_saves_fingers():
    r = compute({"open": poses(0.3, 1.5), "pinch": poses(0.3, 1.3), "fist": poses(0.7, 0.6)})
    assert r.ok and "pinch" in r.message and "finger_extend" in r.thresholds and "pinch_on" not in r.thresholds


def test_unclear_fist_still_saves_pinch():
    r = compute({"open": poses(0.9, 1.0), "pinch": poses(0.1, 1.0), "fist": poses(0.7, 1.0)})
    assert r.ok and "fist" in r.message and "pinch_on" in r.thresholds and "finger_extend" not in r.thresholds


def test_everything_unclear_fails():
    r = compute({"open": poses(0.3, 1.0), "pinch": poses(0.3, 1.0), "fist": poses(0.3, 1.0)})
    assert not r.ok


def test_real_trial_data_calibrates():
    # Medians measured on the real hand in the trial log: open pinch=0.51 fingers=0.98,
    # pinch pinch=0.35 fingers=1.30, fist pinch=0.62 fingers=0.65. The old rules rejected this.
    r = compute({"open": poses(0.49, 0.96), "pinch": poses(0.33, 1.28), "fist": poses(0.60, 0.63)})
    assert r.ok and r.message == "Calibrated to your hand!"
    t = r.thresholds
    assert 0.35 < t["pinch_on"] < t["pinch_off"] < 0.51
    assert 0.65 < t["finger_fold"] < t["finger_extend"] < 0.98


def test_compute_needs_enough_samples():
    assert not compute({"open": poses(0.9, 1.5, n=3), "pinch": poses(0.1, 1.3), "fist": poses(0.7, 0.6)}).ok


def test_summary():
    text = summary({"open": poses(0.9, 1.5), "pinch": [], "fist": []})
    assert text.startswith("open: n=30 pinch=0.92")


OPEN, PINCH, FIST = poses(0.9, 1.5, 1)[0], poses(0.1, 1.3, 1)[0], poses(0.7, 0.6, 1)[0]


def user(cal, frame):
    """A cooperative user: the pose each step asks for. In the pinch step they pinch for 6 frames,
    then open for 6 (0.2 s each at 30 fps)."""
    key = STEPS[min(cal.step_index, len(STEPS) - 1)].key
    if key == "pinch":
        return PINCH if frame % 12 < 6 else OPEN
    return {"open": OPEN, "fist": FIST}[key]


def drive(cal, t=0.0, until=60.0, pose=user):
    frame = 0
    while not cal.done and t < until:
        cal.update(pose(cal, frame), t)
        t += 1 / 30
        frame += 1
    return t


def test_calibration_waits_for_hand():
    cal = Calibration(0.0)
    t = 0.0
    while t < 20:  # no hand for 20 s: still on step 1, nothing failed
        cal.update(None, t)
        t += 1 / 30
    assert not cal.done and cal.step_index == 0 and "Show your hand" in cal.instruction
    drive(cal, t)
    assert cal.done and cal.result.ok


def test_calibration_walks_through_steps():
    cal = Calibration(0.0)
    t = drive(cal)
    assert cal.done and cal.result.ok and cal.result.message == "Calibrated to your hand!"
    assert cal.pinches == PINCHES and len(cal.samples["pinch"]) == PINCHES
    assert t < 2 * (STEP_SECONDS + 1) + PINCHES * 0.4 + 3  # the pinch step ends at the 5th pinch


def test_pinch_step_counts_each_pinch_once_and_keeps_its_tightest_moment():
    cal = Calibration(0.0)
    drive(cal, until=6.0, pose=lambda c, f: OPEN)  # through the open-palm step, no pinches yet
    assert STEPS[cal.step_index].key == "pinch"
    t, closing = 6.0, [poses(d, 1.3, 1)[0] for d in (0.6, 0.3, 0.15, 0.2, 0.4)]  # one pinch, slowly
    for pose in closing + [OPEN] * 3:
        cal.update(pose, t)
        t += 1 / 30
    assert cal.pinches == 1 and cal.samples["pinch"][0].pinch_index == 0.15
    assert "(1/5)" in cal.instruction and cal.progress(t) == 1 / PINCHES


def test_pinch_step_gives_up_after_timeout_but_keeps_fingers():
    cal = Calibration(0.0)
    never_opens = lambda c, f: PINCH if STEPS[min(c.step_index, 2)].key == "pinch" else user(c, f)  # noqa: E731
    t = drive(cal, pose=never_opens, until=90)
    assert cal.done and t > PINCH_STEP_TIMEOUT
    assert cal.result.ok and "pinches" in cal.result.message
    assert "finger_extend" in cal.result.thresholds and "pinch_on" not in cal.result.thresholds


def test_compute_needs_a_few_pinches():
    few = compute({"open": poses(0.9, 1.5), "pinch": poses(0.1, 1.3, MIN_PINCHES - 1), "fist": poses(0.7, 0.6)})
    assert few.ok and "pinch_on" not in few.thresholds
    enough = compute({"open": poses(0.9, 1.5), "pinch": poses(0.1, 1.3, MIN_PINCHES), "fist": poses(0.7, 0.6)})
    assert "pinch_on" in enough.thresholds
