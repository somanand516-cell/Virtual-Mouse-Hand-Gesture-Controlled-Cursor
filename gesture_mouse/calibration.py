"""Guided calibration: learns finger and pinch thresholds for the user's own hand and camera.

Steps: open palm (3 s) -> 5 pinches -> fist (3 s). Thresholds are placed between what the
user's hand actually looks like in each pose, so recognition fits them.

Counting pinches: the pinch thresholds are what is being measured, so they can't be used yet.
Instead each pinch is judged against the user's own open palm from step 1: a pinch starts when
thumb and index get much closer than in the open palm, and ends when they open up again. The
tightest moment of each pinch is kept, which is what a real click looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gesture_mouse.vision.pose import HandPose

STEP_SECONDS = 3.0         # length of the timed steps (open palm, fist)
MIN_SAMPLES = 15           # frames needed from each timed step
PINCHES = 5                # pinches asked for in the pinch step
MIN_PINCHES = 3            # with fewer, the pinch thresholds are left unchanged
PINCH_STEP_TIMEOUT = 20.0  # seconds (hand visible) before the pinch step moves on anyway
PINCH_CLOSED = 0.75        # a pinch starts below this share of the open-palm thumb-index distance...
PINCH_OPENED = 0.85        # ...and ends above this share


@dataclass(frozen=True)
class Step:
    key: str
    instruction: str
    timed: bool = True  # False: the step ends when its pinches are done


STEPS = (
    Step("open", "Show your OPEN PALM, fingers spread"),
    Step("pinch", f"PINCH thumb + index {PINCHES} times", timed=False),
    Step("fist", "Make a FIST and hold"),
)


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(len(s) * p)))]


@dataclass
class Result:
    ok: bool
    message: str
    thresholds: dict[str, float] = field(default_factory=dict)


def summary(samples: dict[str, list[HandPose]]) -> str:
    """Typical values per step, for the log (helps diagnose a failed calibration)."""
    parts = []
    for key, poses in samples.items():
        if poses:
            fingers = [r for p in poses for r in p.finger_ratios]
            parts.append(f"{key}: n={len(poses)} pinch={_pct([p.pinch_index for p in poses], 0.5):.2f} "
                         f"fingers={_pct(fingers, 0.5):.2f} spread={_pct([p.spread for p in poses], 0.5):.2f} "
                         f"thumb_tuck={_pct([p.thumb_tuck for p in poses], 0.5):.2f}")
    return "; ".join(parts)


def compute(samples: dict[str, list[HandPose]]) -> Result:
    open_, pinch, fist = samples["open"], samples["pinch"], samples["fist"]
    if min(len(open_), len(fist)) < MIN_SAMPLES:
        return Result(False, "Hand not seen clearly enough. Try again with more light.")

    # Each part is judged on its own, and whatever was measured clearly is kept, so one unclear
    # pose doesn't throw away the rest. Typical values (not extremes) are compared, because people
    # take a moment to form each pose.
    thresholds: dict[str, float] = {}
    problems: list[str] = []

    # Fingers: "up" ratios come from the open palm, "down" ratios from the fist.
    up = _pct([r for p in open_ for r in p.finger_ratios], 0.3)
    down = _pct([r for p in fist for r in p.finger_ratios], 0.6)
    if down < up * 0.9:
        span = up - down
        thresholds["finger_extend"] = round(down + 0.6 * span, 3)
        thresholds["finger_fold"] = round(down + 0.4 * span, 3)
    else:
        problems.append("fist looked like an open hand (curl all fingers in tightly)")

    # Pinch: between the user's typical pinch and their typical open hand.
    if len(pinch) < MIN_PINCHES:
        problems.append(f"only {len(pinch)} of {PINCHES} pinches seen (touch the tips, then open fully)")
    else:
        pinch_typ = _pct([p.pinch_index for p in pinch], 0.5)
        open_typ = _pct([p.pinch_index for p in open_], 0.5)
        if pinch_typ < open_typ * 0.8:
            gap = open_typ - pinch_typ
            thresholds["pinch_on"] = round(pinch_typ + 0.4 * gap, 3)
            thresholds["pinch_off"] = round(pinch_typ + 0.65 * gap, 3)
        else:
            problems.append("pinch looked like an open hand (touch thumb and index tips firmly)")

    if not thresholds:
        return Result(False, "Calibration unclear: " + "; ".join(problems))

    if problems:
        return Result(True, "Partly calibrated. Press C to redo: " + problems[0], thresholds)
    return Result(True, "Calibrated to your hand!", thresholds)


class Calibration:
    """Drive with `update(pose, t)` each frame; read `instruction` / `progress` for the overlay."""

    def __init__(self, t: float) -> None:
        self.samples: dict[str, list[HandPose]] = {s.key: [] for s in STEPS}
        self.step_index = 0
        self.step_start = t + 1.0  # one second to read the first instruction
        self.result: Result | None = None
        self.hand_visible = False
        self.pinches = 0
        self._tightest: HandPose | None = None  # the pinch in progress, at its closest
        self._open_pinch: float | None = None   # typical thumb-index distance of the open palm
        self._last_t = t

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def instruction(self) -> str:
        if self.result:
            return self.result.message
        if not self.hand_visible:
            return "Show your hand to the camera..."
        step = STEPS[self.step_index]
        text = f"{self.step_index + 1}/{len(STEPS)}  {step.instruction}"
        return text if step.timed else f"{text}  ({self.pinches}/{PINCHES})"

    def progress(self, t: float) -> float:
        if self.step_index < len(STEPS) and not STEPS[self.step_index].timed:
            return self.pinches / PINCHES
        return min(max((t - self.step_start) / STEP_SECONDS, 0.0), 1.0)

    def update(self, pose: HandPose | None, t: float) -> None:
        dt, self._last_t = t - self._last_t, t
        self.hand_visible = pose is not None
        if self.done:
            return
        if pose is None:
            # No hand: pause the clock, so a step only counts time while the hand is visible.
            self.step_start = max(self.step_start, t - STEP_SECONDS) + dt
            return
        if t < self.step_start:
            return
        step = STEPS[self.step_index]
        if step.timed:
            if t - self.step_start > 0.8:  # skip the moment of changing pose
                self.samples[step.key].append(pose)
            finished = t - self.step_start >= STEP_SECONDS
        else:
            self._count_pinch(pose)
            finished = self.pinches >= PINCHES or t - self.step_start >= PINCH_STEP_TIMEOUT
        if finished:
            self.step_index += 1
            self.step_start = t + 0.7  # short pause to read the next instruction
            if self.step_index == len(STEPS):
                self.result = compute(self.samples)

    def _count_pinch(self, pose: HandPose) -> None:
        """A small state machine: open -> closing (remember the tightest frame) -> open again = 1 pinch."""
        if self._open_pinch is None:
            open_ = self.samples["open"]
            self._open_pinch = _pct([p.pinch_index for p in open_], 0.5) if open_ else 1.0
        d = pose.pinch_index
        if self._tightest is None:
            if d < self._open_pinch * PINCH_CLOSED:
                self._tightest = pose
        elif d < self._tightest.pinch_index:
            self._tightest = pose
        elif d > self._open_pinch * PINCH_OPENED:
            self.samples["pinch"].append(self._tightest)
            self._tightest = None
            self.pinches += 1
