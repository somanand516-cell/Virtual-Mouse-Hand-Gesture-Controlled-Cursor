"""Precision benchmark for Mouse mode: jitter, fine control, click accuracy and reach.

Uses the same simulated hands as the stress test, with realistic MediaPipe landmark noise.

    python tools/precision_test.py
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stress_test import User, hand  # noqa: E402

from gesture_mouse.core.config import Config  # noqa: E402
from gesture_mouse.gestures import Click, GestureEngine, Move  # noqa: E402

SCREEN = (1920, 1080)
FPS = 30
TRIALS = 200


def user(rng: random.Random, tremor: float = 0.002) -> User:
    return User(scale=rng.uniform(0.8, 1.2), tilt=0.0, tremor=tremor, curl=0.1, pinch_gap=0.005,
                fps=FPS, dropout=0.0, x=0.5, y=0.5)


def run(frames, cfg: Config):
    """Return cursor positions in pixels (one per frame) and the actions."""
    e = GestureEngine(cfg, screen=SCREEN)
    pos, actions = [], []
    for i, lm in enumerate(frames):
        out = e.update(lm, i / FPS)
        actions += out
        pos.append((e.cursor[0] * SCREEN[0], e.cursor[1] * SCREEN[1]))
    return pos, actions


def rest_jitter(cfg, rng):
    """Hand held still for 3 s: how much does the cursor wander? (px, after 1 s settle)"""
    u = user(rng)
    pos, _ = run([hand(u, rng, "open") for _ in range(FPS * 3)], cfg)
    pos = pos[FPS:]
    cx, cy = statistics.fmean(p[0] for p in pos), statistics.fmean(p[1] for p in pos)
    return max(math.dist(p, (cx, cy)) for p in pos)


def fine_move(cfg, rng):
    """Very slow, deliberate move (0.012 camera units over 2 s, ~1/80 of the image).
    Returns (px travelled, worst single-frame jump px)."""
    u = user(rng)
    n = FPS * 2
    frames = [hand(u, rng, "open") for _ in range(FPS)]
    frames += [hand(u, rng, "open", ox=0.012 * i / n) for i in range(n)]
    pos, _ = run(frames, cfg)
    pos = pos[FPS:]
    jumps = [abs(b[0] - a[0]) for a, b in zip(pos, pos[1:])]
    return pos[-1][0] - pos[0][0], max(jumps)


def click_drift(cfg, rng):
    """Hand still, then fingers close into a pinch while the hand drifts slightly (as real hands do).
    Returns px between where the cursor was when the pinch started closing and where the click landed."""
    u = user(rng)
    frames = [hand(u, rng, "open") for _ in range(FPS)]
    closing = 8
    for i in range(closing):  # thumb approaches index; whole hand drifts 0.006 units
        k = (i + 1) / closing
        lm = hand(u, rng, "open", ox=0.006 * k)
        pinched = hand(u, rng, "pinch", ox=0.006 * k)
        lm[4] = type(lm[4])(lm[4].x + (pinched[4].x - lm[4].x) * k, lm[4].y + (pinched[4].y - lm[4].y) * k)
        frames.append(lm)
    frames += [hand(u, rng, "pinch", ox=0.006) for _ in range(4)] + [hand(u, rng, "open", ox=0.006)] * 2
    e = GestureEngine(cfg, screen=SCREEN)
    aim = clicked = None
    for i, lm in enumerate(frames):
        out = e.update(lm, i / FPS)
        if i == FPS - 1:
            aim = (e.cursor[0] * SCREEN[0], e.cursor[1] * SCREEN[1])
        if any(isinstance(a, Click) for a in out):
            clicked = (e.cursor[0] * SCREEN[0], e.cursor[1] * SCREEN[1])
    return math.dist(aim, clicked) if clicked else float("nan")


def reach(cfg, rng):
    """Quick flick: 0.15 camera units in 0.25 s. Returns px travelled."""
    u = user(rng)
    frames = [hand(u, rng, "open") for _ in range(10)]
    frames += [hand(u, rng, "open", ox=0.15 * i / 8) for i in range(1, 9)] + [hand(u, rng, "open", ox=0.15)] * 10
    e = GestureEngine(cfg, screen=SCREEN)
    e.set_cursor(0.0, 0.5)
    for i, lm in enumerate(frames):
        e.update(lm, i / FPS)
    return e.cursor[0] * SCREEN[0]


def main() -> None:
    cfg = Config()
    rng = random.Random(7)
    jit = [rest_jitter(cfg, rng) for _ in range(TRIALS)]
    fine = [fine_move(cfg, rng) for _ in range(TRIALS)]
    drift = [d for d in (click_drift(cfg, rng) for _ in range(TRIALS)) if not math.isnan(d)]
    far = [reach(cfg, rng) for _ in range(TRIALS)]
    moved = [f[0] for f in fine]
    print(f"Mouse mode at {cfg.dpi} DPI, {TRIALS} simulated trials each (screen {SCREEN[0]}x{SCREEN[1]})\n")
    print(f"Jitter at rest          avg {statistics.fmean(jit):5.1f} px   worst {max(jit):5.1f} px   (lower = steadier)")
    print(f"Fine move travel        avg {statistics.fmean(moved):5.1f} px   "
          f"stuck (<2px) {sum(m < 2 for m in moved) / TRIALS:4.0%}   (0% stuck = small moves always register)")
    print(f"Fine move worst jump    avg {statistics.fmean(f[1] for f in fine):5.1f} px   (lower = smoother)")
    print(f"Click drift             avg {statistics.fmean(drift):5.1f} px   worst {max(drift):5.1f} px   "
          f"(clicks registered {len(drift)}/{TRIALS})")
    print(f"Flick reach             avg {statistics.fmean(far):5.0f} px   (higher = less arm movement needed)")


if __name__ == "__main__":
    main()
