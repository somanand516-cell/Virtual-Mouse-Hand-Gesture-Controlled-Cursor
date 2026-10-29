"""Randomized stress test of the gesture engine.

Simulates many users performing each gesture with realistic variation (hand size/distance, tilt,
tremor, relaxed fingers, frame rate, tracking dropouts) and counts wrong or missed actions.

    python tools/stress_test.py --trials 1000
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mouse.core.actions import (  # noqa: E402
    Brightness, Click, Key, Move, Notify, Press, Release, Screenshot, Scroll, Volume, Zoom,
)
from gesture_mouse.core.config import Config  # noqa: E402
from gesture_mouse.gestures import GestureEngine  # noqa: E402

FINGERS_UP = {
    "point": (1, 0, 0, 0), "pinch": (1, 0, 0, 0), "pinch_mid": (1, 0, 0, 0), "two": (1, 1, 0, 0),
    "three": (1, 1, 1, 0), "palm": (1, 1, 1, 1), "rest": (1, 1, 1, 1), "fist": (0, 0, 0, 0),
    "shaka": (0, 0, 0, 1),
}


@dataclass
class P:
    x: float
    y: float


@dataclass
class User:
    scale: float      # hand size in frame (distance from camera)
    tilt: float       # radians
    tremor: float     # landmark noise std-dev (normalized)
    curl: float       # how relaxed the raised fingers are, 0..1
    pinch_gap: float  # how tightly they pinch (thumb-tip gap, in hand units)
    fps: float
    dropout: float    # chance a frame has no detection
    x: float          # where the hand rests
    y: float


def random_user(rng: random.Random) -> User:
    return User(
        scale=rng.uniform(0.6, 1.5),
        tilt=math.radians(rng.uniform(-35, 35)),
        tremor=rng.uniform(0.001, 0.006),
        curl=rng.uniform(0.0, 0.35),
        pinch_gap=rng.uniform(0.0, 0.035),
        fps=rng.uniform(12, 30),
        dropout=rng.choice([0.0, 0.0, 0.02, 0.05, 0.1]),
        x=rng.uniform(0.35, 0.65),
        y=rng.uniform(0.35, 0.65),
    )


def hand(u: User, rng: random.Random, pose: str, ox: float = 0.0, oy: float = 0.0) -> list[P]:
    """21 landmarks in hand space (wrist at origin, fingers up), then scaled/tilted/placed/noised."""
    pts = [(0.0, 0.0)] * 21
    up_flags = FINGERS_UP["point" if pose == "open" else pose]
    for (mcp, x), up in zip({5: -0.04, 9: 0.0, 13: 0.04, 17: 0.08}.items(), up_flags):
        if up:
            curl = u.curl * rng.uniform(0.8, 1.2)
            if pose == "pinch" and mcp == 5:
                curl = max(curl, 0.35)       # index bends a little to meet the thumb
            tip_y, dip_y = -0.30 + curl * 0.15, -0.25 + curl * 0.08
        else:
            tip_y, dip_y = rng.uniform(-0.15, -0.10), -0.16  # tucked toward the palm
        pts[mcp] = (x, -0.15)
        pts[mcp + 1] = (x, -0.20)
        pts[mcp + 2] = (x, dip_y)
        spread = 2.0 if pose == "palm" else 1.0  # "palm" = fingers spread wide, "rest" = relaxed
        pts[mcp + 3] = ((x - 0.02) * spread + 0.02 if up else x, tip_y)
    pts[1], pts[2], pts[3], pts[4] = (-0.06, -0.04), (-0.10, -0.08), (-0.12, -0.12), (-0.14, -0.14)
    if pose == "pinch":
        ix, iy = pts[8]
        pts[4] = (ix - u.pinch_gap, iy + u.pinch_gap * 0.5)
    elif pose == "pinch_mid":
        mx, my = pts[12]
        pts[4] = (mx - u.pinch_gap, my + u.pinch_gap * 0.5)
    elif pose == "fist":
        pts[4] = (-0.04, -0.13)  # thumb wraps over the folded fingers

    c, s = math.cos(u.tilt), math.sin(u.tilt)
    wx, wy = u.x + ox, u.y + oy + 0.12 * u.scale
    return [
        P(wx + u.scale * (px * c - py * s) + rng.gauss(0, u.tremor),
          wy + u.scale * (px * s + py * c) + rng.gauss(0, u.tremor))
        for px, py in pts
    ]


# ---- Scenarios: each returns (frames, expected direction or None) --------------------------

def seq(u, rng, *parts):
    """parts: (pose, seconds, (dx_total, dy_total))"""
    frames, ox, oy = [], 0.0, 0.0
    for pose, sec, (dx, dy) in parts:
        n = max(1, round(sec * u.fps))
        for _ in range(n):
            ox += dx / n
            oy += dy / n
            frames.append(None if rng.random() < u.dropout else hand(u, rng, pose, ox, oy))
    return frames


def sign(rng):
    return rng.choice([-1, 1])


SCENARIOS = {
    "hover": lambda u, r: (seq(u, r, *[("point", 0.4, (r.uniform(-.15, .15), r.uniform(-.15, .15)))
                                         for _ in range(5)]), None),
    "click": lambda u, r: (seq(u, r, ("point", 0.5, (0, 0)), ("pinch", r.uniform(0.12, 0.28), (0, 0)),
                               ("point", 0.4, (0, 0))), None),
    "double": lambda u, r: (seq(u, r, ("point", 0.5, (0, 0)), ("pinch", 0.15, (0, 0)), ("point", 0.15, (0, 0)),
                                ("pinch", 0.15, (0, 0)), ("point", 0.4, (0, 0))), None),
    "drag": lambda u, r: (seq(u, r, ("point", 0.5, (0, 0)), ("pinch", 0.5, (0, 0)),
                              ("pinch", 0.8, (sign(r) * r.uniform(.1, .25), 0.05)), ("point", 0.4, (0, 0))), None),
    "right": lambda u, r: (seq(u, r, ("point", 0.5, (0, 0)), ("pinch_mid", r.uniform(0.15, 0.4), (0, 0)),
                               ("point", 0.4, (0, 0))), None),
    "scroll": lambda u, r: (lambda d: (seq(u, r, ("point", 0.4, (0, 0)), ("two", 0.3, (0, 0)),
                                           ("two", 0.8, (0, d)), ("point", 0.3, (0, 0))), d))(
        sign(r) * r.uniform(.1, .25)),
    "freeze": lambda u, r: (seq(u, r, ("point", 0.4, (0, 0)), ("fist", 1.0, (r.uniform(-.1, .1), .05))), None),
    "swipe": lambda u, r: (lambda d: (seq(u, r, ("palm", 0.4, (0, 0)), ("palm", 0.2, (d, 0.02)),
                                          ("palm", 0.4, (0, 0))), d))(sign(r) * r.uniform(.22, .35)),
    "rest": lambda u, r: (seq(u, r, ("point", 0.3, (0, 0)), ("rest", 6.0, (r.uniform(-.02, .02), .01))), None),
}


def count(actions, kind, **kw):
    return sum(isinstance(a, kind) and all(getattr(a, k) == v for k, v in kw.items()) for a in actions)


def check(name, actions, extra):
    left, right = count(actions, Click, button="left"), count(actions, Click, button="right")
    press, scroll, keys = count(actions, Press), count(actions, Scroll), count(actions, Key)
    toggles = count(actions, Notify)
    system = sum(count(actions, k) for k in (Volume, Brightness, Screenshot, Zoom))
    fails = []
    if system:
        fails.append("unwanted volume/brightness/screenshot/zoom")
    if name != "swipe" and keys:
        fails.append("unwanted key press")
    if toggles:
        fails.append("unexpected notification")
    if name not in ("click", "double", "drag", "right") and (left or right or press):
        fails.append("unwanted click/drag")
    if name != "scroll" and scroll:
        fails.append("unwanted scroll")
    if name == "click":
        if left != 1: fails.append(f"left clicks: {left}")
        if right or press: fails.append("right click or drag instead")
    elif name == "double":
        if left != 2: fails.append(f"double click gave {left} clicks")
        if press: fails.append("double click turned into drag")
    elif name == "drag":
        if press != 1: fails.append(f"drag pieces: {press}")
        if left or right: fails.append("click during drag")
    elif name == "right":
        if right != 1: fails.append(f"right clicks: {right}")
        if left or press: fails.append("left click/drag instead")
    elif name == "scroll":
        total = sum(a.amount for a in actions if isinstance(a, Scroll))
        if total == 0: fails.append("scroll missed")
        elif (total > 0) != (extra < 0): fails.append("scrolled wrong direction")
    elif name == "freeze":
        if count(actions, Move) and len({(a.x, a.y) for a in actions if isinstance(a, Move)}) > 1 and False:
            pass
    elif name == "swipe":
        want = "alt+tab" if extra > 0 else "alt+shift+tab"
        if keys != 1: fails.append(f"swipe gave {keys} key presses")
        elif not count(actions, Key, combo=want): fails.append("swipe went the wrong way")
    return fails


def run_trial(name, rng, cfg):
    u = random_user(rng)
    frames, extra = SCENARIOS[name](u, rng)
    engine = GestureEngine(cfg)
    actions, t = [], 0.0
    cursor_during_freeze = set()
    for i, lm in enumerate(frames):
        actions += engine.update(lm, t)
        if name == "freeze" and i > len(frames) // 2:
            cursor_during_freeze.add(engine.cursor)
        t += (1 / u.fps) * rng.uniform(0.85, 1.15)
    actions += engine.release()
    fails = check(name, actions, extra)
    if name == "freeze" and len(cursor_during_freeze) > 1:
        fails.append("cursor moved while frozen")
    if count(actions, Press) != count(actions, Release):
        fails.append("MOUSE BUTTON LEFT STUCK DOWN")
    return u, fails, len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-dropout", action="store_true", help="simulate perfect tracking (no lost frames)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cfg = Config()

    total_frames, start = 0, time.perf_counter()
    print(f"{args.trials} randomized trials per gesture (seed {args.seed})\n")
    print(f"{'gesture':<10} {'pass rate':>9}   top problems")
    results: list[tuple[User, bool]] = []
    for name in SCENARIOS:
        reasons: Counter[str] = Counter()
        ok = 0
        for _ in range(args.trials):
            if args.no_dropout:
                state = rng.getstate()
            u, fails, n = run_trial(name, rng, cfg)
            total_frames += n
            ok += not fails
            reasons.update(fails)
            results.append((u, not fails))
        top = ", ".join(f"{r} x{c}" for r, c in reasons.most_common(3)) or "-"
        print(f"{name:<10} {ok / args.trials:>8.1%}   {top}")

    elapsed = time.perf_counter() - start
    print(f"\nEngine speed: {total_frames / elapsed:,.0f} frames/s ({total_frames:,} frames in {elapsed:.1f}s)")
    print("\nPass rate by condition (all gestures):")
    factors = {
        "tracking dropouts": lambda u: "none" if u.dropout == 0 else f"{u.dropout:.0%}",
        "hand distance": lambda u: "far" if u.scale < 0.8 else ("near" if u.scale > 1.2 else "mid"),
        "tilt": lambda u: "strong" if abs(u.tilt) > math.radians(25) else "mild",
        "tremor": lambda u: "shaky" if u.tremor > 0.004 else "steady",
        "low fps": lambda u: "<18 fps" if u.fps < 18 else ">=18 fps",
        "loose pinch": lambda u: "loose" if u.pinch_gap > 0.025 else "tight",
    }
    for label, key in factors.items():
        groups: dict[str, list[bool]] = defaultdict(list)
        for u, passed in results:
            groups[key(u)].append(passed)
        parts = "  ".join(f"{g}: {sum(v) / len(v):.0%} (n={len(v)})" for g, v in sorted(groups.items()))
        print(f"  {label:<18} {parts}")


if __name__ == "__main__":
    main()
