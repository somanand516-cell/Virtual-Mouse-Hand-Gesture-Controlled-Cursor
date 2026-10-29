"""Synthetic hand landmarks for tests: an upright hand with chosen fingers up."""

from __future__ import annotations

from dataclasses import dataclass

FRAME = 1 / 30

POINT = (1, 0, 0, 0)
TWO = (1, 1, 0, 0)
THREE = (1, 1, 1, 0)
ROCK = (1, 0, 0, 1)
SHAKA = (0, 0, 0, 1)
PALM = (1, 1, 1, 1)
FIST = (0, 0, 0, 0)


@dataclass
class P:
    x: float
    y: float
    z: float = 0.0


def make_hand(fingers=POINT, pinch=None, dx=0.0, dy=0.0, x0=0.5, spread=1.0, tucked=False):
    """21 landmarks. fingers: (index, middle, ring, pinky) up flags. pinch: None, "index" or "middle".
    spread: how far raised fingertips fan out (1 = relaxed, 2 = spread wide).
    tucked: thumb tip folded onto the palm, next to the curled middle finger (precision mode)."""
    cy = 0.5
    lm = [P(x0, 0.7)] + [P(0.0, 0.0)] * 20  # wrist
    for (mcp, x), up in zip({5: -0.04, 9: 0.0, 13: 0.04, 17: 0.08}.items(), fingers):
        x += x0
        lm[mcp] = P(x, cy + 0.05)
        lm[mcp + 1] = P(x, cy)                                # PIP
        lm[mcp + 2] = P(x, cy - 0.05 if up else cy + 0.03)    # DIP
        tip_x = x0 + (x - x0 - 0.02) * spread + 0.02 if up else x
        lm[mcp + 3] = P(tip_x, cy - 0.10 if up else cy + 0.08)  # tip
    lm[1], lm[2], lm[3], lm[4] = (P(x0 - 0.06, 0.66), P(x0 - 0.10, 0.62), P(x0 - 0.12, 0.58), P(x0 - 0.14, 0.56))
    if pinch == "index":
        lm[4] = P(lm[8].x - 0.01, lm[8].y)
    elif pinch == "middle":
        # The middle finger bends to meet the thumb in front of the palm (not curled into it).
        lm[12] = P(x0 - 0.01, cy + 0.01)
        lm[4] = P(lm[12].x - 0.01, lm[12].y)
    if tucked:
        lm[4] = P(x0 + 0.02, 0.60)
    return [P(p.x + dx, p.y + dy, p.z) for p in lm]


def run(engine, frames, t0=0.0, second=None):
    """Feed landmark sets (or None) one frame apart; return all actions.
    `second`: optional list of second-hand landmark sets, same length as frames."""
    actions, t = [], t0
    for i, lm in enumerate(frames):
        actions += engine.update(lm, t, second[i] if second else None)
        t += FRAME
    return actions


def seconds(s: float) -> int:
    return round(s / FRAME)
