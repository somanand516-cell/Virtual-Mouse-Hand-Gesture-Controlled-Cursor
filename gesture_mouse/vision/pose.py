"""Hand geometry: finger states, hand shape and pinch distances from 21 MediaPipe landmarks.

All distances are 3D and aspect-corrected, then divided by the hand's own size, so results are
the same near or far from the camera and with the hand turned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGERS = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
FINGER_NAMES = ("index", "middle", "ring", "pinky")

# Averaging stable points cancels tracking noise. Fingers and thumb are excluded so that
# pinching or curling doesn't move the cursor.
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Point(Protocol):
    x: float
    y: float


Landmarks = Sequence[Point]


class Shape(Enum):
    """Hand shapes, by which fingers are up (index, middle, ring, pinky)."""
    POINT = "Point"          # index
    TWO = "Two fingers"      # index + middle
    THREE = "Three fingers"  # index + middle + ring
    ROCK = "Rock"            # index + pinky
    SHAKA = "Shaka"          # pinky only (thumb doesn't matter)
    PALM = "Open palm"       # all four
    FIST = "Fist"            # none
    OTHER = "Other"


SHAPES = {
    (True, False, False, False): Shape.POINT,
    (True, True, False, False): Shape.TWO,
    (True, True, True, False): Shape.THREE,
    (True, False, False, True): Shape.ROCK,
    (False, False, False, True): Shape.SHAKA,
    (True, True, True, True): Shape.PALM,
    (False, False, False, False): Shape.FIST,
}


class _Geometry:
    """MediaPipe normalizes x by image width and y by height (z uses x's scale), so x and z are
    multiplied by the aspect ratio to put everything in the same units."""

    def __init__(self, lm: Landmarks, aspect: float, depth_weight: float) -> None:
        z_scale = aspect * depth_weight  # MediaPipe's depth estimate is noisier than x/y
        self.p = [(q.x * aspect, q.y, getattr(q, "z", 0.0) * z_scale) for q in lm]

    def dist(self, a: int, b: int) -> float:
        return math.dist(self.p[a], self.p[b])

    def dist_to_palm(self, a: int) -> float:
        n = len(PALM_POINTS)
        centre = tuple(sum(self.p[i][k] for i in PALM_POINTS) / n for k in range(3))
        return math.dist(self.p[a], centre)


@dataclass(frozen=True)
class HandPose:
    pinch_index: float                          # thumb-index tip distance / hand size
    pinch_middle: float                         # thumb-middle tip distance / hand size
    finger_ratios: tuple[float, float, float, float]  # tip-wrist / knuckle-wrist (big = finger up)
    palm_x: float                               # palm centre, camera units (x scaled by aspect)
    palm_y: float
    pinch_x: float                              # point between thumb and index tips, camera units
    pinch_y: float
    spread: float = 1.0                         # index tip to pinky tip / hand size (big = fingers spread)
    thumb_tuck: float = 1.0                     # thumb tip to palm centre / hand size (small = thumb tucked)

    @property
    def pinch(self) -> float:  # backwards-compatible name
        return self.pinch_index


def palm_center(lm: Landmarks) -> tuple[float, float]:
    """Centre of the palm in normalized image coordinates."""
    n = len(PALM_POINTS)
    return sum(lm[i].x for i in PALM_POINTS) / n, sum(lm[i].y for i in PALM_POINTS) / n


def analyze(lm: Landmarks, aspect: float = 1.0, depth_weight: float = 0.5) -> HandPose:
    g = _Geometry(lm, aspect, depth_weight)
    hand_size = max(g.dist(WRIST, MIDDLE_MCP), 1e-6)
    ratios = tuple(g.dist(WRIST, tip) / max(g.dist(WRIST, pip), 1e-6) for tip, pip in FINGERS)
    px, py = palm_center(lm)
    return HandPose(
        pinch_index=g.dist(THUMB_TIP, INDEX_TIP) / hand_size,
        pinch_middle=g.dist(THUMB_TIP, MIDDLE_TIP) / hand_size,
        finger_ratios=ratios,  # type: ignore[arg-type]
        palm_x=px * aspect,
        palm_y=py,
        pinch_x=(lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2 * aspect,
        pinch_y=(lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2,
        spread=g.dist(INDEX_TIP, PINKY_TIP) / hand_size,
        thumb_tuck=g.dist_to_palm(THUMB_TIP) / hand_size,
    )


class FingerTracker:
    """Per-finger up/down state with hysteresis, so a half-bent finger doesn't flicker."""

    def __init__(self, extend: float, fold: float) -> None:
        self.extend, self.fold = extend, fold
        self.up: list[bool] | None = None

    def reset(self) -> None:
        self.up = None

    def update(self, ratios: Sequence[float]) -> tuple[bool, ...]:
        if self.up is None:
            mid = (self.extend + self.fold) / 2
            self.up = [r > mid for r in ratios]
        else:
            for i, r in enumerate(ratios):
                if r > self.extend:
                    self.up[i] = True
                elif r < self.fold:
                    self.up[i] = False
        return tuple(self.up)


def shape_of(up: Sequence[bool]) -> Shape:
    return SHAPES.get(tuple(up), Shape.OTHER)  # type: ignore[arg-type]
