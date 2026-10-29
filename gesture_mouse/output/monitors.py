"""Multi-monitor geometry (pure functions, unit-tested).

Windows places all monitors on one "virtual desktop" coordinate plane. Monitors can have
different sizes and offsets, so the bounding box contains dead areas where no monitor exists.
Moving the cursor there makes Windows clamp it somewhere unexpected, which would look like the
user moved the mouse. So every target is first clamped onto a real monitor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:   # inclusive last pixel column
        return self.left + self.width - 1

    @property
    def bottom(self) -> int:  # inclusive last pixel row
        return self.top + self.height - 1

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return min(max(x, self.left), self.right), min(max(y, self.top), self.bottom)


def bounding(rects: list[Rect]) -> Rect:
    left, top = min(r.left for r in rects), min(r.top for r in rects)
    right, bottom = max(r.right for r in rects), max(r.bottom for r in rects)
    return Rect(left, top, right - left + 1, bottom - top + 1)


def monitor_at(monitors: list[Rect], x: float, y: float) -> Rect:
    """The monitor containing the point, or the nearest one."""
    for m in monitors:
        if m.contains(x, y):
            return m
    return min(monitors, key=lambda m: _dist2(m.clamp(x, y), (x, y)))


def clamp_to_monitors(monitors: list[Rect], x: float, y: float) -> tuple[float, float]:
    """Nearest point that is on an actual screen."""
    return monitor_at(monitors, x, y).clamp(x, y)


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
