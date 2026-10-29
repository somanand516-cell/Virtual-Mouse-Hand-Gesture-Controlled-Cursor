"""Actions the gesture engine asks for. The pipeline sends each one to the mouse, keyboard or OS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    x: float  # normalized screen position, 0..1
    y: float


@dataclass(frozen=True)
class Click:
    button: str  # "left" | "right"


@dataclass(frozen=True)
class Press:      # left button down (start of a drag)
    pass


@dataclass(frozen=True)
class Release:    # left button up (end of a drag)
    pass


@dataclass(frozen=True)
class Scroll:
    amount: int   # wheel units, positive = up (120 = one notch)


@dataclass(frozen=True)
class HScroll:
    amount: int   # horizontal wheel units, positive = right


@dataclass(frozen=True)
class Zoom:
    steps: int    # +1 = zoom in one step (Ctrl + wheel up), -1 = zoom out


@dataclass(frozen=True)
class Key:
    combo: str    # e.g. "alt+tab", "ctrl+shift+t"


@dataclass(frozen=True)
class Volume:
    steps: int    # volume key presses; each is about 2%. Positive = louder


@dataclass(frozen=True)
class Brightness:
    delta: int    # percent, positive = brighter


@dataclass(frozen=True)
class Screenshot:
    pass


@dataclass(frozen=True)
class Notify:
    text: str     # shown briefly in the preview


Action = Move | Click | Press | Release | Scroll | HScroll | Zoom | Key | Volume | Brightness | Screenshot | Notify
