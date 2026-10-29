"""Output thread: smooth glide, real-mouse override, ordering, watchdog, multi-monitor."""

import threading
import time

import pytest

from gesture_mouse.core.actions import Click, HScroll, Key, Move, Press, Release, Scroll, Volume, Zoom
from gesture_mouse.core.metrics import Metrics
from gesture_mouse.output.controller import OutputController
from gesture_mouse.output.input_backend import VK, parse_combo
from gesture_mouse.output.monitors import Rect


class FakeIO:
    """Records everything; `pos` is the real cursor (tests move it to simulate the user's hand on the mouse)."""

    def __init__(self, monitors=None):
        self._monitors = monitors or [Rect(0, 0, 1920, 1080)]
        self.pos = (960, 540)
        self.moves: list[tuple[int, int]] = []
        self.events: list[tuple] = []
        self.lock = threading.Lock()

    def monitors(self):
        return self._monitors

    def position(self):
        return self.pos

    def move(self, x, y):
        with self.lock:
            self.pos = (x, y)
            self.moves.append((x, y))

    def click(self, name):
        self.events.append(("click", name, self.pos))

    def button(self, name, down):
        self.events.append(("down" if down else "up", name, self.pos))

    def scroll(self, amount):
        self.events.append(("scroll", amount))

    def hscroll(self, amount):
        self.events.append(("hscroll", amount))

    def keys(self, combo, down):
        self.events.append(("keys", combo, down, time.perf_counter()))

    def hold_key(self, name, down):
        self.events.append(("hold", name, down))


def make(io=None, **kw):
    io = io or FakeIO()
    kw.setdefault("glide_time", 0.02)
    return OutputController(backend=io, **kw), io


@pytest.fixture
def out():
    controller, io = make()
    yield controller, io
    controller.close()


def px(x, y, w=1920, h=1080):
    return round(x * (w - 1)), round(y * (h - 1))


# ---- smooth motion ----

def test_cursor_glides_in_small_steps(out):
    c, io = out
    c.apply(Move(0.5, 0.5))
    time.sleep(0.1)
    io.moves.clear()
    c.apply(Move(0.6, 0.5))  # ~190 px in one camera frame
    time.sleep(0.2)
    xs = [x for x, _ in io.moves]
    assert len(xs) >= 4 and max(b - a for a, b in zip(xs, xs[1:])) < 120
    assert xs[-1] == pytest.approx(0.6 * 1919, abs=2)


# ---- bug 1: snap-back after using the real mouse ----

def test_no_snap_back_after_real_mouse_use():
    c, io = make(override_time=0.15)
    try:
        c.apply(Move(0.2, 0.2))
        time.sleep(0.15)
        io.pos = (1500, 900)          # the user moves the REAL mouse far away
        time.sleep(0.05)
        assert c.overriding
        io.moves.clear()
        # The engine continues from the real cursor, so its next target is a small step from there.
        c.apply(Move(1530 / 1919, 900 / 1079))
        time.sleep(0.4)
        assert io.moves, "gesture movement should resume after the override"
        # Old bug: the first moves jumped back toward (384, 216). Now every move stays near (1500, 900).
        assert all(1500 <= x <= 1530 and abs(y - 900) <= 1 for x, y in io.moves), io.moves[:3]
    finally:
        c.close()


def test_real_mouse_detected_even_before_any_gesture_movement():
    c, io = make(override_time=0.3)
    try:
        time.sleep(0.05)  # the app hasn't moved the cursor yet
        io.pos = (100, 100)
        time.sleep(0.03)
        assert c.overriding
        c.apply(Move(110 / 1919, 100 / 1079))
        time.sleep(0.5)
        assert all(abs(x - 100) <= 10 for x, _ in io.moves), "must not jump back to the old (960, 540)"
    finally:
        c.close()


def test_gestures_pause_while_real_mouse_moves():
    c, io = make(override_time=0.3)
    try:
        c.apply(Move(0.5, 0.5))
        time.sleep(0.1)
        io.pos = (100, 100)
        time.sleep(0.03)
        io.moves.clear()
        c.apply(Move(0.9, 0.9))       # a gesture target arrives while the user is on the mouse
        time.sleep(0.1)
        assert io.moves == [] and c.overriding
    finally:
        c.close()


# ---- bug 2: corners are ordinary places ----

def test_corners_do_not_stop_the_app(out):
    c, io = out
    for corner in [(0, 0), (1919, 0), (0, 1079), (1919, 1079)]:  # Start button, Show desktop...
        io.pos = corner
        time.sleep(0.03)
    c.check()  # no exception
    c.apply(Move(0.0, 1.0))
    time.sleep(1.2)
    c.check()


# ---- bug 3: output never blocks tracking ----

def test_submit_returns_immediately_even_for_slow_actions(out):
    c, _ = out
    t = time.perf_counter()
    c.submit([Volume(10), Key("alt+tab"), Key("ctrl+c")], captured_at=t)
    assert time.perf_counter() - t < 0.005


def test_keys_are_held_briefly_then_released(out):
    c, io = out
    c.apply(Key("alt+tab"))
    time.sleep(0.15)
    keys = [e for e in io.events if e[0] == "keys"]
    assert [(e[1], e[2]) for e in keys] == [("alt+tab", True), ("alt+tab", False)]
    assert 0.02 <= keys[1][3] - keys[0][3] < 0.1


def test_volume_presses_each_step(out):
    c, io = out
    c.apply(Volume(-2))
    c.flush()
    assert [(e[1], e[2]) for e in io.events if e[0] == "keys"] == [("volumedown", True), ("volumedown", False)] * 2


# ---- ordering and never dropping ----

def test_actions_never_dropped_and_in_order(out):
    c, io = out
    for i in range(200):
        c.submit([Scroll(i)], captured_at=time.perf_counter())
    c.flush()
    assert [e[1] for e in io.events if e[0] == "scroll"] == list(range(200))


def test_press_lands_where_it_was_decided(out):
    c, io = out
    # One frame: move to A and press; the next frame already moves on to B.
    c.submit([Move(0.25, 0.25), Press()], captured_at=time.perf_counter())
    c.submit([Move(0.75, 0.75)], captured_at=time.perf_counter())
    c.flush()
    downs = [e for e in io.events if e[0] == "down"]
    assert downs and downs[0][2] == px(0.25, 0.25)


def test_other_actions(out):
    c, io = out
    for a in (Click("right"), HScroll(-120), Zoom(2), Release()):
        c.apply(a)
    c.flush()
    kinds = [e[0] for e in io.events]
    assert kinds == ["click", "hscroll", "hold", "scroll", "hold", "up"]
    assert ("scroll", 240) in io.events


# ---- watchdog and shutdown ----

def test_watchdog_releases_button_when_tracking_freezes():
    c, io = make(watchdog_timeout=0.2)
    try:
        c.heartbeat()
        c.apply(Press())
        time.sleep(0.1)
        assert [e[0] for e in io.events] == ["down"]
        time.sleep(0.4)  # no heartbeats: tracking "froze"
        assert io.events[-1][0] == "up"
    finally:
        c.close()


def test_watchdog_quiet_while_heartbeats_arrive():
    c, io = make(watchdog_timeout=0.2)
    try:
        c.apply(Press())
        for _ in range(10):
            c.heartbeat()
            time.sleep(0.05)
        assert [e[0] for e in io.events] == ["down"]
    finally:
        c.close()


def test_close_releases_held_button_and_keys():
    c, io = make()
    c.apply(Press())
    c.flush()
    c._held_keys.add("ctrl+shift")  # as if a shortcut were mid-hold
    c.close()
    assert ("up", "left", io.pos) in io.events
    assert any(e[0] == "keys" and e[1] == "ctrl+shift" and e[2] is False for e in io.events)


def test_dry_run_sends_nothing():
    io = FakeIO()
    c = OutputController(dry_run=True, backend=io)
    for a in (Move(0.2, 0.2), Press(), Release(), Click("right"), Key("alt+tab")):
        c.apply(a)
    time.sleep(0.1)
    c.close()
    assert io.moves == [] and io.events == []


def test_latency_metrics_recorded(out):
    c, _ = out
    m = c.metrics
    t = time.perf_counter()
    c.submit([Move(0.3, 0.3), Click("left")], captured_at=t - 0.02, decided_at=t)
    c.flush()
    assert m.queue_ms.stats().count >= 1 and m.total_ms.stats().p50 >= 20


# ---- multi-monitor ----

TWO_SCREENS = [Rect(0, 0, 1920, 1080), Rect(1920, 200, 1280, 720)]  # second one lower and smaller


def test_targets_span_all_monitors_and_avoid_gaps():
    c, io = make(FakeIO(TWO_SCREENS))
    try:
        assert (c.desktop.width, c.desktop.height) == (3200, 1080)
        c.apply(Move(1.0, 0.0))  # top-right of the bounding box: no monitor there (gap above screen 2)
        time.sleep(0.3)
        x, y = io.pos
        assert x == 3199 and y == 200  # clamped onto the second monitor's top edge
    finally:
        c.close()


def test_monitor_rect_normalized():
    c, io = make(FakeIO(TWO_SCREENS))
    try:
        io.pos = (2500, 500)
        left, top, right, bottom = c.monitor_rect_normalized()
        assert left == pytest.approx(1920 / 3199) and right == pytest.approx(1.0)
        assert top == pytest.approx(200 / 1079) and bottom == pytest.approx(919 / 1079)
    finally:
        c.close()


def test_parse_combo():
    assert parse_combo("alt+tab") == [VK["alt"], VK["tab"]]
    assert parse_combo("Ctrl + Shift + T") == [VK["ctrl"], VK["shift"], ord("T")]
    assert parse_combo("ctrl++") == [VK["ctrl"], VK["plus"]]
    with pytest.raises(ValueError):
        parse_combo("ctrl+banana")
