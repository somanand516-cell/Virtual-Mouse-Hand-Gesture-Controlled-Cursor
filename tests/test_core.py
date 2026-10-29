"""Thread hand-offs, metrics, monitor geometry, logging and crash cleanup."""

import logging
import threading
import time

import pytest

from gesture_mouse.core import logging_setup
from gesture_mouse.core.events import LatestSlot, SlotClosed
from gesture_mouse.core.metrics import Latency, Metrics, Rate
from gesture_mouse.infra import crash
from gesture_mouse.output.monitors import Rect, bounding, clamp_to_monitors, monitor_at


# ---- LatestSlot: the "queue that always drops old frames" ----

def test_slot_returns_only_the_newest_and_counts_drops():
    slot = LatestSlot()
    for i in range(5):
        slot.put(i)
    assert slot.get(0.1) == 4
    assert slot.dropped == 4
    assert slot.get(0.05) is None  # nothing new


def test_slot_wakes_a_waiting_consumer():
    slot, got = LatestSlot(), []
    t = threading.Thread(target=lambda: got.append(slot.get(1.0)))
    t.start()
    time.sleep(0.05)
    slot.put("frame")
    t.join(1.0)
    assert got == ["frame"]


def test_slot_close_unblocks_consumer():
    slot = LatestSlot()
    threading.Timer(0.05, slot.close).start()
    with pytest.raises(SlotClosed):
        slot.get(1.0)


# ---- metrics ----

def test_latency_percentiles():
    lat = Latency()
    for ms in range(1, 101):
        lat.add(ms)
    s = lat.stats()
    assert s.p50 == 51 and s.p95 == 96 and s.count == 100


def test_rate_from_average_interval():
    r = Rate(smoothing=0.5)
    for i in range(20):
        r.tick(i / 30)
    assert r.per_second == pytest.approx(30, rel=0.01)


def test_metrics_summary_mentions_everything():
    text = Metrics().summary()
    assert all(word in text for word in ("camera", "real fps", "tracking", "output", "latency", "repeated"))


# ---- monitors ----

SCREENS = [Rect(0, 0, 1920, 1080), Rect(1920, 200, 1280, 720)]


def test_bounding_box():
    assert bounding(SCREENS) == Rect(0, 0, 3200, 1080)


def test_monitor_at_and_clamp():
    assert monitor_at(SCREENS, 2000, 500) == SCREENS[1]
    assert monitor_at(SCREENS, 2500, 50) == SCREENS[1]       # in the gap above screen 2: nearest
    assert clamp_to_monitors(SCREENS, 2500, 50) == (2500, 200)
    assert clamp_to_monitors(SCREENS, -40, 500) == (0, 500)


# ---- logging and crash cleanup ----

def test_log_file_created(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = logging_setup.setup()
    logging.getLogger("gesture_mouse.test").info("hello from the test")
    for h in logging.getLogger().handlers:
        h.flush()
    assert path is not None and "hello from the test" in path.read_text(encoding="utf-8")
    for h in list(logging.getLogger().handlers):  # don't leak handlers into other tests
        logging.getLogger().removeHandler(h)
        h.close()


def test_cleanups_run_once_and_never_raise():
    calls = []
    crash.on_exit(lambda: calls.append("release"))
    crash.on_exit(lambda: 1 / 0)  # a failing cleanup must not stop the others
    crash.run_cleanups()
    crash.run_cleanups()
    assert calls == ["release"]


# ---- camera helpers ----

def test_brighten_lifts_dark_frames_only():
    import numpy as np

    from gesture_mouse.vision.camera import brighten

    dark = np.full((54, 96, 3), 22, np.uint8)
    normal = np.full((54, 96, 3), 110, np.uint8)
    assert 100 <= brighten(dark).mean() <= 120
    assert brighten(normal) is normal  # untouched, no copy


def test_repeated_frames_have_equal_signatures():
    import numpy as np

    from gesture_mouse.vision.camera import _signature

    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    b = a.copy()
    b[100:140, 200:260] += 1  # a hand moved a little
    assert np.array_equal(_signature(a), _signature(a.copy()))
    assert not np.array_equal(_signature(a), _signature(b))


def test_light_meter_warns_when_dark_or_slow_and_clears_with_margin():
    import numpy as np

    from gesture_mouse.vision.camera import LightMeter

    frame = lambda level: np.full((54, 96, 3), level, np.uint8)  # noqa: E731
    meter = LightMeter(dark_level=60, low_fps=15)
    assert meter.update(frame(20), 30, 0.0) == ""  # camera still adjusting: no judgement yet
    t = 0.0
    for _ in range(90):
        t += 1 / 30
        meter.update(frame(20), 30, t)
    assert meter.dark and "dark" in meter.warning
    for _ in range(90):
        t += 1 / 30
        meter.update(frame(65), 30, t)  # brighter, but inside the margin: warning stays
    assert meter.dark
    for _ in range(90):
        t += 1 / 30
        meter.update(frame(120), 30, t)
    assert meter.warning == ""
    meter.update(frame(120), 9, t + 0.1)  # bright image but only 9 real fps: dim light, auto exposure
    assert meter.slow and "slow" in meter.warning
