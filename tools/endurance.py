"""Real-camera endurance test: runs the full pipeline (without moving the mouse) and reports
frame rate, per-stage timing, memory growth and camera errors.

    python tools/endurance.py --seconds 120
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from gesture_mouse.vision.camera import Camera  # noqa: E402
from gesture_mouse.core.config import Config  # noqa: E402
from gesture_mouse.gestures import GestureEngine  # noqa: E402
from gesture_mouse.vision.tracker import HandTracker  # noqa: E402


def memory_mb() -> float:
    if sys.platform != "win32":
        return float("nan")

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_info(get_process(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6


def pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    args = ap.parse_args()

    cfg = Config()
    engine = GestureEngine(cfg, aspect=cfg.frame_width / cfg.frame_height)
    read_ms, detect_ms, frame_ms = [], [], []
    detected = frames = 0
    mem_samples: list[tuple[float, float]] = []

    with Camera(cfg.camera_index, cfg.frame_width, cfg.frame_height) as cam, \
            HandTracker(cfg.model_path, cfg.detection_confidence, cfg.tracking_confidence) as tracker:
        start = last = time.perf_counter()
        next_mem = start
        while (now := time.perf_counter()) - start < args.seconds:
            t0 = time.perf_counter()
            frame = cv2.flip(cam.read(), 1)
            t1 = time.perf_counter()
            hands = tracker.detect_hands(frame, t1 - start)
            lm = hands[0].landmarks if hands else None
            t2 = time.perf_counter()
            engine.update(lm, t2, hands[1].landmarks if len(hands) > 1 else None)

            read_ms.append((t1 - t0) * 1000)
            detect_ms.append((t2 - t1) * 1000)
            frame_ms.append((now - last) * 1000)
            last = now
            frames += 1
            detected += lm is not None
            if now >= next_mem:
                mem_samples.append((now - start, memory_mb()))
                next_mem += 10

    elapsed = time.perf_counter() - start
    fps = [1000 / m for m in frame_ms[5:] if m > 0]
    print(f"Ran {elapsed:.0f}s, {frames} frames, hand visible in {detected / frames:.0%} of frames")
    print(f"FPS          avg {frames / elapsed:.1f}   worst 1% {pct(fps, 0.01):.1f}")
    print(f"Camera read  median {statistics.median(read_ms):.1f} ms   p99 {pct(read_ms, 0.99):.1f} ms")
    print(f"Hand detect  median {statistics.median(detect_ms):.1f} ms   p99 {pct(detect_ms, 0.99):.1f} ms")
    stalls = sum(m > 200 for m in frame_ms[5:])
    print(f"Stalls >200ms: {stalls}")
    print("Memory (MB):  " + "  ".join(f"{t:.0f}s={m:.0f}" for t, m in mem_samples))
    if len(mem_samples) >= 3:
        growth = mem_samples[-1][1] - mem_samples[1][1]
        print(f"Memory growth after warm-up: {growth:+.1f} MB")


if __name__ == "__main__":
    main()
