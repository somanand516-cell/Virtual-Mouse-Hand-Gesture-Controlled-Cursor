"""Entry point: sets everything up, then runs the preview window on the main thread.

Threads and the hand-offs between them:

  camera ──[latest-frame slot]──▶ tracking ──[action queue]──▶ output ──▶ Windows
    vision/camera.py                pipeline.py                  output/controller.py
                                       │  └──▶ system worker (brightness, screenshots)
                                       └──[latest snapshot]──▶ main (preview window, keys)
  hotkeys: global pause (F9) and kill switch (infra/hotkeys.py)

Frames may be dropped (only the newest matters); actions never are (a lost "button up"
would leave the mouse stuck down).
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2

from gesture_mouse import overlay
from gesture_mouse.core import config as config_file
from gesture_mouse.core import logging_setup
from gesture_mouse.core.config import DPI_LEVELS, HAND_MODES, POINTER_MODES, Config
from gesture_mouse.core.metrics import Metrics
from gesture_mouse.gestures import GestureEngine
from gesture_mouse.infra import crash
from gesture_mouse.infra.hotkeys import GlobalHotkeys
from gesture_mouse.output.controller import OutputController
from gesture_mouse.output.system import SystemActions
from gesture_mouse.pipeline import Pipeline
from gesture_mouse.vision.camera import Camera, CameraError

log = logging.getLogger("gesture_mouse")

WINDOW = "Gesture Mouse"
PREVIEW_WIDTH = 640
KEY_ESC = 27


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="gesture_mouse", description="Control your mouse with hand gestures.")
    p.add_argument("--dpi", type=int, help=f"pointer sensitivity, e.g. {', '.join(map(str, DPI_LEVELS))}")
    p.add_argument("--mode", choices=POINTER_MODES, help="mouse: trackpad-like (default); direct: absolute")
    p.add_argument("--hand", choices=HAND_MODES, help="which hand controls the mouse (default: any)")
    p.add_argument("--calibrate", action="store_true",
                   help="start with hand calibration (automatic until a calibration is saved)")
    p.add_argument("--camera", type=int, help="webcam index (default: 0)")
    p.add_argument("--config", default=str(config_file.CONFIG_PATH), help="settings file (default: config.json)")
    p.add_argument("--dry-run", action="store_true", help="track and show gestures without controlling anything")
    p.add_argument("--duration", type=float, default=0, help="exit automatically after N seconds")
    p.add_argument("-v", "--verbose", action="store_true", help="log every action")
    return p.parse_args(argv)


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _open_window(screen_w: int, screen_h: int, size: tuple[int, int]) -> None:
    """Always-on-top preview parked in the bottom-right corner, out of the way."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    w, h = size
    cv2.moveWindow(WINDOW, max(0, screen_w - w - 16), max(0, screen_h - h - 90))  # clear of the taskbar


def run(cfg: Config, dry_run: bool = False, duration: float = 0, calibrate: bool = False) -> None:
    # Imported here so --help is instant and a broken MediaPipe install gets a clear error.
    from gesture_mouse.vision.tracker import HandTracker

    quit_requested = threading.Event()
    metrics = Metrics()
    mouse = OutputController(metrics, dry_run=dry_run, glide_time=cfg.cursor_glide,
                             override_time=cfg.override_time, watchdog_timeout=cfg.watchdog_timeout,
                             rate_hz=cfg.output_rate_hz)
    crash.on_exit(mouse.release_all)  # however the app ends, no button or key stays held
    if len(mouse.monitors) > 1:
        log.info("%d monitors: %s", len(mouse.monitors),
                 ", ".join(f"{m.width}x{m.height} at ({m.left},{m.top})" for m in mouse.monitors))
    pipeline: Pipeline | None = None
    system = SystemActions(cfg.screenshot_dir, notify=lambda text: pipeline and pipeline.notify(text),
                           dry_run=dry_run)
    hotkey = GlobalHotkeys({
        cfg.quit_hotkey: quit_requested.set,
        # Goes through the pipeline's key queue, so pausing (and releasing held buttons) happens
        # on the tracking thread exactly as if P had been pressed in the preview.
        cfg.pause_hotkey: lambda: pipeline and pipeline.command(ord("p")),
    })
    try:
        exposure = cfg.camera_exposure
        if cfg.low_light_mode and exposure is None:
            exposure = -5  # gave 30 real fps on the development laptop's webcam
        if cfg.low_light_mode:
            log.info("Low-light mode (experimental): fixed exposure %s + software brightening.", exposure)
        with Camera(cfg.camera_index, cfg.frame_width, cfg.frame_height, exposure, metrics) as cam, \
                HandTracker(cfg.model_path, cfg.detection_confidence, cfg.tracking_confidence) as tracker:
            engine = GestureEngine(cfg, aspect=cfg.frame_width / cfg.frame_height,
                                   screen=(mouse.screen_w, mouse.screen_h))
            pipeline = Pipeline(cfg, cam, tracker, engine, mouse, system, metrics)
            hotkey.start()
            preview_size = (PREVIEW_WIDTH, round(PREVIEW_WIDTH * cfg.frame_height / cfg.frame_width))
            primary = mouse.monitors[0]
            _open_window(primary.left + primary.width, primary.top + primary.height, preview_size)
            log.info("Running: %s mode, %d DPI, %s hand. Pause: %s anywhere. Quit: Q in the preview, "
                     "or %s anywhere.", engine.pointer_mode, engine.dpi, cfg.hand, cfg.pause_hotkey.upper(),
                     cfg.quit_hotkey.upper())
            pipeline.start(calibrate)
            try:
                _ui_loop(pipeline, preview_size, duration, quit_requested)
            finally:
                log.debug("Shutting down")
                pipeline.stop()  # must stop using the camera and tracker before they are closed
    finally:
        hotkey.stop()
        system.close()
        mouse.close()  # releases any held mouse button
        cv2.destroyAllWindows()
        log.debug("Threads stopped")


def _ui_loop(pipeline: Pipeline, preview_size: tuple[int, int], duration: float,
             quit_requested: threading.Event) -> None:
    start = time.monotonic()
    while not quit_requested.is_set():
        snap = pipeline.wait_snapshot(timeout=0.5)
        pipeline.check()  # re-raise tracking-thread errors (camera lost, output thread failure, ...)
        if snap is not None:
            preview = cv2.resize(snap.frame, preview_size, interpolation=cv2.INTER_AREA)
            overlay.draw(preview, snap.primary.landmarks if snap.primary else None, snap.hud,
                         snap.second.landmarks if snap.second else None)
            cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), KEY_ESC) or _window_closed():
            break
        if key != 0xFF:
            pipeline.command(key)
        if duration and time.monotonic() - start >= duration:
            break


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = logging_setup.setup(verbose=args.verbose)
    crash.install()
    if log_file:
        log.debug("Logging to %s", log_file)
    try:
        config_file.CONFIG_PATH = Path(args.config)
        cfg = config_file.load()
        overrides = {k: v for k, v in {"dpi": args.dpi, "pointer_mode": args.mode, "hand": args.hand,
                                       "camera_index": args.camera}.items() if v is not None}
        cfg = replace(cfg, **overrides)
        calibrate = args.calibrate
        if not calibrate and not config_file.has_calibration():
            log.info("Not calibrated yet: starting calibration (press C in the preview to skip).")
            calibrate = True
        run(cfg, dry_run=args.dry_run, duration=args.duration, calibrate=calibrate)
    except KeyboardInterrupt:
        pass
    except (CameraError, RuntimeError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Unexpected error (details in %s)", log_file)
        return 1
    finally:
        crash.run_cleanups()
    log.info("Bye!")
    return 0
