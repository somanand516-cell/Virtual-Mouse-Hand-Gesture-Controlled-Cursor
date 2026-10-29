"""Slow OS actions (brightness, screenshots) on a background worker, so they never stall tracking.

Brightness uses the optional `screen-brightness-control` library (laptop screens and most
monitors). Screenshots use Pillow's ImageGrab (Pillow is installed with mediapipe).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from gesture_mouse.core.actions import Brightness, Screenshot

log = logging.getLogger(__name__)


class SystemActions:
    def __init__(self, screenshot_dir: str, notify: Callable[[str], None], dry_run: bool = False) -> None:
        self.screenshot_dir = Path(screenshot_dir)
        self.notify = notify
        self.dry_run = dry_run
        self._jobs: queue.Queue = queue.Queue()
        self._pending_brightness = 0
        self._com_ready = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="system", daemon=True)
        self._thread.start()

    def apply(self, action: Brightness | Screenshot) -> None:
        log.debug("%s", action)
        if isinstance(action, Brightness):
            with self._lock:  # coalesce: many small changes become one call
                first = self._pending_brightness == 0
                self._pending_brightness += action.delta
            if first:
                self._jobs.put("brightness")
        elif isinstance(action, Screenshot):
            self._jobs.put("screenshot")

    def close(self) -> None:
        self._running = False
        self._jobs.put(None)
        self._thread.join(timeout=2.0)

    # ---- worker ----

    def _worker(self) -> None:
        while self._running:
            job = self._jobs.get()
            if job is None:
                return
            try:
                if job == "brightness":
                    time.sleep(0.05)  # let a few more changes accumulate
                    with self._lock:
                        delta, self._pending_brightness = self._pending_brightness, 0
                    if delta:
                        self._brightness(delta)
                elif job == "screenshot":
                    self._screenshot()
            except Exception as e:  # never let an OS hiccup kill the worker
                log.warning("%s failed: %s", job, e)

    def _brightness(self, delta: int) -> None:
        if self.dry_run:
            self.notify(f"Brightness {delta:+d}% (dry run)")
            return
        try:
            import screen_brightness_control as sbc
        except ImportError:
            self.notify("Brightness needs: pip install screen-brightness-control")
            return
        if not self._com_ready:  # Windows talks to the display through COM, set up per thread
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except ImportError:
                pass
            self._com_ready = True
        current = sbc.get_brightness(display=0)[0]
        new = min(max(current + delta, 0), 100)
        sbc.set_brightness(new, display=0)
        self.notify(f"Brightness {new}%")

    def _screenshot(self) -> None:
        from PIL import ImageGrab

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / time.strftime("screenshot-%Y%m%d-%H%M%S.png")
        if not self.dry_run:
            ImageGrab.grab(all_screens=True).save(path)
        self.notify(f"Screenshot saved: {path.name}")
        log.info("Screenshot saved to %s", path)
