"""Global hotkeys (defaults: F9 = pause, Ctrl+Alt+Q = quit) that work even when the preview window
isn't focused.

Windows only, via the Win32 RegisterHotKey API through ctypes (standard library). On other systems
use the keys in the preview window, or Ctrl+C in the terminal.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable

from gesture_mouse.output.input_backend import parse_combo

log = logging.getLogger(__name__)

MODIFIERS = {0x11: 0x0002, 0x12: 0x0001, 0x10: 0x0004, 0x5B: 0x0008}  # ctrl, alt, shift, win
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


def _split(combo: str) -> tuple[int, int]:
    vks = parse_combo(combo)
    mods = sum(MODIFIERS[v] for v in vks if v in MODIFIERS)
    keys = [v for v in vks if v not in MODIFIERS]
    if len(keys) != 1:
        raise ValueError(f"A hotkey needs exactly one non-modifier key: {combo!r}")
    return mods | MOD_NOREPEAT, keys[0]


class GlobalHotkeys:
    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None:
        self.bindings = {combo: cb for combo, cb in bindings.items() if combo}
        self._thread_id: int | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if sys.platform != "win32" or not self.bindings:
            return
        parsed = [(_split(combo), combo, cb) for combo, cb in self.bindings.items()]  # validate first
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(parsed, ready), name="hotkeys", daemon=True)
        self._thread.start()
        ready.wait(1.0)

    def _run(self, parsed, ready: threading.Event) -> None:
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        callbacks = {}
        for hotkey_id, ((mods, vk), combo, cb) in enumerate(parsed, start=1):
            if user32.RegisterHotKey(None, hotkey_id, mods, vk):
                callbacks[hotkey_id] = cb
            else:
                log.warning("Could not register %s (another app may be using it).", combo.upper())
        ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in callbacks:
                    callbacks[msg.wParam]()
        finally:
            for hotkey_id in callbacks:
                user32.UnregisterHotKey(None, hotkey_id)

    def stop(self) -> None:
        if self._thread_id is not None and self._thread is not None:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=1.0)
