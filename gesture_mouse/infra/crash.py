"""Crash-safe shutdown: release held mouse buttons and keys however the app ends.

Normal quit and Python errors are covered by `finally` blocks and `atexit`. Two exits are not:
closing the terminal window and logging off, where Windows ends the process without running
Python's cleanup. For those, Windows calls a console control handler first, which we use.
"""

from __future__ import annotations

import atexit
import logging
import sys
from collections.abc import Callable

log = logging.getLogger(__name__)

_cleanups: list[Callable[[], None]] = []
_handler = None  # keep a reference: Windows calls it later, the GC must not collect it


def on_exit(fn: Callable[[], None]) -> None:
    """Run `fn` at exit (once), whatever the reason."""
    _cleanups.append(fn)


def run_cleanups() -> None:
    while _cleanups:
        fn = _cleanups.pop()
        try:
            fn()
        except Exception as e:  # cleanup must never raise
            log.warning("Cleanup failed: %s", e)


def install() -> None:
    global _handler
    atexit.register(run_cleanups)
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def handle(event: int) -> bool:
        # 0 Ctrl+C, 1 Ctrl+Break: Python turns these into KeyboardInterrupt, let it.
        # 2 window closed, 5 log off, 6 shutdown: clean up now, Windows ends the process next.
        if event in (2, 5, 6):
            run_cleanups()
        return False  # let the default handling continue

    _handler = HANDLER(handle)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True)
