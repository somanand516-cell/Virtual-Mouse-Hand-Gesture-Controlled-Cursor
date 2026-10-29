"""Logging to the console and to a rotating file, plus logging of crashes in any thread.

The log file is what you attach to a bug report: it keeps the last few sessions
(%LOCALAPPDATA%\\GestureMouse\\logs on Windows, ~/.gesture_mouse/logs elsewhere).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(threadName)-9s %(name)s: %(message)s"


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "GestureMouse" / "logs" if base else Path.home() / ".gesture_mouse" / "logs"


def setup(verbose: bool = False) -> Path | None:
    """Console: short and readable. File: detailed, rotated at 1 MB, 3 files kept."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    console.setLevel(logging.INFO)
    # Our own debug messages only when asked; third-party libraries stay at WARNING.
    console.addFilter(lambda r: r.levelno >= logging.INFO or (verbose and r.name.startswith("gesture_mouse")))
    if verbose:
        console.setLevel(logging.DEBUG)
    root.addHandler(console)

    path: Path | None = None
    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "gesture_mouse.log"
        file = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file.setFormatter(logging.Formatter(LOG_FORMAT))
        file.setLevel(logging.DEBUG)
        file.addFilter(lambda r: r.levelno >= logging.WARNING or r.name.startswith("gesture_mouse"))
        root.addHandler(file)
    except OSError as e:
        logging.getLogger(__name__).warning("No log file (%s); logging to the console only.", e)

    # Uncaught errors anywhere end up in the log instead of vanishing with a dead thread.
    def thread_hook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("gesture_mouse").critical(
            "Uncaught error in thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    def main_hook(exc_type, exc, tb) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("gesture_mouse").critical("Uncaught error", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    threading.excepthook = thread_hook
    sys.excepthook = main_hook
    return path
