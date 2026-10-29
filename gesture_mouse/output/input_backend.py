"""Low-level mouse and keyboard output.

On Windows this calls the Win32 API directly through `ctypes` (part of Python's standard library):
SetCursorPos / SendInput are what pyautogui uses underneath, minus its per-call overhead, safety
pauses and checks. Elsewhere it falls back to pyautogui.

Both backends have the same small interface:
  monitors() position() move(x, y) button(name, down) click(name) scroll(amount) hscroll(amount)
  keys(combo, down) hold_key(name, down)
Keys are pressed and released in separate calls so the caller can hold them briefly without
blocking (the output thread schedules the release instead of sleeping).
"""

from __future__ import annotations

import sys

from gesture_mouse.output.monitors import Rect

# Virtual-key codes for key combos in config.json, e.g. "ctrl+shift+t", "alt+tab", "win+d".
VK = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B, "cmd": 0x5B,
    "tab": 0x09, "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "plus": 0xBB, "=": 0xBB, "minus": 0xBD, "-": 0xBD,
    "volumeup": 0xAF, "volumedown": 0xAE, "volumemute": 0xAD,
    "playpause": 0xB3, "nexttrack": 0xB0, "prevtrack": 0xB1,
    **{chr(c): ord(chr(c).upper()) for c in range(ord("a"), ord("z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"f{n}": 0x6F + n for n in range(1, 13)},
}
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB3}


def parse_combo(combo: str) -> list[int]:
    keys = [k.strip().lower() for k in combo.split("+") if k.strip()]
    if combo.strip().endswith("++"):  # "ctrl++" means ctrl + the plus key
        keys.append("plus")
    try:
        return [VK[k] for k in keys]
    except KeyError as e:
        raise ValueError(f"Unknown key {e.args[0]!r} in {combo!r}") from None


class Win32Input:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        try:  # use real pixels on scaled (e.g. 125%) displays
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        self._user32 = ctypes.windll.user32

        ULONG_PTR = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._MOUSEINPUT, self._KEYBDINPUT, self._INPUT = MOUSEINPUT, KEYBDINPUT, INPUT
        self._user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self._wintypes = wintypes

    # ---- helpers ----

    def _send(self, *inputs) -> None:
        arr = (self._INPUT * len(inputs))(*inputs)
        self._user32.SendInput(len(inputs), arr, self._ct.sizeof(self._INPUT))

    def _mouse(self, flags: int, data: int = 0):
        inp = self._INPUT(type=0)
        inp.u.mi = self._MOUSEINPUT(0, 0, data & 0xFFFFFFFF, flags, 0, 0)
        return inp

    def _key(self, vk: int, up: bool):
        flags = (0x0002 if up else 0) | (0x0001 if vk in EXTENDED else 0)
        inp = self._INPUT(type=1)
        inp.u.ki = self._KEYBDINPUT(vk, 0, flags, 0, 0)
        return inp

    # ---- interface ----

    def monitors(self) -> list[Rect]:
        """Every monitor's rectangle on the virtual desktop, in physical pixels."""
        ct = self._ct
        from ctypes import wintypes

        rects: list[Rect] = []

        class MONITORINFO(ct.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        def found(hmonitor, _hdc, _rect, _data):
            info = MONITORINFO()
            info.cbSize = ct.sizeof(MONITORINFO)
            if self._user32.GetMonitorInfoW(hmonitor, ct.byref(info)):
                r = info.rcMonitor
                rects.append(Rect(r.left, r.top, r.right - r.left, r.bottom - r.top))
            return True

        proc = ct.WINFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(wintypes.RECT), ct.c_void_p)
        self._user32.EnumDisplayMonitors(None, None, proc(found), 0)
        if not rects:  # very unusual; fall back to the primary screen
            rects.append(Rect(0, 0, self._user32.GetSystemMetrics(0), self._user32.GetSystemMetrics(1)))
        return rects

    def position(self) -> tuple[int, int]:
        point = self._wintypes.POINT()  # a fresh buffer per call: several threads read the cursor
        self._user32.GetCursorPos(self._ct.byref(point))
        return point.x, point.y

    def move(self, x: int, y: int) -> None:
        self._user32.SetCursorPos(int(x), int(y))

    def button(self, name: str, down: bool) -> None:
        flags = {("left", True): 0x0002, ("left", False): 0x0004,
                 ("right", True): 0x0008, ("right", False): 0x0010}[(name, down)]
        self._send(self._mouse(flags))

    def click(self, name: str) -> None:
        down = {"left": 0x0002, "right": 0x0008}[name]
        self._send(self._mouse(down), self._mouse(down * 2))  # down + up in one call

    def scroll(self, amount: int) -> None:
        self._send(self._mouse(0x0800, amount))  # MOUSEEVENTF_WHEEL

    def hscroll(self, amount: int) -> None:
        self._send(self._mouse(0x1000, amount))  # MOUSEEVENTF_HWHEEL, positive = right

    def keys(self, combo: str, down: bool) -> None:
        """Press (down=True) all keys of a combo in order, or release them in reverse order."""
        vks = parse_combo(combo)
        self._send(*[self._key(vk, not down) for vk in (vks if down else reversed(vks))])

    def hold_key(self, name: str, down: bool) -> None:
        self._send(self._key(VK[name], not down))


class PyAutoGuiInput:
    """Fallback for macOS / Linux."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = False  # corners are ordinary places to click; we use a kill-switch hotkey
        pyautogui.PAUSE = 0         # no built-in delay after every call
        self._g = pyautogui

    def monitors(self) -> list[Rect]:
        w, h = self._g.size()
        return [Rect(0, 0, w, h)]  # pyautogui only knows the primary screen

    def position(self) -> tuple[int, int]:
        return tuple(self._g.position())  # type: ignore[return-value]

    def move(self, x: int, y: int) -> None:
        self._g.moveTo(x, y, _pause=False)

    def button(self, name: str, down: bool) -> None:
        (self._g.mouseDown if down else self._g.mouseUp)(button=name, _pause=False)

    def click(self, name: str) -> None:
        self._g.click(button=name, _pause=False)

    def scroll(self, amount: int) -> None:
        self._g.scroll(max(1, abs(amount) // 40) * (1 if amount > 0 else -1), _pause=False)

    def hscroll(self, amount: int) -> None:
        self._g.hscroll(max(1, abs(amount) // 40) * (1 if amount > 0 else -1), _pause=False)

    def keys(self, combo: str, down: bool) -> None:
        parse_combo(combo)  # validate
        names = [k.strip() for k in combo.split("+")]
        for name in names if down else reversed(names):
            (self._g.keyDown if down else self._g.keyUp)(name, _pause=False)

    def hold_key(self, name: str, down: bool) -> None:
        (self._g.keyDown if down else self._g.keyUp)(name, _pause=False)


def create():
    return Win32Input() if sys.platform == "win32" else PyAutoGuiInput()
