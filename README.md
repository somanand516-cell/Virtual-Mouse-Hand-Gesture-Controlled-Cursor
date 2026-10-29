# Gesture Mouse

Control your computer with hand gestures through a webcam: a smooth, low-latency virtual mouse
plus volume, brightness, window switching, zoom and screenshots. Python, MediaPipe and OpenCV.

## Setup

```bash
pip install -r requirements.txt
```

The hand model (~8 MB) downloads to `models/` on first run. `config.json` is created on first run.

## Run

From this folder:

```bash
python -m gesture_mouse
```

The first launch starts a short calibration: show an open palm (3 s), pinch thumb and index
5 times, then make a fist (3 s). Press **C** to skip it; it comes back on the next launch until a
calibration has been saved. Redo it any time with **C** or `--calibrate`.
In VS Code: open this folder and press **F5**.

Pause / resume gesture control: **F9** from anywhere (or **P** in the preview window).
Quit: **Q** in the preview window, or **Ctrl+Alt+Q** from anywhere (emergency stop).
Both global hotkeys are Windows-only; on macOS/Linux use the preview keys or Ctrl+C.

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Point (index up) | Move the cursor |
| ☝️ Point with the thumb tucked against the palm | Precision mode: slower cursor for small targets (Mouse mode) |
| 🤏 Pinch thumb + index | Left click. Two quick pinches = double click |
| 🤏 Pinch and hold (0.35 s), or pinch and move | Drag and drop |
| 🤏 Pinch thumb + middle | Right click |
| ✌️ Two fingers up, move up/down | Scroll (faster movement = faster scroll) |
| 🤟 Three fingers up, move up/down | Volume |
| 🤘 Rock (index + pinky), move up/down | Screen brightness |
| ✊ Fist | Freeze the cursor |
| ✋ Open palm swiped right / left | Alt+Tab / Alt+Shift+Tab |
| 🤙 Shaka (pinky up) held ~0.8 s | Screenshot to `Pictures/Gesture Mouse` |
| 🤏🤏 Pinch with both hands, move apart / together | Zoom in / out (Ctrl + wheel) |
| 🙈 Hide your hand | Stop. In Mouse mode: "lift the mouse" to reposition |

Every gesture is debounced: a hand shape must be stable for 0.1 s, a pinch for 0.05 s, and
one-shot gestures (swipe, screenshot) have a 0.8 s cooldown. All of them are remappable
in `config.json` (see `gesture_map` below).

## Keys (in the preview window)

| Key | Action |
|---|---|
| **+** / **-** | Sensitivity (DPI) up / down |
| **M** | Mouse mode (trackpad-like) / Direct mode (hand position = screen position) |
| **L** | Which hand controls the mouse: any / right / left |
| **C** | Calibrate to your hand (press again to skip) |
| **H** | Show / hide the gesture guide and meters |
| **P** / Space | Pause (also **F9** from anywhere; held buttons are released) |
| **Q** / Esc | Quit (also **Ctrl+Alt+Q** from anywhere) |

Change the global hotkeys with `pause_hotkey` and `quit_hotkey` in `config.json`.

The preview shows the current gesture and hand shape, mode, DPI, FPS and latency, a pinch meter,
which fingers are up (I M R P), and (in Direct mode) the active-area box. An orange warning
appears under the mode when the light is too low: the image is dark, or the camera has slowed
below 15 real fps to keep it bright.

## Tuning sensitivity (`config.json`)

Settings live in two files next to the project:

- **`config.json`: only your own changes.** Calibration, DPI, mode and hand are saved there
  automatically; add anything else you want to change. Restart the app after editing.
- **`config.defaults.json`: every setting with its default**, rewritten on each start. Copy a
  line from it into `config.json` to change that setting. (Editing the defaults file does nothing.)

Because `config.json` only holds what you changed, improved defaults in future versions still
reach you.

| What you want | Setting | Default |
|---|---|---|
| Cursor faster / slower overall | `dpi` (or + / - keys): 400 ... 3200 | 800 |
| More precise slow movement | lower `accel_min` | 0.5 |
| Bigger jumps on fast flicks | raise `accel_max` | 2.4 |
| Steadier when still (less jitter) | raise `precision_radius` (dead zone) | 0.0025 |
| Finer tiny movements | lower `precision_radius` | 0.0025 |
| Less smoothing lag | raise `filter_beta`, lower `cursor_glide` | 10, 0.025 |
| Smaller hand movement in Direct mode | raise `direct_gain_at_800` (smaller active box) | 1.8 |
| Clicks too easy / too hard | `pinch_on` / `pinch_off` (or press C) | 0.32 / 0.47 |
| Drag starts too soon | raise `drag_hold` | 0.35 s |
| Scroll / volume / brightness speed | `scroll_speed` / `volume_speed` / `brightness_speed` | |
| Swipes too easy / hard | `swipe_distance`, `swipe_time` | 0.18, 0.35 s |
| Faster camera, lower precision | `frame_width` / `frame_height` (e.g. 640 x 480) | 960 x 540 |
| Precision mode slower / faster | `precision_gain` | 0.35 |
| Precision mode triggers too easily / not at all | lower / raise `thumb_tuck_on` (and `thumb_tuck_off`) | 0.35 / 0.45 |
| Low-light warning too eager / too late | `low_light_level` (brightness 0-255), `low_fps_warning` | 60, 15 |
| Camera only ~10 fps in dim light | more light, or `low_light_mode`: `true` (experimental: 30 fps, noisier image) | false |
| Gestures resume sooner / later after you use the real mouse | `override_time` | 1.0 s |

`gesture_map` decides what each gesture does:

```json
"gesture_map": {
  "two_fingers": "scroll",            // continuous: "scroll", "volume", "brightness", "none"
  "three_fingers": "volume",
  "rock": "brightness",
  "fist": "freeze",                   // "freeze" or "none"
  "swipe_left": "key:alt+shift+tab",  // one-shot: "screenshot", "none", "key:<combo>"
  "swipe_right": "key:alt+tab",       // e.g. "key:ctrl+tab" to switch browser tabs instead
  "shaka_hold": "screenshot",         // e.g. "key:win+shift+s" for the Windows snipping tool
  "two_hand_pinch": "zoom"            // "zoom" or "none"
}
```

(JSON has no comments; they are only here for explanation.)

## How it works

```text
camera ──[latest-frame slot]──▶ tracking ──[action queue]──▶ output ──▶ Windows
                                   │  └──▶ system worker (brightness, screenshots)
                                   └──[latest snapshot]──▶ main thread (preview window, keys)
hotkey: Ctrl+Alt+Q kill switch
```

| Thread | Job |
|---|---|
| camera | Grabs frames; **skips repeated images** (some drivers repeat frames in dim light) |
| tracking | MediaPipe hand landmarks, then the gesture engine decides actions. Never touches the OS |
| output | The only code that moves the mouse or presses keys. 240 Hz cursor glide, keeps shortcut keys held without blocking anyone, multi-monitor |
| system | Slow OS work (brightness, screenshots) |
| main | Preview window and keys |

- **Frames may be dropped, actions never are.** Only the newest frame matters, so frames go
  through a one-item slot that overwrites. A lost "button up" would leave the mouse stuck down,
  so clicks and keys go through a real queue.
- **Real-mouse override:** touch the real mouse and gestures pause for a second, then continue
  from wherever you left the cursor. (Corners are no longer an emergency stop: Windows puts Start
  and Show Desktop there. Use **Ctrl+Alt+Q** instead.)
- **Watchdog:** if tracking ever freezes while a button or key is held, everything is released.
  Buttons are also released on any exit: quit, error, Ctrl+C, or closing the terminal window.
- **Logs:** `%LOCALAPPDATA%\GestureMouse\logs\gesture_mouse.log` (last few sessions kept).
  On exit the log shows real FPS and latency (p50 / p95).
- **Smoothing:** One Euro filter (steady when still, low lag when moving), then a 240 Hz glide.
- **Mouse mode:** relative movement with pointer acceleration and a "rubber-band" dead zone:
  tremor is ignored, but every deliberate movement registers fully.
- **Click accuracy:** the cursor brakes as fingers close, and a click lands where the cursor was
  just *before* the pinch started.
- **Scale invariance:** all finger and pinch measurements are 3D, aspect-corrected, and divided
  by hand size, so they work near or far from the camera.

### Libraries

| Library | Why |
|---|---|
| `mediapipe` | Google's hand tracking: 21 3D landmarks per hand, up to 2 hands |
| `opencv` (installed with mediapipe) | Camera capture and the preview window |
| `ctypes` (built into Python) | Calls the Windows API directly (`SetCursorPos`, `SendInput`, `RegisterHotKey`): faster than pyautogui, with no per-call pauses |
| `screen-brightness-control` (new) | Reads and sets laptop/monitor brightness. Optional: without it, only the brightness gesture is disabled |
| `Pillow` (new; usually already installed) | Takes screenshots (`ImageGrab`) |
| `pyautogui` | Only on macOS/Linux, as the mouse/keyboard fallback |

## Tests

```bash
pip install pytest
python -m pytest                              # unit tests
python tools/stress_test.py --trials 1000     # 1000 simulated users per gesture
python tools/precision_test.py                # jitter, fine control, click accuracy
python tools/endurance.py --seconds 120       # real camera: FPS, timing, memory
```

## Project layout

```text
virtual_mouse.py      shortcut: `python virtual_mouse.py` = `python -m gesture_mouse`
gesture_mouse/
  __main__.py         `python -m gesture_mouse`
  app.py              entry point: setup, preview window, command-line options
  pipeline.py         tracking thread: camera -> tracker -> gestures -> actions; keys; FPS/latency
  gestures.py         gesture engine: all gestures, debounce, cooldowns, pointer modes, DPI
  calibration.py      learns finger and pinch thresholds for your hand
  filters.py          One Euro filter
  overlay.py          preview drawing (HUD)
  core/
    config.py         all settings + config.json loading/saving
    actions.py        the actions the engine can ask for
    events.py         latest-only slot between threads
    metrics.py        FPS and latency measurement
    logging_setup.py  console + rotating log file
  vision/
    camera.py         picks the fastest camera backend, captures on a background thread
    tracker.py        MediaPipe hand landmarks (2 hands, left/right)
    pose.py           hand geometry: finger states, hand shapes, pinch distances
  output/
    controller.py     mouse/keyboard output thread, 240 Hz cursor glide, watchdog
    input_backend.py  Win32 (ctypes) or pyautogui backend
    monitors.py       multi-monitor layout
    system.py         brightness and screenshots on a worker thread
  infra/
    hotkeys.py        global hotkeys (F9 pause, Ctrl+Alt+Q quit)
    crash.py          releases buttons and keys on any exit
tests/                unit tests
tools/                stress, precision and endurance tests
```

## Troubleshooting

- **Gestures not recognized:** press **C** to calibrate. The log line `Calibration measurements`
  lists what it saw per step, including `thumb_tuck` (useful for tuning precision mode). Watch
  the finger boxes (I M R P) and the pinch meter in the preview to see what the app sees.
- **Cursor feels choppy:** check the log line `Performance: camera N real fps`. Many webcams
  drop to ~10 fps in dim light (some drivers hide this by repeating frames). Light your hand from
  the front, or try `"low_light_mode": true`.
- **"Could not read from camera":** close Zoom/Teams/Camera, check Windows Settings > Privacy >
  Camera, or set `camera_index` to 1.
- **The wrong hand controls the mouse:** press **L** to choose right or left hand.
- **Brightness gesture does nothing:** `pip install screen-brightness-control`. Some external
  monitors don't support software brightness.
- **`No module named ...` in VS Code:** Ctrl+Shift+P > *Python: Select Interpreter* and pick the
  Python you installed the requirements into.
- **Note on MediaPipe:** MediaPipe sends anonymous usage statistics to Google. If your network
  blocks that, its shutdown would hang for ~40 s; this app avoids that by not waiting for it.
