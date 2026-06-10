# Leather Defect Detection — Live GUI

PyQt5 desktop application for live leather-defect segmentation using a
trained Attention U-Net. Accepts webcam, image, or video input and
overlays per-pixel defect predictions in real time, with per-class area
statistics and a PASS/FAIL verdict.

## Defect classes

| Class      | Color    |
|------------|----------|
| background | black    |
| color      | red      |
| cut        | green    |
| fold       | blue     |
| glue       | yellow   |
| poke       | magenta  |

## Project structure

```
leather_defect_gui/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # entry point + CLI args
├── src/
│   ├── __init__.py
│   ├── detector.py          # Attention U-Net wrapper + preprocessing
│   ├── worker.py            # QThread: capture + inference
│   └── gui.py               # PyQt5 main window
├── models/                  # put your .keras model here
└── snapshots/               # default output for snapshots
```

## Installation

Requires Python 3.9 – 3.12.

```bash
# Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **TensorFlow / Keras version note.** The model was saved with Keras 3.x,
> so you need `tensorflow >= 2.16` (Keras 3 is bundled). If the model
> fails to load with layer-config errors, install the same TF version
> you used to train it.

## Providing the model

The app looks for a `.keras` model in this order — first match wins:

1. **`--model` CLI argument**
   ```bash
   python main.py --model path/to/your_model.keras
   ```
2. **`LEATHER_MODEL_PATH` environment variable**
   ```bash
   # Linux / macOS
   LEATHER_MODEL_PATH=/path/to/model.keras python main.py
   # Windows (PowerShell)
   $env:LEATHER_MODEL_PATH = "C:\path\to\model.keras"; python main.py
   ```
3. **Default location: `models/best_attention_unet.keras`**
   Drop your file there and just run `python main.py`.
4. **GUI button.** If none of the above resolve, the app still launches
   normally — click **Load model (.keras)…** in the top-left panel.

## Running

```bash
python main.py
```

Workflow inside the app:

1. Wait for the model to load (status bar shows progress; takes 10–30 s
   the first time as TensorFlow compiles the graph).
2. Pick a **Source**: Webcam / Image / Video.
3. For Webcam: set the camera index (0 is usually the built-in camera).
   For Image / Video: click **Browse…** and pick the file.
4. Click **▶ Start**.
5. Adjust display:
   * **Display mode** — Overlay, Mask only, Side-by-side, Original
   * **Overlay opacity** — slider
   * **Defect classes** — checkboxes hide individual classes from the
     overlay (stats are unaffected)
6. Set **Pass/Fail threshold** (% defect area) — the right-hand badge
   turns red above the threshold, green below.
7. Optional: enter **Sample W/H** in mm to get a real-world mm²
   reading in the stats panel.
8. Click **📷 Save snapshot** to dump the current frame plus mask and
   stats.

## Snapshot outputs

Each snapshot produces five files in the chosen folder, all sharing the
same timestamp prefix `snapshot_YYYYMMDD_HHMMSS_`:

| Suffix             | Contents                                        |
|--------------------|-------------------------------------------------|
| `_original.png`    | Original BGR frame                              |
| `_overlay.png`     | Overlay with defects highlighted                |
| `_mask_color.png`  | Colorized mask (per-class colors)               |
| `_mask_raw.png`    | Raw integer class mask (0–5) for downstream use |
| `_stats.json`      | Per-class pixels/percent, verdict, threshold    |

## Performance

Per-frame inference time depends on hardware:

* **GPU (any CUDA-capable):** real-time at webcam frame rate (30+ FPS).
* **CPU only:** roughly 3–10 FPS at typical webcam resolution.

If CPU performance is unacceptable, ensure `tensorflow` was installed
with GPU support (`pip install tensorflow[and-cuda]` on Linux) and that
a GPU is visible (`tf.config.list_physical_devices('GPU')` is non-empty).

## A note on import order (Windows)

`main.py` and `src/gui.py` both import `src.detector` (which loads
TensorFlow) **before** PyQt5 and OpenCV. This is deliberate: on Windows,
PyQt5 and cv2 ship bundled MSVC runtime DLLs that, if loaded first,
prevent TF's native runtime from initialising (`DllMain returned FALSE`,
Windows error 0x45A). The fix is to load TF first so it resolves against
the system MSVC runtime; PyQt5/cv2 then reuse the same runtime.

Do not reorder the imports near the top of `main.py` or `src/gui.py`
without testing on Windows.

## Troubleshooting

* **Model fails to load with a `quantization_config` error.**
  `detector.py` already patches `keras.layers.Dense.__init__` to absorb
  this kwarg. If you still see the error, your TF/Keras version is too
  old — upgrade to `tensorflow >= 2.16`.
* **`DllMain returned FALSE` / TF fails to initialise on Windows.**
  See the import-order note above. Make sure `src.detector` is imported
  before PyQt5/cv2.
* **Webcam opens but no frames arrive.**
  Try a different camera index (0, 1, 2…). On Windows, some webcam
  drivers only expose certain backends; OpenCV will pick automatically
  but slowly. If it hangs, kill the app and try again.
* **First frame is very slow.**
  Expected — TF compiles the model graph on the first prediction.
  The detector runs one dummy prediction during `warmup()` to absorb
  this, but a small additional stall on the first real frame can still
  happen.
* **Video file plays slowly.**
  Video mode processes frames as fast as inference allows; it does not
  respect the video's native FPS. That is intentional (each frame is
  segmented, not just shown).
* **Snapshot button is disabled.**
  No frame has been processed yet. Wait for at least one prediction.