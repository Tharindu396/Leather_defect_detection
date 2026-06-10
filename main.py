"""
main.py — Entry point for the Leather Defect Detection GUI.

Usage:
    python main.py
    python main.py --model path/to/model.keras
    python main.py -m path/to/model.keras

Model path resolution order (first match wins):
    1. --model / -m CLI argument
    2. LEATHER_MODEL_PATH environment variable
    3. ./models/best_attention_unet.keras (default project location)
    4. None — use the "Load model…" button in the GUI

If none of 1–3 resolve, the app still launches normally; the user just
needs to load a model manually via the button.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_NAME = "v1.keras"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / DEFAULT_MODEL_NAME


def resolve_model_path(cli_arg: Optional[str]) -> Optional[str]:
    """Return the .keras model path to auto-load, or None for no auto-load."""
    # 1. Explicit CLI arg wins.
    if cli_arg:
        p = Path(cli_arg).expanduser()
        if not p.exists():
            print(f"[WARN] --model path not found: {p}", file=sys.stderr)
            return None
        return str(p)

    # 2. Environment variable.
    env = os.environ.get("LEATHER_MODEL_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return str(p)
        print(f"[WARN] LEATHER_MODEL_PATH not found: {p}", file=sys.stderr)

    # 3. Default project location.
    if DEFAULT_MODEL_PATH.exists():
        return str(DEFAULT_MODEL_PATH)

    # 4. Fall through — user will load via the button.
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live leather-defect detection GUI (Attention U-Net).",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to the .keras model file. Overrides env / default location.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    # ----------------------------------------------------------------------
    # IMPORTANT — import order matters on Windows.
    #
    # `src.detector` loads TensorFlow, which must be loaded BEFORE PyQt5 and
    # cv2. Both PyQt5 and OpenCV ship bundled MSVC runtime DLLs that, if
    # loaded first, prevent TF's native runtime from initialising
    # (error 0x45A: DllMain returned FALSE).
    #
    # Triggering `from src.gui import MainWindow` would also work — because
    # src/gui.py imports `.detector` at its very top — but doing it
    # explicitly here makes the intent obvious to anyone reading main.py.
    # ----------------------------------------------------------------------
    import src.detector  # noqa: F401 — load TF before any GUI library

    # Safe to import PyQt5 now (TF is already initialised).
    from PyQt5.QtWidgets import QApplication
    from src.gui import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent look across platforms

    window = MainWindow(initial_model_path=model_path)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()