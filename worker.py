"""
worker.py
---------
Background QThread that handles frame capture + model inference, emitting
processed frames to the GUI thread via Qt signals (queued automatically).

The worker emits the RAW integer class mask (not a colorised RGB image) so
that the GUI can re-colorise instantly when the user toggles class visibility
or the overlay opacity slider — without re-running inference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from detector import LeatherDefectDetector


# ---------------------------------------------------------------------------
# Source enum / settings dataclass
# ---------------------------------------------------------------------------
class SourceType(Enum):
    WEBCAM = "webcam"
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class DetectionSettings:
    """User-tunable settings that the worker can read each loop iteration.

    Only contains things that *might* change while the worker is running.
    Model-side things (defect class visibility, overlay opacity) are handled
    GUI-side because they don't affect inference.
    """
    # Reserved for future use (e.g. min-confidence threshold, ROI crop).
    placeholder: bool = field(default=True, repr=False)


# ---------------------------------------------------------------------------
# Detection worker
# ---------------------------------------------------------------------------
class DetectionWorker(QThread):
    """Captures frames from a source, runs inference, emits results.

    Signals:
        frame_ready(bgr_frame, class_mask, stats)
            Emitted after each successful inference.
              * bgr_frame  : np.ndarray, original BGR frame at source resolution
              * class_mask : np.ndarray, uint8 integer class mask at same resolution
              * stats      : dict, per-class pixel counts and percentages
        error(msg)
            Emitted on capture / inference failure.
        finished_running()
            Emitted on clean exit (user stop or end-of-source).
        fps_updated(fps)
            Emitted ~1 Hz with the current effective FPS (inference + capture).
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error = pyqtSignal(str)
    finished_running = pyqtSignal()
    fps_updated = pyqtSignal(float)

    def __init__(
        self,
        detector: LeatherDefectDetector,
        source_type: SourceType,
        source: object,  # int for webcam index, str path for image/video
        settings: DetectionSettings,
        parent=None,
    ):
        super().__init__(parent)
        self._detector = detector
        self._source_type = source_type
        self._source = source
        self._settings = settings
        self._running = False

    # ------------------------------------------------------------------
    # Public API (called from the UI thread)
    # ------------------------------------------------------------------
    def update_settings(self, settings: DetectionSettings) -> None:
        """Hot-update settings used by the loop. Atomic assignment in CPython."""
        self._settings = settings

    def stop(self) -> None:
        """Signal the loop to exit. Caller should `wait()` afterwards."""
        self._running = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True
        try:
            if self._source_type == SourceType.IMAGE:
                self._run_image()
            else:
                self._run_stream()
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self.error.emit(f"Worker error: {exc}")
        finally:
            self._running = False
            self.finished_running.emit()

    # ------------------------------------------------------------------
    # Single-image mode: predict once, emit, exit.
    # ------------------------------------------------------------------
    def _run_image(self) -> None:
        frame = cv2.imread(str(self._source))
        if frame is None:
            self.error.emit(f"Could not read image: {self._source}")
            return
        self._infer_and_emit(frame)
        self.fps_updated.emit(0.0)
        # Exit immediately — GUI keeps showing the result from its cache.

    # ------------------------------------------------------------------
    # Webcam / video stream mode
    # ------------------------------------------------------------------
    def _run_stream(self) -> None:
        if self._source_type == SourceType.WEBCAM:
            cap = cv2.VideoCapture(int(self._source))
        else:
            cap = cv2.VideoCapture(str(self._source))

        if not cap.isOpened():
            self.error.emit(f"Could not open source: {self._source}")
            return

        # Best-effort: keep the capture buffer small so we get the freshest
        # frame from the webcam (inference is slower than capture, so without
        # this we'd accumulate latency).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 — not all backends support this
            pass

        fps_window_start = time.perf_counter()
        fps_frame_count = 0

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self._source_type == SourceType.VIDEO:
                        break  # end of video file, clean exit
                    self.error.emit("Lost frame from source.")
                    break

                self._infer_and_emit(frame)

                fps_frame_count += 1
                elapsed = time.perf_counter() - fps_window_start
                if elapsed >= 1.0:
                    self.fps_updated.emit(fps_frame_count / elapsed)
                    fps_window_start = time.perf_counter()
                    fps_frame_count = 0
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    def _infer_and_emit(self, frame_bgr: np.ndarray) -> None:
        class_mask, stats = self._detector.predict(frame_bgr)
        # frame_bgr is freshly allocated by cv2 each call, and class_mask is
        # freshly allocated by cv2.resize — both are safe to pass by reference.
        self.frame_ready.emit(frame_bgr, class_mask, stats)
