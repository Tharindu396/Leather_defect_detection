"""
gui.py
------
PyQt5 GUI for live leather-defect detection with the Attention U-Net.

Layout:
    +--------------+-----------------------+--------------+
    | controls     |   live video view     |  stats       |
    | (left, 280)  |   (center, stretch)   |  (right,280) |
    +--------------+-----------------------+--------------+
    |                  status bar                          |
    +------------------------------------------------------+

Run:
    python gui.py
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Import order matters on Windows.
# TensorFlow must load before PyQt5 and cv2 — both ship bundled MSVC runtime
# DLLs (msvcp140.dll, vcruntime140*.dll) that, if loaded first, prevent TF's
# native runtime from initialising (error 0x45A: DllMain returned FALSE).
# Importing `detector` first lets it set TF env vars and load tensorflow
# against the system MSVC runtime; PyQt5/cv2 then reuse the already-resolved
# runtime in-process.
# Do NOT reorder these imports without testing `python gui.py` on Windows.
# --------------------------------------------------------------------------
from detector import (  # noqa: E402  (intentional: must precede PyQt5/cv2)
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    DEFECT_CLASS_NAMES,
    LeatherDefectDetector,
)

# Standard library
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third-party (safe now that TF is already loaded)
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Local (the rest)
from worker import DetectionSettings, DetectionWorker, SourceType


# ---------------------------------------------------------------------------
# Aspect-ratio-preserving video display widget
# ---------------------------------------------------------------------------
class VideoDisplay(QLabel):
    """QLabel that scales pixmaps to fit while preserving aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet(
            "background-color: #1e1e1e; color: #888; "
            "border: 1px solid #444; font-size: 14px;"
        )
        self.setText("No source")
        self._pix: Optional[QPixmap] = None

    def set_image(self, bgr: np.ndarray) -> None:
        # BGR -> RGB (contiguous required for QImage to wrap safely).
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        # copy() because the numpy buffer is reused by the worker on the next frame.
        self._pix = QPixmap.fromImage(qimg.copy())
        self._refresh()

    def clear_image(self) -> None:
        self._pix = None
        self.setText("No source")
        self.repaint()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))


# ---------------------------------------------------------------------------
# One row in the per-class stats panel: colour swatch + name + % + px
# ---------------------------------------------------------------------------
class ClassStatRow(QWidget):
    def __init__(self, class_idx: int, class_name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        r, g, b = CLASS_COLORS_RGB[class_idx]
        swatch.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;"
        )
        layout.addWidget(swatch)

        name_label = QLabel(class_name)
        name_label.setMinimumWidth(70)
        layout.addWidget(name_label)

        layout.addStretch()

        self.pct_label = QLabel("0.00%")
        self.pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.pct_label.setMinimumWidth(60)
        layout.addWidget(self.pct_label)

        self.px_label = QLabel("0 px")
        self.px_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.px_label.setMinimumWidth(80)
        self.px_label.setStyleSheet("color: #888;")
        layout.addWidget(self.px_label)

    def update_stats(self, pct: float, px: int) -> None:
        self.pct_label.setText(f"{pct:.2f}%")
        self.px_label.setText(f"{px:,} px")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leather Defect Detection — Live")
        self.resize(1280, 800)

        # State -----------------------------------------------------------
        self.detector: Optional[LeatherDefectDetector] = None
        self.worker: Optional[DetectionWorker] = None
        self._selected_file_path: Optional[str] = None

        self._last_bgr: Optional[np.ndarray] = None
        self._last_class_mask: Optional[np.ndarray] = None
        self._last_stats: Optional[dict] = None

        self._pass_fail_threshold: float = 1.0  # %

        # Build UI --------------------------------------------------------
        self._build_ui()
        self._update_button_states()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        central_layout.addWidget(self._build_left_panel(), 0)
        central_layout.addWidget(self._build_center_panel(), 1)
        central_layout.addWidget(self._build_right_panel(), 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._set_status("Ready — load a model to begin.")

    # ------------------------------------------------------------------
    # Left panel: controls
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_classes_group())
        layout.addWidget(self._build_threshold_group())
        layout.addStretch()
        return panel

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        v = QVBoxLayout(group)

        self.btn_load_model = QPushButton("Load model (.keras)…")
        self.btn_load_model.clicked.connect(self._on_load_model)
        v.addWidget(self.btn_load_model)

        self.lbl_model = QLabel("<i>No model loaded</i>")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet("color: #666;")
        v.addWidget(self.lbl_model)
        return group

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        v = QVBoxLayout(group)

        self.cmb_source_type = QComboBox()
        self.cmb_source_type.addItems(["Webcam", "Image", "Video"])
        self.cmb_source_type.currentIndexChanged.connect(self._on_source_type_changed)
        v.addWidget(self.cmb_source_type)

        # Webcam index row -------------------------------------------------
        self.webcam_row = QWidget()
        wc = QHBoxLayout(self.webcam_row)
        wc.setContentsMargins(0, 0, 0, 0)
        wc.addWidget(QLabel("Camera index:"))
        self.spn_webcam = QSpinBox()
        self.spn_webcam.setRange(0, 9)
        wc.addWidget(self.spn_webcam)
        wc.addStretch()
        v.addWidget(self.webcam_row)

        # File row ---------------------------------------------------------
        self.file_row = QWidget()
        fr = QHBoxLayout(self.file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        self.lbl_file = QLabel("<i>No file selected</i>")
        self.lbl_file.setStyleSheet("color: #666;")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._on_browse_file)
        fr.addWidget(self.lbl_file, 1)
        fr.addWidget(self.btn_browse, 0)
        v.addWidget(self.file_row)

        # Start / Stop -----------------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        v.addLayout(btn_row)

        self.btn_snapshot = QPushButton("📷  Save snapshot")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        v.addWidget(self.btn_snapshot)

        self._on_source_type_changed(0)  # initial visibility
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Display")
        v = QVBoxLayout(group)

        self.rad_overlay = QRadioButton("Overlay")
        self.rad_mask = QRadioButton("Mask only")
        self.rad_sbs = QRadioButton("Side-by-side")
        self.rad_orig = QRadioButton("Original")
        self.rad_overlay.setChecked(True)
        for r in (self.rad_overlay, self.rad_mask, self.rad_sbs, self.rad_orig):
            r.toggled.connect(self._refresh_view)
            v.addWidget(r)

        v.addWidget(QLabel("Overlay opacity:"))
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(55)
        self.sld_alpha.valueChanged.connect(self._refresh_view)
        v.addWidget(self.sld_alpha)
        return group

    def _build_classes_group(self) -> QGroupBox:
        group = QGroupBox("Defect classes (display only)")
        v = QVBoxLayout(group)

        self.class_checkboxes = []
        for class_idx, name in enumerate(DEFECT_CLASS_NAMES, start=1):
            cb = QCheckBox(name)
            cb.setChecked(True)
            r, g, b = CLASS_COLORS_RGB[class_idx]
            cb.setStyleSheet(
                "QCheckBox { padding-left: 4px; } "
                f"QCheckBox::indicator {{ width: 14px; height: 14px; "
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888; }}"
                "QCheckBox::indicator:unchecked { background-color: #ddd; }"
            )
            cb.stateChanged.connect(self._refresh_view)
            self.class_checkboxes.append(cb)
            v.addWidget(cb)
        return group

    def _build_threshold_group(self) -> QGroupBox:
        group = QGroupBox("Pass / Fail & physical scale")
        g = QGridLayout(group)
        g.setColumnStretch(1, 1)

        g.addWidget(QLabel("Defect %\nthreshold:"), 0, 0)
        self.spn_thresh = QDoubleSpinBox()
        self.spn_thresh.setRange(0.0, 100.0)
        self.spn_thresh.setSingleStep(0.1)
        self.spn_thresh.setValue(1.0)
        self.spn_thresh.setSuffix(" %")
        self.spn_thresh.valueChanged.connect(self._on_threshold_changed)
        g.addWidget(self.spn_thresh, 0, 1)

        g.addWidget(QLabel("Sample W:"), 1, 0)
        self.spn_width_mm = QDoubleSpinBox()
        self.spn_width_mm.setRange(0.0, 10000.0)
        self.spn_width_mm.setSuffix(" mm")
        self.spn_width_mm.valueChanged.connect(self._refresh_stats_display)
        g.addWidget(self.spn_width_mm, 1, 1)

        g.addWidget(QLabel("Sample H:"), 2, 0)
        self.spn_height_mm = QDoubleSpinBox()
        self.spn_height_mm.setRange(0.0, 10000.0)
        self.spn_height_mm.setSuffix(" mm")
        self.spn_height_mm.valueChanged.connect(self._refresh_stats_display)
        g.addWidget(self.spn_height_mm, 2, 1)
        return group

    # ------------------------------------------------------------------
    # Center panel: video display
    # ------------------------------------------------------------------
    def _build_center_panel(self) -> QWidget:
        self.video = VideoDisplay()
        return self.video

    # ------------------------------------------------------------------
    # Right panel: stats / verdict
    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Verdict badge ---------------------------------------------------
        self.lbl_verdict = QLabel("NO DATA")
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setMinimumHeight(64)
        f = self.lbl_verdict.font()
        f.setPointSize(18)
        f.setBold(True)
        self.lbl_verdict.setFont(f)
        self._set_verdict("NO DATA", "#444")
        v.addWidget(self.lbl_verdict)

        # FPS -------------------------------------------------------------
        fps_group = QGroupBox("Performance")
        fps_v = QVBoxLayout(fps_group)
        self.lbl_fps = QLabel("FPS: —")
        f_fps = self.lbl_fps.font()
        f_fps.setPointSize(13)
        self.lbl_fps.setFont(f_fps)
        fps_v.addWidget(self.lbl_fps)
        v.addWidget(fps_group)

        # Defect summary --------------------------------------------------
        sum_group = QGroupBox("Defect summary")
        sum_v = QVBoxLayout(sum_group)
        self.lbl_total_pct = QLabel("Total defect: 0.00%")
        f_total = self.lbl_total_pct.font()
        f_total.setPointSize(13)
        f_total.setBold(True)
        self.lbl_total_pct.setFont(f_total)
        self.lbl_mm2 = QLabel("Area: — mm²  (set sample W/H)")
        self.lbl_mm2.setStyleSheet("color: #666;")
        sum_v.addWidget(self.lbl_total_pct)
        sum_v.addWidget(self.lbl_mm2)
        v.addWidget(sum_group)

        # Per-class breakdown ---------------------------------------------
        cls_group = QGroupBox("Per-class breakdown")
        cls_v = QVBoxLayout(cls_group)
        self.class_rows = {}
        for idx, name in enumerate(CLASS_NAMES):
            row = ClassStatRow(idx, name)
            self.class_rows[name] = row
            cls_v.addWidget(row)
        v.addWidget(cls_group)

        v.addStretch()
        return panel

    # ==================================================================
    # State / button management
    # ==================================================================
    def _set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def _set_verdict(self, text: str, color_hex: str) -> None:
        self.lbl_verdict.setText(text)
        self.lbl_verdict.setStyleSheet(
            f"background-color: {color_hex}; color: white; border-radius: 6px;"
        )

    def _update_button_states(self) -> None:
        has_model = self.detector is not None
        is_running = self.worker is not None and self.worker.isRunning()

        self.btn_start.setEnabled(has_model and not is_running)
        self.btn_stop.setEnabled(is_running)
        self.btn_snapshot.setEnabled(self._last_bgr is not None)
        self.btn_load_model.setEnabled(not is_running)
        self.cmb_source_type.setEnabled(not is_running)
        self.spn_webcam.setEnabled(not is_running)
        self.btn_browse.setEnabled(not is_running)

    def _on_source_type_changed(self, idx: int) -> None:
        is_webcam = (idx == 0)
        self.webcam_row.setVisible(is_webcam)
        self.file_row.setVisible(not is_webcam)

    def _on_browse_file(self) -> None:
        if self.cmb_source_type.currentIndex() == 1:
            filt = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        else:
            filt = "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            self._selected_file_path = path
            self.lbl_file.setText(os.path.basename(path))
            self.lbl_file.setStyleSheet("")
            self.lbl_file.setToolTip(path)

    # ==================================================================
    # Model loading
    # ==================================================================
    def _on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Keras model", "", "Keras model (*.keras *.h5)",
        )
        if not path:
            return

        self._set_status("Loading model… (this can take 10–30 seconds)")
        QApplication.processEvents()
        try:
            self.detector = LeatherDefectDetector(path)
            self.detector.warmup()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model load failed", str(exc))
            self.detector = None
            self.lbl_model.setText("<i>No model loaded</i>")
            self.lbl_model.setStyleSheet("color: #666;")
            self._set_status("Model load failed.")
            self._update_button_states()
            return

        self.lbl_model.setText(f"✓ {os.path.basename(path)}")
        self.lbl_model.setStyleSheet("color: #0a7a0a; font-weight: bold;")
        self.lbl_model.setToolTip(path)
        self._set_status(f"Model loaded: {os.path.basename(path)}")
        self._update_button_states()

    # ==================================================================
    # Start / Stop
    # ==================================================================
    def _on_start(self) -> None:
        if self.detector is None:
            QMessageBox.warning(self, "No model", "Please load a model first.")
            return

        src_idx = self.cmb_source_type.currentIndex()
        if src_idx == 0:
            source_type = SourceType.WEBCAM
            source: object = self.spn_webcam.value()
        elif src_idx == 1:
            if not self._selected_file_path:
                QMessageBox.warning(self, "No file", "Please select an image file.")
                return
            source_type = SourceType.IMAGE
            source = self._selected_file_path
        else:
            if not self._selected_file_path:
                QMessageBox.warning(self, "No file", "Please select a video file.")
                return
            source_type = SourceType.VIDEO
            source = self._selected_file_path

        self.worker = DetectionWorker(
            detector=self.detector,
            source_type=source_type,
            source=source,
            settings=DetectionSettings(),
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished_running.connect(self._on_worker_finished)
        self.worker.fps_updated.connect(self._on_fps_updated)
        self.worker.start()

        self._set_status("Running…")
        self._update_button_states()

    def _on_stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self._set_status("Stopped.")
        self._update_button_states()

    # ==================================================================
    # Worker signal handlers
    # ==================================================================
    def _on_frame_ready(self, bgr: np.ndarray, class_mask: np.ndarray, stats: dict) -> None:
        self._last_bgr = bgr
        self._last_class_mask = class_mask
        self._last_stats = stats
        self._refresh_view()
        self._refresh_stats_display()
        # Enable the snapshot button now that we have data.
        self.btn_snapshot.setEnabled(True)

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Worker error", msg)
        self._on_stop()

    def _on_worker_finished(self) -> None:
        self._set_status("Finished.")
        # Don't drop self.worker here — _on_stop already does it cleanly.
        # But we *do* need to refresh button states for end-of-video case.
        self._update_button_states()

    def _on_fps_updated(self, fps: float) -> None:
        self.lbl_fps.setText("FPS: —" if fps <= 0 else f"FPS: {fps:.1f}")

    def _on_threshold_changed(self, _v: float) -> None:
        self._pass_fail_threshold = self.spn_thresh.value()
        self._refresh_stats_display()

    # ==================================================================
    # Display refresh (called whenever cached state or display settings change)
    # ==================================================================
    def _current_class_visibility(self):
        return tuple(cb.isChecked() for cb in self.class_checkboxes)

    def _refresh_view(self) -> None:
        if self._last_bgr is None or self._last_class_mask is None:
            return
        bgr = self._last_bgr
        class_mask = self._last_class_mask

        color_mask_rgb = LeatherDefectDetector.colorize_mask(
            class_mask, class_visibility=self._current_class_visibility(),
        )

        if self.rad_overlay.isChecked():
            alpha = self.sld_alpha.value() / 100.0
            view = LeatherDefectDetector.make_overlay(bgr, color_mask_rgb, alpha=alpha)
        elif self.rad_mask.isChecked():
            view = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
        elif self.rad_sbs.isChecked():
            mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
            view = np.hstack([bgr, mask_bgr])
        else:  # original
            view = bgr

        self.video.set_image(view)

    def _refresh_stats_display(self) -> None:
        if self._last_stats is None:
            return
        s = self._last_stats
        self.lbl_total_pct.setText(f"Total defect: {s['defect_pct']:.2f}%")

        # mm² readout (optional)
        w = self.spn_width_mm.value()
        h = self.spn_height_mm.value()
        if w > 0 and h > 0 and self._last_bgr is not None:
            fh, fw = self._last_bgr.shape[:2]
            mm2 = LeatherDefectDetector.defect_mm2(
                s["defect_pixels"], fh, fw, h, w,
            )
            self.lbl_mm2.setText(f"Area: {mm2:,.1f} mm²  ({mm2 / 100:.2f} cm²)")
            self.lbl_mm2.setStyleSheet("")
        else:
            self.lbl_mm2.setText("Area: — mm²  (set sample W/H)")
            self.lbl_mm2.setStyleSheet("color: #666;")

        # Per-class rows
        for name, row in self.class_rows.items():
            pct = s["per_class_pct"].get(name, 0.0)
            px = s["per_class_px"].get(name, 0)
            row.update_stats(pct, px)

        # Pass / Fail badge
        if s["defect_pct"] >= self._pass_fail_threshold:
            self._set_verdict(f"FAIL  ({s['defect_pct']:.2f}%)", "#b32424")
        else:
            self._set_verdict(f"PASS  ({s['defect_pct']:.2f}%)", "#2a8a3e")

    # ==================================================================
    # Snapshot
    # ==================================================================
    def _on_snapshot(self) -> None:
        if (
            self._last_bgr is None
            or self._last_class_mask is None
            or self._last_stats is None
        ):
            QMessageBox.information(self, "Nothing to save", "No frame available yet.")
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Choose snapshot folder")
        if not dir_path:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(dir_path) / f"snapshot_{stamp}"

        bgr = self._last_bgr
        visibility = self._current_class_visibility()
        color_mask_rgb = LeatherDefectDetector.colorize_mask(
            self._last_class_mask, class_visibility=visibility,
        )
        mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
        overlay = LeatherDefectDetector.make_overlay(
            bgr, color_mask_rgb, alpha=self.sld_alpha.value() / 100.0,
        )

        cv2.imwrite(f"{base}_original.png", bgr)
        cv2.imwrite(f"{base}_mask_color.png", mask_bgr)
        cv2.imwrite(f"{base}_overlay.png", overlay)
        # Also save the raw integer class mask (useful for downstream analysis).
        cv2.imwrite(f"{base}_mask_raw.png", self._last_class_mask)

        snapshot_meta = {
            "timestamp": stamp,
            "pass_fail_threshold_pct": self._pass_fail_threshold,
            "verdict": (
                "FAIL"
                if self._last_stats["defect_pct"] >= self._pass_fail_threshold
                else "PASS"
            ),
            "class_visibility": dict(zip(DEFECT_CLASS_NAMES, visibility)),
            "stats": self._last_stats,
        }
        with open(f"{base}_stats.json", "w", encoding="utf-8") as f:
            json.dump(snapshot_meta, f, indent=2)

        self._set_status(f"Snapshot saved: {base.name}_*")

    # ==================================================================
    # Clean shutdown
    # ==================================================================
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent look across platforms
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
