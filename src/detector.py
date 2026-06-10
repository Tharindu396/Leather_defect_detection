"""
detector.py
-----------
Core leather-defect detection module.

Wraps the trained Attention U-Net (best_attention_unet.keras) and exposes a
clean per-frame API for the live-detection GUI.

Model contract (matches Calculator.ipynb):
    Input  : grayscale, 256x256, float32 in [0, 1], shape (1, 256, 256, 1)
    Output : softmax over 6 classes, shape (1, 256, 256, 6)
             classes = [background, color, cut, fold, glue, poke]

All preprocessing / postprocessing / colour palette match the original
notebook exactly. No algorithmic changes.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Silence TF logs *before* importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf  # noqa: E402
import keras as _keras   # noqa: E402


# ---------------------------------------------------------------------------
# Constants — kept identical to Calculator.ipynb
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (256, 256)
NUM_CLASSES: int = 6
CLASS_NAMES: Tuple[str, ...] = (
    "background", "color", "cut", "fold", "glue", "poke",
)
DEFECT_CLASS_NAMES: Tuple[str, ...] = CLASS_NAMES[1:]

# RGB palette used by the GUI and for colorized masks.
CLASS_COLORS_RGB: np.ndarray = np.array(
    [
        [0,   0,   0  ],  # background -> black
        [255, 0,   0  ],  # color      -> red
        [0,   255, 0  ],  # cut        -> green
        [0,   0,   255],  # fold       -> blue
        [255, 255, 0  ],  # glue       -> yellow
        [255, 0,   255],  # poke       -> magenta
    ],
    dtype=np.uint8,
)

FOCAL_GAMMA: float = 2.0
FOCAL_ALPHA: float = 0.25
DICE_SMOOTH: float = 1.0


# ---------------------------------------------------------------------------
# Custom objects — required when calling tf.keras.models.load_model
# (copied verbatim from Calculator.ipynb)
# ---------------------------------------------------------------------------
def focal_loss(y_true, y_pred, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    y_oh = tf.one_hot(y_true, NUM_CLASSES)
    ce = -y_oh * tf.math.log(y_pred)
    pt = tf.reduce_sum(y_oh * y_pred, axis=-1)
    fw = alpha * tf.pow(1.0 - pt, gamma)
    return tf.reduce_mean(tf.reduce_sum(ce, axis=-1) * fw)


def dice_loss_fn(y_true, y_pred, smooth: float = DICE_SMOOTH):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    y_oh = tf.one_hot(y_true, NUM_CLASSES)
    inter = tf.reduce_sum(y_oh * y_pred, axis=0)
    union = tf.reduce_sum(y_oh, axis=0) + tf.reduce_sum(y_pred, axis=0)
    dice_pc = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice_pc[1:])


def combined_loss(y_true, y_pred):
    return 0.5 * focal_loss(y_true, y_pred) + 0.5 * dice_loss_fn(y_true, y_pred)


def mean_iou_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(
        tf.argmax(tf.reshape(y_pred, [-1, NUM_CLASSES]), axis=-1), tf.int32
    )
    cm = tf.cast(
        tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES),
        tf.float32,
    )
    diag = tf.linalg.diag_part(cm)
    denom = tf.reduce_sum(cm, 1) + tf.reduce_sum(cm, 0) - diag
    iou = tf.where(denom > 0, diag / denom, tf.zeros_like(diag))
    valid = tf.cast(denom > 0, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(iou), tf.reduce_sum(valid))


def mean_dice_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(
        tf.argmax(tf.reshape(y_pred, [-1, NUM_CLASSES]), axis=-1), tf.int32
    )
    cm = tf.cast(
        tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES),
        tf.float32,
    )
    diag = tf.linalg.diag_part(cm)
    denom = tf.reduce_sum(cm, 1) + tf.reduce_sum(cm, 0)
    dice = tf.where(denom > 0, 2.0 * diag / denom, tf.zeros_like(diag))
    valid = tf.cast(denom > 0, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(dice), tf.reduce_sum(valid))


CUSTOM_OBJECTS: Dict[str, object] = {
    "combined_loss": combined_loss,
    "mean_iou_metric": mean_iou_metric,
    "mean_dice_metric": mean_dice_metric,
}


# ---------------------------------------------------------------------------
# Keras 3.x Dense quantization_config patch
# (copied from Calculator.ipynb — required to load models saved by a newer
# Keras version that serialised the `quantization_config` kwarg)
# ---------------------------------------------------------------------------
def _load_with_dense_patch(model_path: str):
    """Load a .keras model while temporarily monkey-patching Dense.__init__
    to absorb the unknown `quantization_config` kwarg, then always restore."""
    original_init = _keras.layers.Dense.__init__

    def patched_init(self, *args, quantization_config=None, **kwargs):
        original_init(self, *args, **kwargs)

    _keras.layers.Dense.__init__ = patched_init
    try:
        # compile=False -> skip optimiser/metrics rebuild (faster, inference-only).
        return tf.keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, compile=False,
        )
    finally:
        _keras.layers.Dense.__init__ = original_init


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class LeatherDefectDetector:
    """Per-frame defect detector wrapping the trained Attention U-Net.

    Usage:
        det = LeatherDefectDetector("best_attention_unet.keras")
        det.warmup()
        class_mask, stats = det.predict(bgr_frame)
        color_mask = LeatherDefectDetector.colorize_mask(
            class_mask, class_visibility=(True, True, True, True, True),
        )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path: str = model_path
        self.model = _load_with_dense_patch(model_path)
        self._warmed: bool = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Run one dummy prediction so the first real frame doesn't stall on
        TF graph compilation. Cheap (~0.5-3 s on CPU, faster on GPU)."""
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)
        _ = self.model(dummy, training=False)
        self._warmed = True

    @staticmethod
    def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
        """BGR uint8 frame -> (1, 256, 256, 1) float32 tensor in [0, 1]."""
        if bgr_frame.ndim == 3:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_frame
        gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        gray = gray.astype(np.float32) / 255.0
        return gray[np.newaxis, ..., np.newaxis]

    def predict(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run inference on a single BGR frame.

        Returns:
            class_mask : uint8 array (H_in, W_in) — integer class per pixel,
                         upscaled to the input frame size with nearest-neighbour.
            stats      : dict with total/per-class pixel counts and percentages.
        """
        h_in, w_in = bgr_frame.shape[:2]
        tensor = self.preprocess(bgr_frame)

        # Direct call (NOT model.predict) — significantly faster for
        # single-batch inference; avoids the tf.data and progress-bar overhead.
        probs = self.model(tensor, training=False).numpy()[0]  # (256, 256, 6)
        small_mask = np.argmax(probs, axis=-1).astype(np.uint8)

        # Nearest-neighbour upscaling preserves crisp class boundaries
        # (bilinear would bleed class indices across boundaries).
        class_mask = cv2.resize(
            small_mask, (w_in, h_in), interpolation=cv2.INTER_NEAREST,
        )
        stats = self._compute_stats(class_mask)
        return class_mask, stats

    # ------------------------------------------------------------------
    # Visual helpers (pure — safe to call from any thread)
    # ------------------------------------------------------------------
    @staticmethod
    def colorize_mask(
        class_mask: np.ndarray,
        class_visibility: Optional[Tuple[bool, ...]] = None,
    ) -> np.ndarray:
        """Integer class mask -> RGB color image.

        `class_visibility` is an optional length-5 tuple of bools matching the
        5 defect classes (color, cut, fold, glue, poke). Classes set to False
        are folded into background (black) for display only — stats are unaffected
        because they are computed from the unfiltered mask upstream.
        """
        if class_visibility is None:
            return CLASS_COLORS_RGB[class_mask]

        display_mask = class_mask.copy()
        for class_idx, visible in enumerate(class_visibility, start=1):
            if not visible:
                display_mask[display_mask == class_idx] = 0
        return CLASS_COLORS_RGB[display_mask]

    @staticmethod
    def make_overlay(
        bgr_frame: np.ndarray,
        color_mask_rgb: np.ndarray,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Blend the RGB color mask onto the BGR frame; background pixels
        remain unchanged. Returns a BGR image at the original frame size."""
        if color_mask_rgb.shape[:2] != bgr_frame.shape[:2]:
            color_mask_rgb = cv2.resize(
                color_mask_rgb,
                (bgr_frame.shape[1], bgr_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)

        defect_pixels = np.any(color_mask_bgr > 0, axis=-1)
        out = bgr_frame.copy()
        if not defect_pixels.any():
            return out

        blended = cv2.addWeighted(
            bgr_frame, 1.0 - alpha, color_mask_bgr, alpha, 0.0,
        )
        out[defect_pixels] = blended[defect_pixels]
        return out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_stats(class_mask: np.ndarray) -> Dict:
        total = int(class_mask.size)
        per_class_px = {
            name: int(np.sum(class_mask == idx))
            for idx, name in enumerate(CLASS_NAMES)
        }
        defect_px = total - per_class_px["background"]
        return {
            "total_pixels": total,
            "defect_pixels": defect_px,
            "defect_pct": (defect_px / total) * 100.0 if total > 0 else 0.0,
            "per_class_px": per_class_px,
            "per_class_pct": {
                name: (px / total) * 100.0 if total > 0 else 0.0
                for name, px in per_class_px.items()
            },
        }

    @staticmethod
    def defect_mm2(
        defect_pixels: int,
        frame_h: int,
        frame_w: int,
        sample_h_mm: float,
        sample_w_mm: float,
    ) -> float:
        """Convert a defect pixel count to mm² given the physical sample size."""
        if sample_h_mm <= 0 or sample_w_mm <= 0 or frame_h <= 0 or frame_w <= 0:
            return 0.0
        px_per_mm2 = (frame_h * frame_w) / (sample_h_mm * sample_w_mm)
        return defect_pixels / px_per_mm2 if px_per_mm2 > 0 else 0.0
