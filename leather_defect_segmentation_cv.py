"""
Leather Defect Segmentation using Pure Image Processing & Computer Vision
=========================================================================
VERSION 2 — IMPROVED
No ML models used. Techniques employed:
  1. Reference-based anomaly detection (mean good image as reference)
  2. Multi-scale Gaussian difference maps
  3. Gabor filter bank for texture anomaly detection
  4. Laplacian of Gaussian (LoG) for edge/structure anomalies
  5. Color channel analysis in LAB color space
  6. Percentile-based + Otsu thresholding
  7. Morphological operations (opening, closing, dilation)
  8. Connected component filtering with relaxed constraints
  9. Per-defect-type parameter tuning with improved values
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.morphology import remove_small_objects, disk, opening, closing
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"d:\7th sem\Image processing and CV\Leather_defect_detection\leather"
TRAIN_GOOD_DIR = os.path.join(BASE_DIR, "train", "good")
TEST_DIR = os.path.join(BASE_DIR, "test")
GT_DIR = os.path.join(BASE_DIR, "ground_truth")
OUTPUT_DIR = r"d:\7th sem\Image processing and CV\Leather_defect_detection\segmentation_results_v2"

DEFECT_TYPES = ["color", "cut", "fold", "glue", "poke"]
IMG_SIZE = (512, 512)  # Resize for processing efficiency

# Per-defect-type parameters — V2 improved
DEFECT_PARAMS = {
    "color": {
        "gauss_ksize": 21,
        "percentile_thresh": 97.0,    # Top 3% of anomaly scores
        "morph_kernel": 9,
        "morph_close_iter": 3,
        "morph_open_iter": 1,
        "min_area": 50,
        "use_color": True,
        "use_gabor": True,
        "gabor_weight": 0.25,
        "diff_weight": 0.35,
        "color_weight": 0.30,
        "log_weight": 0.10,
        "final_dilate": 5,
    },
    "cut": {
        "gauss_ksize": 9,
        "percentile_thresh": 96.0,
        "morph_kernel": 5,
        "morph_close_iter": 2,
        "morph_open_iter": 1,
        "min_area": 20,
        "use_color": True,
        "use_gabor": False,
        "gabor_weight": 0.0,
        "diff_weight": 0.45,
        "color_weight": 0.35,
        "log_weight": 0.20,
        "final_dilate": 3,
    },
    "fold": {
        "gauss_ksize": 25,
        "percentile_thresh": 95.5,
        "morph_kernel": 11,
        "morph_close_iter": 3,
        "morph_open_iter": 1,
        "min_area": 100,
        "use_color": False,
        "use_gabor": True,
        "gabor_weight": 0.35,
        "diff_weight": 0.35,
        "color_weight": 0.15,
        "log_weight": 0.15,
        "final_dilate": 7,
    },
    "glue": {
        "gauss_ksize": 15,
        "percentile_thresh": 95.0,
        "morph_kernel": 9,
        "morph_close_iter": 3,
        "morph_open_iter": 1,
        "min_area": 80,
        "use_color": True,
        "use_gabor": True,
        "gabor_weight": 0.25,
        "diff_weight": 0.35,
        "color_weight": 0.25,
        "log_weight": 0.15,
        "final_dilate": 5,
    },
    "poke": {
        "gauss_ksize": 11,
        "percentile_thresh": 96.5,
        "morph_kernel": 7,
        "morph_close_iter": 2,
        "morph_open_iter": 1,
        "min_area": 30,
        "use_color": True,
        "use_gabor": True,
        "gabor_weight": 0.20,
        "diff_weight": 0.40,
        "color_weight": 0.25,
        "log_weight": 0.15,
        "final_dilate": 5,
    },
}


# ============================================================================
# STEP 1: BUILD REFERENCE MODEL FROM GOOD TRAINING IMAGES
# ============================================================================
def build_reference_model(train_dir, img_size=IMG_SIZE, max_images=50):
    """
    Build a reference model by computing the pixelwise mean and standard
    deviation of defect-free (good) training images.
    Also compute reference texture energy via Gabor filter bank.
    """
    print("=" * 70)
    print("STEP 1: Building reference model from good training images...")
    print("=" * 70)

    good_images = sorted(glob.glob(os.path.join(train_dir, "*.png")))
    n = min(len(good_images), max_images)
    print(f"  Using {n} out of {len(good_images)} good images")

    # Accumulators
    sum_gray = None
    sum_sq_gray = None
    sum_color = None
    sum_sq_color = None
    count = 0

    for i, img_path in enumerate(good_images[:n]):
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        color = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

        if sum_gray is None:
            sum_gray = np.zeros_like(gray)
            sum_sq_gray = np.zeros_like(gray)
            sum_color = np.zeros_like(color)
            sum_sq_color = np.zeros_like(color)

        sum_gray += gray
        sum_sq_gray += gray ** 2
        sum_color += color
        sum_sq_color += color ** 2
        count += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n} images")

    mean_gray = sum_gray / count
    std_gray = np.sqrt(np.clip(sum_sq_gray / count - mean_gray ** 2, 0, None))
    std_gray = np.clip(std_gray, 1.0, None)  # Avoid division by zero

    mean_color = sum_color / count
    std_color = np.sqrt(np.clip(sum_sq_color / count - mean_color ** 2, 0, None))
    std_color = np.clip(std_color, 1.0, None)

    # Compute reference Gabor texture energy
    mean_gabor = compute_gabor_energy(mean_gray.astype(np.uint8))

    # Compute reference LoG
    mean_log = compute_log_edge_response(mean_gray.astype(np.uint8))

    print(f"  Reference model built: mean_gray shape={mean_gray.shape}")
    print(f"  Mean gray range: [{mean_gray.min():.1f}, {mean_gray.max():.1f}]")
    print(f"  Std gray range:  [{std_gray.min():.1f}, {std_gray.max():.1f}]")

    return {
        "mean_gray": mean_gray,
        "std_gray": std_gray,
        "mean_color": mean_color,
        "std_color": std_color,
        "mean_gabor": mean_gabor,
        "mean_log": mean_log,
    }


# ============================================================================
# STEP 2: FEATURE EXTRACTION FUNCTIONS
# ============================================================================
def compute_gabor_energy(gray_img, frequencies=[0.05, 0.1, 0.15, 0.2, 0.3],
                         thetas=[0, np.pi/6, np.pi/4, np.pi/3, np.pi/2,
                                 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]):
    """
    Compute texture energy using an expanded Gabor filter bank.
    Returns the mean response across all filter orientations and frequencies.
    """
    gray_float = gray_img.astype(np.float64) / 255.0
    energy = np.zeros_like(gray_float)
    count = 0

    for freq in frequencies:
        for theta in thetas:
            filt_real, filt_imag = gabor(gray_float, frequency=freq, theta=theta)
            magnitude = np.sqrt(filt_real ** 2 + filt_imag ** 2)
            energy += magnitude
            count += 1

    return energy / count


def compute_difference_map(test_gray, ref_mean, ref_std, gauss_ksize=11):
    """
    Compute a normalized difference map between test image and reference.
    Uses multi-scale Gaussian smoothing to suppress texture noise.
    Returns a Z-score normalized difference (higher = more anomalous).
    """
    results = []
    
    for scale_factor in [1, 2, 4]:
        k = gauss_ksize * scale_factor
        if k % 2 == 0:
            k += 1
        
        test_smooth = cv2.GaussianBlur(test_gray.astype(np.float64), (k, k), 0)
        ref_smooth = cv2.GaussianBlur(ref_mean, (k, k), 0)
        ref_std_smooth = cv2.GaussianBlur(ref_std, (k, k), 0)
        ref_std_smooth = np.clip(ref_std_smooth, 1.0, None)

        diff = np.abs(test_smooth - ref_smooth) / ref_std_smooth
        results.append(diff)

    # Combine multi-scale with decreasing weights
    combined = 0.5 * results[0] + 0.3 * results[1] + 0.2 * results[2]
    
    return combined


def compute_color_anomaly(test_color, ref_mean_color, ref_std_color):
    """
    Detect color anomalies in LAB color space.
    Uses the Mahalanobis-like distance per channel.
    """
    diff = np.abs(test_color - ref_mean_color) / ref_std_color

    # L channel = luminance, a,b = chrominance
    # For color defects, chrominance is key
    # For brightness defects (glue, fold), luminance matters
    color_anomaly = 0.2 * diff[:, :, 0] + 0.4 * diff[:, :, 1] + 0.4 * diff[:, :, 2]

    return color_anomaly


def compute_log_edge_response(gray_img, ksize=9):
    """
    Laplacian of Gaussian for detecting structural anomalies (cuts, folds).
    """
    blurred = cv2.GaussianBlur(gray_img.astype(np.float64), (ksize, ksize), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    return np.abs(log)


def compute_local_std_anomaly(test_gray, ref_mean, window=31):
    """
    Compute local standard deviation difference — captures texture anomalies.
    Defects often change the local texture variance.
    """
    test_f = test_gray.astype(np.float64)
    ref_f = ref_mean.astype(np.float64)
    
    # Local mean
    test_mean = cv2.blur(test_f, (window, window))
    ref_mean_local = cv2.blur(ref_f, (window, window))
    
    # Local variance
    test_var = cv2.blur(test_f ** 2, (window, window)) - test_mean ** 2
    ref_var = cv2.blur(ref_f ** 2, (window, window)) - ref_mean_local ** 2
    
    test_std = np.sqrt(np.clip(test_var, 0, None))
    ref_std_local = np.sqrt(np.clip(ref_var, 0, None))
    
    # Difference in local texture variance
    std_diff = np.abs(test_std - ref_std_local)
    
    return std_diff


# ============================================================================
# STEP 3: SEGMENTATION PIPELINE
# ============================================================================
def segment_defect(test_img_path, ref_model, defect_type, img_size=IMG_SIZE):
    """
    Main segmentation pipeline V2 for a single test image.

    Improvements over V1:
    - Percentile-based thresholding instead of pure statistical
    - Local texture variance analysis
    - Better multi-scale fusion
    - More aggressive morphological closing to connect nearby regions
    - Final dilation for better coverage of defect boundaries
    """
    params = DEFECT_PARAMS[defect_type]

    # Load test image
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img, img_size)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    test_color = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # --- Feature 1: Multi-scale grayscale difference from reference ---
    diff_map = compute_difference_map(
        test_gray, ref_model["mean_gray"], ref_model["std_gray"],
        gauss_ksize=params["gauss_ksize"]
    )
    diff_map_norm = cv2.normalize(diff_map, None, 0, 1, cv2.NORM_MINMAX)

    # --- Feature 2: Color anomaly in LAB space ---
    color_map = compute_color_anomaly(
        test_color, ref_model["mean_color"], ref_model["std_color"]
    )
    color_map_norm = cv2.normalize(color_map, None, 0, 1, cv2.NORM_MINMAX)

    # --- Feature 3: Gabor texture anomaly ---
    if params["use_gabor"]:
        test_gabor = compute_gabor_energy(test_gray.astype(np.uint8))
        gabor_diff = np.abs(test_gabor - ref_model["mean_gabor"])
        gabor_map_norm = cv2.normalize(gabor_diff, None, 0, 1, cv2.NORM_MINMAX)
    else:
        gabor_map_norm = np.zeros_like(diff_map_norm)

    # --- Feature 4: LoG edge response ---
    test_log = compute_log_edge_response(test_gray.astype(np.uint8))
    log_diff = np.abs(test_log - ref_model["mean_log"])
    log_map_norm = cv2.normalize(log_diff, None, 0, 1, cv2.NORM_MINMAX)

    # --- Feature 5: Local texture variance anomaly ---
    local_std_map = compute_local_std_anomaly(
        test_gray, ref_model["mean_gray"]
    )
    local_std_norm = cv2.normalize(local_std_map, None, 0, 1, cv2.NORM_MINMAX)

    # --- Fuse anomaly maps ---
    w_diff = params["diff_weight"]
    w_color = params["color_weight"]
    w_gabor = params["gabor_weight"]
    w_log = params["log_weight"]

    fused = (w_diff * diff_map_norm +
             w_color * color_map_norm +
             w_gabor * gabor_map_norm +
             w_log * log_map_norm)

    # Add local texture variance as a boosting factor
    # (multiply to enhance areas that are also anomalous in texture)
    boost = 1.0 + 0.5 * local_std_norm
    fused = fused * boost

    # Smooth the fused map
    smooth_k = max(5, params["gauss_ksize"] // 2 * 2 + 1)
    fused = cv2.GaussianBlur(fused, (smooth_k, smooth_k), 0)
    
    # Normalize to 0-255
    fused_255 = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- IMPROVED THRESHOLDING ---
    # Method 1: Percentile-based threshold (key improvement)
    threshold_value = np.percentile(fused_255, params["percentile_thresh"])
    percentile_mask = (fused_255 > threshold_value).astype(np.uint8) * 255

    # Method 2: Otsu's threshold
    _, otsu_mask = cv2.threshold(fused_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 3: Statistical threshold
    mean_val = np.mean(fused_255)
    std_val = np.std(fused_255)
    stat_thresh = mean_val + 2.0 * std_val
    stat_mask = (fused_255 > stat_thresh).astype(np.uint8) * 255

    # Combine: Union of percentile and intersection of Otsu+statistical
    otsu_stat = cv2.bitwise_and(otsu_mask, stat_mask)
    combined_mask = cv2.bitwise_or(percentile_mask, otsu_stat)

    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params["morph_kernel"], params["morph_kernel"])
    )

    # Close gaps first (connect nearby detections)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel,
        iterations=params["morph_close_iter"]
    )
    # Remove small noise
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, small_kernel,
        iterations=params["morph_open_iter"]
    )

    # --- Connected component filtering: remove tiny regions ---
    if params["min_area"] > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        clean_mask = np.zeros_like(combined_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= params["min_area"]:
                clean_mask[labels == i] = 255
        combined_mask = clean_mask

    # --- Final dilation to cover defect boundaries better ---
    if params["final_dilate"] > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params["final_dilate"], params["final_dilate"])
        )
        combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)

    # Final smoothing of mask edges
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    return combined_mask, fused_255, {
        "diff": diff_map_norm,
        "color": color_map_norm,
        "gabor": gabor_map_norm,
        "log": log_map_norm,
        "local_std": local_std_norm,
        "fused": fused_255,
    }


# ============================================================================
# STEP 4: EVALUATION METRICS
# ============================================================================
def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union."""
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient."""
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_pixel_accuracy(pred_mask, gt_mask):
    """Compute pixel-level accuracy."""
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    return np.mean(pred_bool == gt_bool)


def compute_precision_recall(pred_mask, gt_mask):
    """Compute precision and recall."""
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
def visualize_result(test_img_path, pred_mask, gt_mask, feature_maps,
                     defect_type, img_name, save_path=None):
    """
    Create a comprehensive visualization showing original, features,
    predicted mask, ground truth, and overlay comparison.
    """
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img, IMG_SIZE)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Leather Defect Segmentation — {defect_type.upper()} / {img_name}",
                 fontsize=16, fontweight='bold')

    # Row 1: Original, Diff, Color, Gabor, Local Std
    axes[0, 0].imshow(test_img_rgb)
    axes[0, 0].set_title("Original Image", fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(feature_maps["diff"], cmap='hot')
    axes[0, 1].set_title("Grayscale Diff Map", fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(feature_maps["color"], cmap='hot')
    axes[0, 2].set_title("Color Anomaly (LAB)", fontsize=11)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(feature_maps["gabor"], cmap='hot')
    axes[0, 3].set_title("Gabor Texture", fontsize=11)
    axes[0, 3].axis('off')

    axes[0, 4].imshow(feature_maps["local_std"], cmap='hot')
    axes[0, 4].set_title("Local Std Anomaly", fontsize=11)
    axes[0, 4].axis('off')

    # Row 2: LoG, Fused, Predicted, GT, Overlay
    axes[1, 0].imshow(feature_maps["log"], cmap='hot')
    axes[1, 0].set_title("LoG Edge Map", fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(feature_maps["fused"], cmap='hot')
    axes[1, 1].set_title("Fused Anomaly Map", fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_mask, cmap='gray')
    axes[1, 2].set_title("Predicted Mask", fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(gt_mask, cmap='gray')
    axes[1, 3].set_title("Ground Truth Mask", fontsize=11)
    axes[1, 3].axis('off')

    # Overlay: green = TP, red = FP, blue = FN
    overlay = test_img_rgb.copy()
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    tp = np.logical_and(pred_bool, gt_bool)
    fp = np.logical_and(pred_bool, ~gt_bool)
    fn = np.logical_and(~pred_bool, gt_bool)
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 0, 255]

    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)

    axes[1, 4].imshow(overlay)
    axes[1, 4].set_title(f"Overlay (IoU={iou:.3f}, Dice={dice:.3f})\n"
                          "Green=TP  Red=FP  Blue=FN", fontsize=10)
    axes[1, 4].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_side_by_side(test_img_path, pred_mask, gt_mask,
                            defect_type, img_name, save_path=None):
    """Simple side-by-side: Original | Predicted Mask | Ground Truth Mask"""
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img, IMG_SIZE)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{defect_type.upper()} — {img_name}", fontsize=14, fontweight='bold')

    axes[0].imshow(test_img_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Predicted Mask (CV)", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title("Ground Truth Mask", fontsize=12)
    axes[2].axis('off')

    # Overlay
    overlay = test_img_rgb.copy()
    pred_bool = pred_mask > 127
    gt_bool = gt_mask > 127
    tp = np.logical_and(pred_bool, gt_bool)
    fp = np.logical_and(pred_bool, ~gt_bool)
    fn = np.logical_and(~pred_bool, gt_bool)
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 0, 255]
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay", fontsize=12)
    axes[3].axis('off')

    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)
    fig.text(0.5, 0.02, f"IoU: {iou:.4f}  |  Dice: {dice:.4f}",
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# STEP 6: MAIN EXECUTION
# ============================================================================
def run_segmentation():
    """Run the full segmentation pipeline on all defect types."""
    ref_model = build_reference_model(TRAIN_GOOD_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for defect_type in DEFECT_TYPES:
        print(f"\n{'=' * 70}")
        print(f"Processing defect type: {defect_type.upper()}")
        print(f"{'=' * 70}")

        test_images = sorted(glob.glob(
            os.path.join(TEST_DIR, defect_type, "*.png")
        ))

        defect_results = []

        for img_path in test_images:
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            gt_path = os.path.join(GT_DIR, defect_type, f"{img_id}_mask.png")
            if not os.path.exists(gt_path):
                print(f"  WARNING: Ground truth not found for {img_name}, skipping")
                continue

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, IMG_SIZE)

            pred_mask, fused_map, feature_maps = segment_defect(
                img_path, ref_model, defect_type
            )

            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            acc = compute_pixel_accuracy(pred_mask, gt_mask)
            prec, recall = compute_precision_recall(pred_mask, gt_mask)

            defect_results.append({
                "image": img_name,
                "iou": iou,
                "dice": dice,
                "accuracy": acc,
                "precision": prec,
                "recall": recall,
            })

            print(f"  {img_name}: IoU={iou:.4f}, Dice={dice:.4f}, "
                  f"Prec={prec:.4f}, Recall={recall:.4f}")

            # Save detailed visualization for first 5 images per type
            if len(defect_results) <= 5:
                save_path = os.path.join(
                    OUTPUT_DIR, defect_type, f"{img_id}_detailed.png"
                )
                visualize_result(
                    img_path, pred_mask, gt_mask, feature_maps,
                    defect_type, img_name, save_path
                )

            # Save side-by-side comparison
            save_path = os.path.join(
                OUTPUT_DIR, defect_type, f"{img_id}_comparison.png"
            )
            visualize_side_by_side(
                img_path, pred_mask, gt_mask,
                defect_type, img_name, save_path
            )

            # Save predicted mask
            mask_save_path = os.path.join(
                OUTPUT_DIR, defect_type, f"{img_id}_pred_mask.png"
            )
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            cv2.imwrite(mask_save_path, pred_mask)

        all_results[defect_type] = defect_results

        if defect_results:
            mean_iou = np.mean([r["iou"] for r in defect_results])
            mean_dice = np.mean([r["dice"] for r in defect_results])
            mean_prec = np.mean([r["precision"] for r in defect_results])
            mean_recall = np.mean([r["recall"] for r in defect_results])
            print(f"\n  --- {defect_type.upper()} Summary ---")
            print(f"  Mean IoU:       {mean_iou:.4f}")
            print(f"  Mean Dice:      {mean_dice:.4f}")
            print(f"  Mean Precision: {mean_prec:.4f}")
            print(f"  Mean Recall:    {mean_recall:.4f}")

    # Overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Defect Type':<12} {'Images':>7} {'Mean IoU':>10} {'Mean Dice':>10} "
          f"{'Mean Prec':>10} {'Mean Recall':>11}")
    print("-" * 62)

    all_ious = []
    for defect_type in DEFECT_TYPES:
        results = all_results.get(defect_type, [])
        if results:
            m_iou = np.mean([r["iou"] for r in results])
            m_dice = np.mean([r["dice"] for r in results])
            m_prec = np.mean([r["precision"] for r in results])
            m_recall = np.mean([r["recall"] for r in results])
            all_ious.extend([r["iou"] for r in results])
            print(f"{defect_type:<12} {len(results):>7} {m_iou:>10.4f} "
                  f"{m_dice:>10.4f} {m_prec:>10.4f} {m_recall:>11.4f}")

    if all_ious:
        print("-" * 62)
        print(f"{'OVERALL':<12} {len(all_ious):>7} {np.mean(all_ious):>10.4f}")

    create_summary_chart(all_results)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    return all_results


def create_summary_chart(all_results):
    """Create summary bar chart."""
    defect_names = []
    mean_ious = []
    mean_dices = []
    mean_precs = []
    mean_recalls = []

    for dt in DEFECT_TYPES:
        results = all_results.get(dt, [])
        if results:
            defect_names.append(dt.upper())
            mean_ious.append(np.mean([r["iou"] for r in results]))
            mean_dices.append(np.mean([r["dice"] for r in results]))
            mean_precs.append(np.mean([r["precision"] for r in results]))
            mean_recalls.append(np.mean([r["recall"] for r in results]))

    if not defect_names:
        return

    x = np.arange(len(defect_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - 1.5*width, mean_ious, width, label='Mean IoU',
                    color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x - 0.5*width, mean_dices, width, label='Mean Dice',
                    color='#FF9800', edgecolor='white')
    bars3 = ax.bar(x + 0.5*width, mean_precs, width, label='Mean Precision',
                    color='#4CAF50', edgecolor='white')
    bars4 = ax.bar(x + 1.5*width, mean_recalls, width, label='Mean Recall',
                    color='#E91E63', edgecolor='white')

    ax.set_xlabel('Defect Type', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Leather Defect Segmentation — Pure CV Performance (V2)',
                  fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(defect_names, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "summary_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary chart saved to: {save_path}")


if __name__ == "__main__":
    results = run_segmentation()
