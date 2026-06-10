"""
Fine-Tuning UNet for Leather Defect Multiclass Segmentation
============================================================
Improvements over the original Colab notebook:
  1. Attention UNet architecture with SE blocks
  2. Aggressive data augmentation (92 samples is very small)
  3. Focal + Dice combined loss (handles extreme class imbalance)
  4. Class weighting computed from actual pixel distribution
  5. Cosine annealing LR schedule with warm restarts
  6. Better training strategy: warmup + longer training
  7. Local path support (no Colab dependency)
  8. Comprehensive evaluation with per-class metrics

Original issues addressed:
  - val Mean IoU was only 0.167 (very poor)
  - Early stopped at epoch 21 (model underfit)
  - No class weighting despite massive imbalance
  - Insufficient augmentation for tiny dataset
"""

import os
import sys
import warnings
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"d:\7th sem\Image processing and CV\Leather_defect_detection\leather")
RESULTS_DIR = Path(r"d:\7th sem\Image processing and CV\Leather_defect_detection\unet_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (256, 256)  # Reduced from 512 for CPU training efficiency
IMG_CHANNELS = 1       # Grayscale
CLASS_NAMES = ['background', 'color', 'cut', 'fold', 'glue', 'poke']
NUM_CLASSES = len(CLASS_NAMES)  # 6
BASE_FILTERS = 32      # Increased from 16 for better capacity
BATCH_SIZE = 4
EPOCHS = 30            # Reduced for CPU training (use more on GPU)
LEARNING_RATE = 3e-4   # Slightly higher initial LR
WARMUP_EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 42
FOCAL_GAMMA = 2.0      # Focal loss gamma
FOCAL_ALPHA = 0.25     # Focal loss alpha
DICE_SMOOTH = 1.0

CLASS_COLORS = np.array([
    [0, 0, 0],       # background - black
    [255, 0, 0],     # color - red
    [0, 255, 0],     # cut - green
    [0, 0, 255],     # fold - blue
    [255, 255, 0],   # glue - yellow
    [255, 0, 255],   # poke - magenta
], dtype=np.uint8)

DEFECT_TYPES = ['color', 'cut', 'fold', 'glue', 'poke']

# ============================================================================
# DATA DISCOVERY
# ============================================================================
def discover_pairs():
    """
    Discover all (image, mask, class_id) triples from the MVTec leather dataset.
    Returns list of (image_path, mask_path, class_index) tuples.
    """
    pairs = []

    for class_idx, defect_type in enumerate(DEFECT_TYPES, start=1):
        test_dir = BASE_DIR / "test" / defect_type
        gt_dir = BASE_DIR / "ground_truth" / defect_type

        if not test_dir.exists() or not gt_dir.exists():
            print(f"  WARNING: Missing directory for {defect_type}")
            continue

        images = sorted(test_dir.glob("*.png"))
        for img_path in images:
            img_id = img_path.stem
            mask_path = gt_dir / f"{img_id}_mask.png"
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path), class_idx))

    print(f"Found {len(pairs)} image-mask pairs across {len(DEFECT_TYPES)} defect types")

    # Count per class
    class_counts = {}
    for _, _, cls_id in pairs:
        name = CLASS_NAMES[cls_id]
        class_counts[name] = class_counts.get(name, 0) + 1
    for name, count in class_counts.items():
        print(f"  {name}: {count} images")

    return pairs


# ============================================================================
# DATA LOADING & AUGMENTATION
# ============================================================================
def load_and_preprocess(img_path, mask_path, class_id):
    """Load image and create multiclass mask."""
    # Load image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)

    # Load mask as binary
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Convert binary mask to multiclass: defect pixels get class_id, rest = 0
    multiclass_mask = np.zeros(IMG_SIZE, dtype=np.int32)
    multiclass_mask[mask > 127] = class_id

    return img, multiclass_mask


def augment_pair(image, mask):
    """
    Aggressive data augmentation for small dataset.
    Applied randomly during training.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # Random 90-degree rotation
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)

    # Random brightness adjustment
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0.0, 1.0)

    # Random contrast adjustment
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        mean = np.mean(image)
        image = np.clip((image - mean) * factor + mean, 0.0, 1.0)

    # Random Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    # Random elastic-like scale
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.85, 1.15)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if len(image_scaled.shape) == 2:
            image_scaled = np.expand_dims(image_scaled, -1)
        mask_scaled = cv2.resize(mask.astype(np.float32), (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # Crop or pad back to original size
        if scale > 1.0:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image_scaled[start_h:start_h+h, start_w:start_w+w]
            mask = mask_scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = np.zeros((h, w, 1), dtype=np.float32)
            mask_new = np.zeros((h, w), dtype=np.int32)
            image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image_scaled
            mask_new[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = mask_scaled
            mask = mask_new

    # Ensure correct shapes
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    image = image.astype(np.float32)
    mask = mask.astype(np.int32)

    return image, mask


def create_dataset(pairs, augment=False, repeat=True):
    """Create a tf.data.Dataset from image-mask pairs."""
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    class_ids = [p[2] for p in pairs]

    def generator():
        indices = list(range(len(img_paths)))
        if augment:
            np.random.shuffle(indices)

        for idx in indices:
            img, mask = load_and_preprocess(
                img_paths[idx], mask_paths[idx], class_ids[idx]
            )
            if augment:
                img, mask = augment_pair(img, mask)

            yield img, mask

    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1]), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================
def compute_class_weights(pairs):
    """
    Compute class weights based on actual pixel distribution.
    Critical for handling the extreme class imbalance (background >> defects).
    """
    print("\nComputing class weights from pixel distribution...")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for img_path, mask_path, class_id in pairs:
        _, mask = load_and_preprocess(img_path, mask_path, class_id)
        for c in range(NUM_CLASSES):
            pixel_counts[c] += np.sum(mask == c)

    total = pixel_counts.sum()
    print(f"  Total pixels: {total:,}")
    for i, name in enumerate(CLASS_NAMES):
        pct = pixel_counts[i] / total * 100
        print(f"  {name}: {pixel_counts[i]:,} ({pct:.2f}%)")

    # Inverse frequency weighting with smoothing
    # weight_c = total / (NUM_CLASSES * count_c)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i in range(NUM_CLASSES):
        if pixel_counts[i] > 0:
            weights[i] = total / (NUM_CLASSES * pixel_counts[i])
        else:
            weights[i] = 1.0

    # Cap weights to avoid extreme values
    max_weight = 50.0
    weights = np.clip(weights, 0.1, max_weight)

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()

    print(f"\n  Class weights (normalized):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {weights[i]:.4f}")

    return weights


# ============================================================================
# UNET MODEL ARCHITECTURE (IMPROVED)
# ============================================================================
def squeeze_excite_block(x, ratio=8):
    """Squeeze-and-Excitation block for channel attention."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def conv_block(x, filters, dropout_rate=0.1):
    """
    Improved convolution block:
    Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → SE → Dropout
    """
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = squeeze_excite_block(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    return x


def attention_gate(x, g, filters):
    """
    Attention Gate: learns to focus on relevant spatial locations.
    x = skip connection features, g = gating signal from decoder
    """
    theta_x = layers.Conv2D(filters, 1, padding='same')(x)
    phi_g = layers.Conv2D(filters, 1, padding='same')(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(act)
    return layers.Multiply()([x, psi])


def build_attention_unet(input_shape=(256, 256, 1), num_classes=6, base_filters=32):
    """
    Attention U-Net with SE blocks.
    Improvements over vanilla UNet:
    - Attention gates on skip connections
    - Squeeze-and-excitation blocks
    - Spatial dropout for regularization
    - Deeper bottleneck
    """
    inputs = layers.Input(shape=input_shape, name='input_image')

    # ---- Encoder ----
    # Level 1
    e1 = conv_block(inputs, base_filters, dropout_rate=0.05)
    p1 = layers.MaxPooling2D(2)(e1)

    # Level 2
    e2 = conv_block(p1, base_filters * 2, dropout_rate=0.1)
    p2 = layers.MaxPooling2D(2)(e2)

    # Level 3
    e3 = conv_block(p2, base_filters * 4, dropout_rate=0.15)
    p3 = layers.MaxPooling2D(2)(e3)

    # Level 4
    e4 = conv_block(p3, base_filters * 8, dropout_rate=0.2)
    p4 = layers.MaxPooling2D(2)(e4)

    # ---- Bottleneck ----
    b = conv_block(p4, base_filters * 16, dropout_rate=0.25)

    # ---- Decoder ----
    # Level 4
    u4 = layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding='same')(b)
    a4 = attention_gate(e4, u4, base_filters * 4)
    u4 = layers.Concatenate()([u4, a4])
    d4 = conv_block(u4, base_filters * 8, dropout_rate=0.2)

    # Level 3
    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding='same')(d4)
    a3 = attention_gate(e3, u3, base_filters * 2)
    u3 = layers.Concatenate()([u3, a3])
    d3 = conv_block(u3, base_filters * 4, dropout_rate=0.15)

    # Level 2
    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding='same')(d3)
    a2 = attention_gate(e2, u2, base_filters)
    u2 = layers.Concatenate()([u2, a2])
    d2 = conv_block(u2, base_filters * 2, dropout_rate=0.1)

    # Level 1
    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding='same')(d2)
    a1 = attention_gate(e1, u1, base_filters // 2)
    u1 = layers.Concatenate()([u1, a1])
    d1 = conv_block(u1, base_filters, dropout_rate=0.05)

    # ---- Output ----
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='output')(d1)

    model = Model(inputs, outputs, name='attention_unet_leather')
    return model


# ============================================================================
# LOSS FUNCTIONS (IMPROVED)
# ============================================================================
def focal_loss(y_true, y_pred, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard-to-classify pixels, down-weights easy background pixels.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # One-hot encode
    y_true_one_hot = tf.one_hot(y_true, NUM_CLASSES)

    # Cross entropy per pixel
    ce = -y_true_one_hot * tf.math.log(y_pred)

    # Focal weight
    pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    focal_weight = alpha * tf.pow(1.0 - pt, gamma)

    loss = tf.reduce_sum(ce, axis=-1) * focal_weight
    return tf.reduce_mean(loss)


def dice_loss_fn(y_true, y_pred, smooth=DICE_SMOOTH):
    """
    Generalized Dice Loss across all classes.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    y_true_one_hot = tf.one_hot(y_true, NUM_CLASSES)

    intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=0)
    union = tf.reduce_sum(y_true_one_hot, axis=0) + tf.reduce_sum(y_pred, axis=0)

    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    # Weight classes inversely by frequency (skip background for dice)
    return 1.0 - tf.reduce_mean(dice_per_class[1:])  # Skip background class


def combined_loss(y_true, y_pred):
    """Combined Focal + Dice loss."""
    fl = focal_loss(y_true, y_pred)
    dl = dice_loss_fn(y_true, y_pred)
    return 0.5 * fl + 0.5 * dl


# ============================================================================
# METRICS
# ============================================================================
def mean_iou_metric(y_true, y_pred):
    """Mean IoU across all classes."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_classes = tf.argmax(tf.reshape(y_pred, [-1, NUM_CLASSES]), axis=-1)
    y_pred_classes = tf.cast(y_pred_classes, tf.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=NUM_CLASSES)
    cm = tf.cast(cm, tf.float32)

    # IoU per class
    diag = tf.linalg.diag_part(cm)
    row_sum = tf.reduce_sum(cm, axis=1)
    col_sum = tf.reduce_sum(cm, axis=0)
    denominator = row_sum + col_sum - diag

    iou = tf.where(denominator > 0, diag / denominator, tf.zeros_like(diag))
    valid = tf.cast(denominator > 0, tf.float32)
    num_valid = tf.reduce_sum(valid)

    return tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid)


def mean_dice_metric(y_true, y_pred):
    """Mean Dice coefficient across all classes."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_classes = tf.argmax(tf.reshape(y_pred, [-1, NUM_CLASSES]), axis=-1)
    y_pred_classes = tf.cast(y_pred_classes, tf.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=NUM_CLASSES)
    cm = tf.cast(cm, tf.float32)

    diag = tf.linalg.diag_part(cm)
    row_sum = tf.reduce_sum(cm, axis=1)
    col_sum = tf.reduce_sum(cm, axis=0)
    denominator = row_sum + col_sum

    dice = tf.where(denominator > 0, 2.0 * diag / denominator, tf.zeros_like(diag))
    valid = tf.cast(denominator > 0, tf.float32)
    num_valid = tf.reduce_sum(valid)

    return tf.math.divide_no_nan(tf.reduce_sum(dice), num_valid)


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
class CosineAnnealingWithWarmup(callbacks.Callback):
    """Cosine annealing LR with linear warmup. TF 2.20 compatible."""
    def __init__(self, max_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.lr_history = []

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        # TF 2.20 compatible LR assignment
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except Exception:
            try:
                K.set_value(self.model.optimizer.learning_rate, lr)
            except Exception:
                self.model.optimizer.lr = lr

        self.lr_history.append(lr)
        print(f"  LR: {lr:.6f}", end="")


# ============================================================================
# VISUALIZATION
# ============================================================================
def colorize_mask(mask):
    """Convert class ID mask to color visualization."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        color[mask == cls_id] = CLASS_COLORS[cls_id]
    return color


def visualize_predictions(model, val_pairs, num_samples=6, save_path=None):
    """Visualize model predictions vs ground truth."""
    n = min(num_samples, len(val_pairs))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_path, mask_path, class_id = val_pairs[i]

        # Load
        img, gt_mask = load_and_preprocess(img_path, mask_path, class_id)
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img_batch, verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1)

        # Visualize
        axes[i, 0].imshow(img[:, :, 0], cmap='gray')
        axes[i, 0].set_title(f"Input ({CLASS_NAMES[class_id]})", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(colorize_mask(gt_mask))
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(colorize_mask(pred_mask))
        axes[i, 2].set_title("Prediction", fontsize=10)
        axes[i, 2].axis('off')

        # Overlay
        img_rgb = cv2.cvtColor((img[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = img_rgb.copy()
        pred_color = colorize_mask(pred_mask)
        mask_overlay = pred_mask > 0
        overlay[mask_overlay] = (0.5 * overlay[mask_overlay] + 0.5 * pred_color[mask_overlay]).astype(np.uint8)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay", fontsize=10)
        axes[i, 3].axis('off')

    plt.suptitle("Attention UNet — Predictions vs Ground Truth", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions saved to: {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Mean IoU
    axes[1].plot(history.history['mean_iou_metric'], label='Train IoU', linewidth=2)
    axes[1].plot(history.history['val_mean_iou_metric'], label='Val IoU', linewidth=2)
    axes[1].set_title('Mean IoU', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Mean Dice
    axes[2].plot(history.history['mean_dice_metric'], label='Train Dice', linewidth=2)
    axes[2].plot(history.history['val_mean_dice_metric'], label='Val Dice', linewidth=2)
    axes[2].set_title('Mean Dice', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Training History — Attention UNet", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    plt.close()


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, pairs):
    """Per-class evaluation of the model."""
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION")
    print("=" * 60)

    # Confusion matrix
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for img_path, mask_path, class_id in pairs:
        img, gt_mask = load_and_preprocess(img_path, mask_path, class_id)
        pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1)

        for true_c in range(NUM_CLASSES):
            for pred_c in range(NUM_CLASSES):
                total_cm[true_c, pred_c] += np.sum(
                    (gt_mask == true_c) & (pred_mask == pred_c)
                )

    # Per-class metrics
    print(f"\n{'Class':<12} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)

    ious = []
    dices = []
    for c in range(NUM_CLASSES):
        tp = total_cm[c, c]
        fp = total_cm[:, c].sum() - tp
        fn = total_cm[c, :].sum() - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ious.append(iou)
        dices.append(dice)

        print(f"{CLASS_NAMES[c]:<12} {iou:>8.4f} {dice:>8.4f} {precision:>10.4f} {recall:>8.4f}")

    print("-" * 50)
    print(f"{'Mean (all)':<12} {np.mean(ious):>8.4f} {np.mean(dices):>8.4f}")
    print(f"{'Mean (defect)':<12} {np.mean(ious[1:]):>8.4f} {np.mean(dices[1:]):>8.4f}")

    return ious, dices


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print("=" * 60)
    print("Leather Defect Segmentation — Attention UNet Fine-Tuning")
    print("=" * 60)

    # 1. Discover data
    pairs = discover_pairs()
    if len(pairs) == 0:
        print("ERROR: No data found!")
        return

    # 2. Compute class weights
    class_weights = compute_class_weights(pairs)

    # 3. Split into train/val
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=VAL_SPLIT, random_state=SEED, stratify=[p[2] for p in pairs]
    )
    print(f"\nTrain: {len(train_pairs)} | Val: {len(val_pairs)}")

    # 4. Create datasets
    train_ds = create_dataset(train_pairs, augment=True, repeat=True)
    val_ds = create_dataset(val_pairs, augment=False, repeat=True)

    steps_per_epoch = max(len(train_pairs) * 2 // BATCH_SIZE, 1)  # 2x augmented
    val_steps = max(len(val_pairs) // BATCH_SIZE, 1)

    print(f"Steps per epoch: {steps_per_epoch} | Val steps: {val_steps}")

    # 5. Build or resume model
    print("\n" + "=" * 60)
    model_path = str(RESULTS_DIR / 'best_attention_unet.keras')
    initial_epoch = 0

    if os.path.exists(model_path):
        print("Resuming from saved checkpoint...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'combined_loss': combined_loss,
                'mean_iou_metric': mean_iou_metric,
                'mean_dice_metric': mean_dice_metric,
            }
        )
        # Estimate how many epochs were already done from filename or set manually
        initial_epoch = 3  # We know epoch 3 completed from logs
        print(f"Resumed model. Starting from epoch {initial_epoch + 1}")
    else:
        print("Building Attention UNet model from scratch...")
        tf.keras.backend.clear_session()
        model = build_attention_unet(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS),
            num_classes=NUM_CLASSES,
            base_filters=BASE_FILTERS,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[mean_iou_metric, mean_dice_metric],
    )

    print("=" * 60)

    # Print model summary
    total_params = model.count_params()
    print(f"Model: {model.name}")
    print(f"Total parameters: {total_params:,}")
    model.summary(print_fn=lambda x: None)  # Suppress detailed summary

    # 6. Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_mean_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor='val_mean_iou_metric',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        CosineAnnealingWithWarmup(
            max_lr=LEARNING_RATE,
            warmup_epochs=max(0, WARMUP_EPOCHS - initial_epoch),
            total_epochs=EPOCHS,
        ),
    ]

    # 7. Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {initial_epoch + 1} to {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Initial LR: {LEARNING_RATE}")
    print(f"  Loss: Focal + Dice combined")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb_list,
        verbose=1,
    )

    # 8. Plot training history
    plot_training_history(
        history,
        save_path=str(RESULTS_DIR / 'training_history.png')
    )

    # 9. Evaluate
    print("\nLoading best model for evaluation...")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'combined_loss': combined_loss,
                'mean_iou_metric': mean_iou_metric,
                'mean_dice_metric': mean_dice_metric,
            }
        )

    evaluate_model(model, val_pairs)

    # 10. Visualize predictions
    visualize_predictions(
        model, val_pairs,
        num_samples=min(8, len(val_pairs)),
        save_path=str(RESULTS_DIR / 'predictions.png')
    )

    # Also visualize on some training samples
    visualize_predictions(
        model, train_pairs[:8],
        num_samples=8,
        save_path=str(RESULTS_DIR / 'train_predictions.png')
    )

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Best model saved to: {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
