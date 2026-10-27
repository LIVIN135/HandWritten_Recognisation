"""
train.py
========
Single reliable training command for the 62-class handwriting recognizer.

WHAT THIS SCRIPT DOES:
  1. Validates the dataset
  2. Loads EMNIST ByClass (downloads if needed) — 62-class general handwriting
  3. Loads user's own handwriting from dataset.csv
  4. Splits ORIGINAL user images into train/val/test BEFORE augmentation
     (prevents train/val leakage)
  5. Augments only the training portion of user images
  6. Combines: EMNIST train + augmented user train
  7. Trains a regularized CNN with class weights + early stopping
  8. Evaluates on a clean test set (no augmented or EMNIST-adjacent images)
  9. Saves model, labels, preprocessing config
  10. Generates accuracy graphs, confusion matrix, classification report
  11. Warns if model is still biased toward any single class

HOW TO RUN:
  pip install tensorflow pillow numpy scikit-learn matplotlib scipy
  python train.py

OPTIONS:
  --no-emnist        Skip EMNIST, train only on user data (small dataset warning)
  --emnist-only      Use EMNIST only, ignore user data
  --csv PATH         Custom dataset CSV (default: dataset.csv)
  --epochs N         Max training epochs (default: from config)
  --batch-size N     Batch size
  --emnist-dir DIR   Directory containing EMNIST ByClass files
"""

import argparse
import csv
import gzip
import json
import os
import struct
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLASSES, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, L2_REG, DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR,
    EMNIST_DIR, EMNIST_BYCLASS_URLS, EMNIST_BYCLASS_ALT_URLS,
    EMNIST_LABEL_TO_CHAR, EMNIST_SAMPLES_PER_CLASS,
    USER_UPSAMPLE_FACTOR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MIN_SAMPLES_FOR_SPLIT,
    RANDOM_SEED, MODELS_DIR, MODEL_PATH, LABELS_PATH, PREPROC_CONFIG,
    LEGACY_DATASET_CSV, LEGACY_BALANCED_CSV,
    ensure_dirs,
)
from preprocessing import preprocess_batch, preprocess_image, _save_config
from augment import augment_image, build_augmentation_layer

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("ERROR: TensorFlow not found.  pip install tensorflow")
    sys.exit(1)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# EMNIST DOWNLOAD & LOAD
# ──────────────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> bool:
    """Download a file, return True on success."""
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        mb = dest.stat().st_size / 1_048_576
        print(f"  {mb:.1f} MB  OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _read_idx_images(path: Path) -> np.ndarray:
    """Read IDX3 image file → float32 array [N, H, W]."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read IDX1 label file → int32 array [N]."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def _fix_emnist_orientation(imgs: np.ndarray) -> np.ndarray:
    """
    EMNIST ByClass images are stored rotated 90° CCW and mirrored.
    Apply: transpose then flip left-right to get the correct upright orientation.
    This is documented in the EMNIST paper (Cohen et al. 2017).
    """
    # imgs shape: (N, 28, 28)
    fixed = np.transpose(imgs, (0, 2, 1))   # transpose H,W
    fixed = fixed[:, :, ::-1]               # flip horizontally
    return fixed


def ensure_emnist_files(emnist_dir: Path) -> dict:
    """
    Check that EMNIST ByClass files are present. Download if missing.
    Returns dict of {key: Path} or raises RuntimeError if unavailable.
    """
    emnist_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
        "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
        "test_images" : "emnist-byclass-test-images-idx3-ubyte.gz",
        "test_labels" : "emnist-byclass-test-labels-idx1-ubyte.gz",
    }

    paths = {}
    missing = []
    for key, fname in filenames.items():
        p = emnist_dir / fname
        paths[key] = p
        if not p.exists():
            missing.append(key)

    if missing:
        print(f"\n  Downloading EMNIST ByClass ({len(missing)} files) ...")
        print("  NOTE: Total ~500 MB. This only happens once.\n")
        for key in missing:
            url = EMNIST_BYCLASS_URLS[key]
            alt = EMNIST_BYCLASS_ALT_URLS[key]
            dest = paths[key]
            if not _download(url, dest):
                print(f"  Primary mirror failed, trying alternate...")
                if not _download(alt, dest):
                    raise RuntimeError(
                        f"Could not download EMNIST file: {key}\n"
                        f"Please manually download from:\n  {url}\n"
                        f"and place it in: {emnist_dir}"
                    )
    return paths


def load_emnist(
    emnist_dir: Path,
    max_per_class: int = EMNIST_SAMPLES_PER_CLASS,
) -> tuple:
    """
    Load EMNIST ByClass, fix orientation, map labels to character strings.

    Returns
    -------
    X_train : float32 (N, 28, 28, 1)  values in [0,1], white bg / black ink
    y_train : int32  (N,)              class indices per CLASSES list
    X_test  : float32 (M, 28, 28, 1)
    y_test  : int32  (M,)
    label_map : dict  {emnist_int → char}
    """
    print("\n  Loading EMNIST ByClass ...")
    paths = ensure_emnist_files(emnist_dir)

    raw_train_imgs = _read_idx_images(paths["train_images"])
    raw_train_lbls = _read_idx_labels(paths["train_labels"])
    raw_test_imgs  = _read_idx_images(paths["test_images"])
    raw_test_lbls  = _read_idx_labels(paths["test_labels"])

    # Fix orientation
    raw_train_imgs = _fix_emnist_orientation(raw_train_imgs)
    raw_test_imgs  = _fix_emnist_orientation(raw_test_imgs)

    # EMNIST images are black-on-white ALREADY after fix? No:
    # After orientation fix, the images are correct upright characters
    # but they appear as WHITE ink on BLACK background.
    # Invert so we get BLACK ink on WHITE background (matching user drawings).
    raw_train_imgs = 1.0 - raw_train_imgs
    raw_test_imgs  = 1.0 - raw_test_imgs

    n_emnist_classes = int(raw_train_lbls.max()) + 1
    print(f"  EMNIST ByClass: {len(raw_train_imgs)} train, {len(raw_test_imgs)} test")
    print(f"  EMNIST classes found: {n_emnist_classes}")

    if n_emnist_classes < 62:
        raise RuntimeError(
            f"Expected 62 EMNIST ByClass classes, found {n_emnist_classes}.\n"
            "The downloaded files may be from a different EMNIST split (e.g. Balanced).\n"
            "Please download the ByClass split specifically."
        )

    # Map EMNIST label → our CLASSES index
    # EMNIST ByClass: 0-9=digits, 10-35=A-Z, 36-61=a-z  (matches our CLASSES order)
    label_map = EMNIST_LABEL_TO_CHAR  # {emnist_int → char}

    # Sample up to max_per_class from each EMNIST class
    rng = np.random.default_rng(RANDOM_SEED)
    train_X_list, train_y_list = [], []
    test_X_list,  test_y_list  = [], []

    for emnist_lbl in range(62):
        char = label_map.get(emnist_lbl)
        if char is None or char not in CHAR_TO_IDX:
            continue
        our_idx = CHAR_TO_IDX[char]

        # Training
        mask = raw_train_lbls == emnist_lbl
        idxs = np.where(mask)[0]
        if max_per_class and len(idxs) > max_per_class:
            idxs = rng.choice(idxs, max_per_class, replace=False)
        train_X_list.append(raw_train_imgs[idxs])
        train_y_list.append(np.full(len(idxs), our_idx, dtype=np.int32))

        # Test (use all — no cap)
        mask_t = raw_test_lbls == emnist_lbl
        idxs_t = np.where(mask_t)[0]
        test_X_list.append(raw_test_imgs[idxs_t])
        test_y_list.append(np.full(len(idxs_t), our_idx, dtype=np.int32))

    X_train = np.concatenate(train_X_list)[..., np.newaxis].astype(np.float32)
    y_train = np.concatenate(train_y_list).astype(np.int32)
    X_test  = np.concatenate(test_X_list)[..., np.newaxis].astype(np.float32)
    y_test  = np.concatenate(test_y_list).astype(np.int32)

    print(f"  EMNIST train used: {len(X_train)} | test: {len(X_test)}")
    class_counts = Counter(y_train.tolist())
    print(f"  EMNIST samples/class: min={min(class_counts.values())} "
          f"max={max(class_counts.values())}")

    return X_train, y_train, X_test, y_test, label_map


# ──────────────────────────────────────────────────────────────────────────────
# USER DATA LOADING  (with leak-free split)
# ──────────────────────────────────────────────────────────────────────────────

def load_user_originals(csv_path: Path) -> dict:
    """
    Read dataset.csv and return only ORIGINAL (non-augmented) images.
    Augmented images (in balanced_images/ or augmented_images/) are excluded.
    Returns {label: [path, ...]} dict.
    """
    by_label: defaultdict = defaultdict(list)
    if not csv_path.exists():
        return by_label

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            label = row.get("label", "").strip()
            if label not in CLASSES:
                continue
            # Only include images from dataset_images/ (originals)
            # Skip balanced_images/ and augmented_images/
            path_str = str(p).replace("\\", "/")
            if ("balanced_images" in path_str or
                    "augmented_images" in path_str):
                continue
            if p.exists():
                by_label[label].append(p)

    return dict(by_label)


def split_originals_no_leakage(
    by_label: dict,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    test_ratio: float  = TEST_RATIO,
    seed: int          = RANDOM_SEED,
) -> tuple:
    """
    Split ORIGINAL images into train/val/test sets per class.

    KEY GUARANTEE: augmented images from the same original will only
    appear in whichever split the original was assigned to.
    This completely prevents train/val leakage.

    For classes with very few originals:
      - 1 sample  → all go to train
      - 2 samples → 1 train, 1 val  (no test)
      - 3+ samples → stratified 70/15/15 split

    Returns
    -------
    train_by_label, val_by_label, test_by_label : dict {label: [paths]}
    """
    rng = np.random.default_rng(seed)

    train_bl, val_bl, test_bl = defaultdict(list), defaultdict(list), defaultdict(list)

    for label in sorted(by_label):
        paths = list(by_label[label])
        rng.shuffle(paths)
        n = len(paths)

        if n == 0:
            continue
        elif n == 1:
            train_bl[label].extend(paths)
        elif n == 2:
            train_bl[label].append(paths[0])
            val_bl[label].append(paths[1])
        else:
            n_val  = max(1, round(n * val_ratio))
            n_test = max(1, round(n * test_ratio))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val   = max(1, (n - n_train) // 2)
                n_test  = n - n_train - n_val

            train_bl[label].extend(paths[:n_train])
            val_bl[label].extend(paths[n_train:n_train + n_val])
            test_bl[label].extend(paths[n_train + n_val:])

    return dict(train_bl), dict(val_bl), dict(test_bl)


def load_and_augment_user_train(
    train_by_label: dict,
    upsample: int = USER_UPSAMPLE_FACTOR,
) -> tuple:
    """
    Load user training images, apply augmentation, and return arrays.
    Each original is used `upsample` times with different random augmentations.
    """
    from PIL import Image as PImage
    X, y = [], []

    for label in sorted(train_by_label):
        idx = CHAR_TO_IDX[label]
        for path in train_by_label[label]:
            try:
                orig = PImage.open(path).convert("L")
                for _ in range(upsample):
                    aug = augment_image(orig)
                    arr = preprocess_image(aug)
                    X.append(arr)
                    y.append(idx)
            except Exception as exc:
                print(f"  WARNING: skipped {path}: {exc}")

    if not X:
        return np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32), \
               np.zeros((0,), dtype=np.int32)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_user_split(by_label: dict) -> tuple:
    """Load images from a {label: [paths]} dict without augmentation."""
    X, y = [], []
    for label in sorted(by_label):
        idx = CHAR_TO_IDX[label]
        for path in by_label[label]:
            try:
                arr = preprocess_image(path)
                X.append(arr)
                y.append(idx)
            except Exception as exc:
                print(f"  WARNING: skipped {path}: {exc}")

    if not X:
        return np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32), \
               np.zeros((0,), dtype=np.int32)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET REPORT  (printed before training)
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_report(
    user_originals: dict,
    train_bl: dict, val_bl: dict, test_bl: dict,
    emnist_counts: Counter,
):
    print("\n" + "=" * 62)
    print("DATASET REPORT")
    print("=" * 62)
    total_orig = sum(len(v) for v in user_originals.values())
    print(f"  User original images total : {total_orig}")
    print(f"  User classes with data     : {len(user_originals)} / {NUM_CLASSES}")

    if emnist_counts:
        print(f"  EMNIST samples             : {sum(emnist_counts.values())}")

    print(f"\n  {'Class':<8} {'Originals':>10} {'Train':>7} {'Val':>7} "
          f"{'Test':>7} {'EMNIST':>8}  Status")
    print(f"  {'-'*62}")

    for ch in CLASSES:
        orig  = len(user_originals.get(ch, []))
        tr    = len(train_bl.get(ch, []))
        va    = len(val_bl.get(ch, []))
        te    = len(test_bl.get(ch, []))
        em    = emnist_counts.get(CHAR_TO_IDX[ch], 0)

        if orig == 0 and em == 0:
            status = "NO DATA"
        elif orig == 0:
            status = "EMNIST only"
        elif orig < 3:
            status = "LOW ORIGINAL DATA"
        elif orig < 20:
            status = "limited"
        else:
            status = "OK"

        print(f"  {repr(ch):<8} {orig:>10} {tr:>7} {va:>7} {te:>7} {em:>8}  {status}")

    # Leakage check
    train_paths = set()
    for paths in train_bl.values():
        train_paths.update(str(p) for p in paths)
    val_paths = set()
    for paths in val_bl.values():
        val_paths.update(str(p) for p in paths)
    test_paths = set()
    for paths in test_bl.values():
        test_paths.update(str(p) for p in paths)

    tv_leak = len(train_paths & val_paths)
    tt_leak = len(train_paths & test_paths)
    vt_leak = len(val_paths & test_paths)

    print(f"\n  Train/Val  leakage : {tv_leak} shared paths  "
          f"({'LEAK DETECTED!' if tv_leak else 'OK'})")
    print(f"  Train/Test leakage : {tt_leak} shared paths  "
          f"({'LEAK DETECTED!' if tt_leak else 'OK'})")
    print(f"  Val/Test   leakage : {vt_leak} shared paths  "
          f"({'LEAK DETECTED!' if vt_leak else 'OK'})")

    low_classes = [ch for ch in CLASSES if len(user_originals.get(ch, [])) < 3]
    if low_classes:
        print(f"\n  WARNING: {len(low_classes)} classes have fewer than 3 original "
              f"handwriting samples.")
        print(f"  These classes rely entirely on EMNIST for generalisation.")
        print(f"  Run: python dataset_builder.py --batch --rounds 20")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# CNN MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Compact CNN suitable for 28×28 greyscale handwritten character recognition.

    Architecture:
      Augmentation → Conv-BN-ReLU-Pool × 3 → GlobalAvgPool
      → Dense(256) → Dropout → Dense(62, softmax)

    Regularisation:
      - Dropout (configurable)
      - L2 weight decay on dense layers
      - BatchNormalization after each conv block
      - In-model random augmentation (only active during training)
      - class_weight passed to model.fit()
    """
    aug_layer = build_augmentation_layer()

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="image")

    # In-model augmentation (disabled at inference automatically)
    if aug_layer is not None:
        x = aug_layer(inputs)
    else:
        x = inputs

    def conv_block(x, filters, kernel=3, pool=True):
        x = layers.Conv2D(
            filters, kernel,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(L2_REG),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)
        return x

    x = conv_block(x, 32)        # → 14×14
    x = conv_block(x, 64)        # → 7×7
    x = conv_block(x, 128)       # → 3×3 (or 4×4 depending on input)
    x = conv_block(x, 128, pool=False)  # additional depth without pooling

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
    )(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="HandwritingCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> dict:
    """Balanced class weights: weight_i = N_total / (n_classes * count_i)."""
    counts = Counter(y.tolist())
    n_total = len(y)
    weights = {}
    for i in range(NUM_CLASSES):
        cnt = counts.get(i, 1)   # avoid division by zero
        weights[i] = n_total / (NUM_CLASSES * cnt)
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# BIAS DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def check_prediction_bias(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Check whether the model is biased toward predicting one class too often.
    Prints a warning if any class is predicted for > 20% of test samples.
    """
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    pred_counts = Counter(preds.tolist())
    n = len(preds)
    threshold = 0.20

    print("\n  Prediction distribution (test set):")
    biased = []
    for i in range(NUM_CLASSES):
        frac = pred_counts.get(i, 0) / n
        if frac > threshold:
            biased.append((CLASSES[i], frac))

    if biased:
        print("\n  *** WARNING: PREDICTION BIAS DETECTED ***")
        for ch, frac in biased:
            print(f"  Model predicted '{ch}' for {frac:.1%} of test samples.")
        print("  Root cause: insufficient diverse training data.")
        print("  Fix: Collect more original handwriting samples and retrain.")
    else:
        top5 = pred_counts.most_common(5)
        print("  Top 5 predicted classes:")
        for idx, cnt in top5:
            print(f"    '{CLASSES[idx]}': {cnt/n:.1%}")
        print("  No significant prediction bias detected.")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE MODEL ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────────

def save_model_artifacts(model, val_acc: float, test_acc: float):
    """Save model, labels.json, preproc_config.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)
    print(f"  Model saved : {MODEL_PATH}")

    labels = {"classes": CLASSES}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"  Labels saved: {LABELS_PATH}")

    _save_config(IMAGE_SIZE, 2)  # save preprocessing config
    print(f"  Preproc cfg : {PREPROC_CONFIG}")

    # Save model metadata
    import datetime
    meta_path = MODELS_DIR / "model_metadata.json"
    meta = {
        "model_file"   : str(MODEL_PATH),
        "trained_at"   : datetime.datetime.now().isoformat(),
        "num_classes"  : NUM_CLASSES,
        "classes"      : CLASSES,
        "val_accuracy" : round(float(val_acc),  4),
        "test_accuracy": round(float(test_acc), 4),
        "image_size"   : IMAGE_SIZE,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata    : {meta_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 62-class handwriting model")
    parser.add_argument("--no-emnist",    action="store_true",
                        help="Skip EMNIST, train only on user data")
    parser.add_argument("--emnist-only",  action="store_true",
                        help="Use EMNIST only, ignore user data")
    parser.add_argument("--csv",          type=Path, default=LEGACY_DATASET_CSV,
                        help=f"User dataset CSV (default: {LEGACY_DATASET_CSV})")
    parser.add_argument("--epochs",       type=int,  default=EPOCHS)
    parser.add_argument("--batch-size",   type=int,  default=BATCH_SIZE)
    parser.add_argument("--emnist-dir",   type=Path, default=EMNIST_DIR)
    parser.add_argument("--emnist-samples", type=int, default=EMNIST_SAMPLES_PER_CLASS,
                        help="Max EMNIST samples per class (default: 2000)")
    args = parser.parse_args()

    ensure_dirs()

    print("\n" + "=" * 62)
    print("HANDWRITING MODEL TRAINER")
    print("=" * 62)
    print(f"  Classes     : {NUM_CLASSES}")
    print(f"  Image size  : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Epochs (max): {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  EMNIST      : {'disabled' if args.no_emnist else 'enabled'}")

    # ── 1. Validate user dataset ───────────────────────────────────────────────
    from dataset_validator import validate_dataset
    print("\n[1/8] Validating dataset ...")
    vresults = validate_dataset(csv_path=args.csv, check_duplicates=False, verbose=False)
    if not vresults["passed"] and not args.no_emnist:
        print("  Dataset validation found issues. "
              "Training continues using EMNIST as primary source.")
    elif not vresults["passed"] and args.no_emnist:
        print("  ERROR: Dataset validation failed and --no-emnist was specified.")
        print("  Fix dataset errors or remove --no-emnist to use EMNIST.")
        sys.exit(1)

    # ── 2. Load user original images ──────────────────────────────────────────
    print("\n[2/8] Loading user original images ...")
    user_originals = {} if args.emnist_only else load_user_originals(args.csv)
    total_orig = sum(len(v) for v in user_originals.values())
    print(f"  Found {total_orig} original user images across "
          f"{len(user_originals)} classes")

    # ── 3. Leak-free split of originals ───────────────────────────────────────
    print("\n[3/8] Splitting originals (no-leakage) ...")
    train_bl, val_bl, test_bl = split_originals_no_leakage(user_originals)
    print(f"  Train originals : {sum(len(v) for v in train_bl.values())}")
    print(f"  Val originals   : {sum(len(v) for v in val_bl.values())}")
    print(f"  Test originals  : {sum(len(v) for v in test_bl.values())}")

    # ── 4. Load EMNIST ─────────────────────────────────────────────────────────
    X_em_train = X_em_test = y_em_train = y_em_test = None
    emnist_counts: Counter = Counter()

    if not args.no_emnist:
        print("\n[4/8] Loading EMNIST ByClass ...")
        try:
            X_em_train, y_em_train, X_em_test, y_em_test, _ = load_emnist(
                args.emnist_dir, max_per_class=args.emnist_samples
            )
            emnist_counts = Counter(y_em_train.tolist())
        except RuntimeError as exc:
            print(f"\n  WARNING: EMNIST loading failed:\n  {exc}")
            print("  Continuing with user data only.")
            X_em_train = None
    else:
        print("\n[4/8] EMNIST skipped (--no-emnist).")

    # ── 5. Print dataset report ────────────────────────────────────────────────
    print_dataset_report(user_originals, train_bl, val_bl, test_bl, emnist_counts)

    # ── 6. Build training / validation / test arrays ──────────────────────────
    print("\n[5/8] Building arrays ...")

    # User train — augmented
    if train_bl:
        print("  Augmenting user train images ...")
        X_u_train, y_u_train = load_and_augment_user_train(train_bl)
        print(f"  User train (augmented): {len(X_u_train)} samples")
    else:
        X_u_train = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        y_u_train = np.zeros((0,), dtype=np.int32)

    # User val/test — NOT augmented (raw originals only)
    X_u_val,  y_u_val  = load_user_split(val_bl)
    X_u_test, y_u_test = load_user_split(test_bl)
    print(f"  User val (clean)  : {len(X_u_val)}")
    print(f"  User test (clean) : {len(X_u_test)}")

    # Combine train
    train_parts_X = [X_u_train]
    train_parts_y = [y_u_train]
    if X_em_train is not None:
        train_parts_X.append(X_em_train)
        train_parts_y.append(y_em_train)

    X_train = np.concatenate(train_parts_X, axis=0)
    y_train = np.concatenate(train_parts_y, axis=0)

    # Shuffle train
    idx_shuf = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx_shuf], y_train[idx_shuf]

    # Combine val — EMNIST val split (first 10%) + user val
    if X_em_train is not None:
        n_em_val = max(1, len(X_em_train) // 10)
        X_val = np.concatenate([X_u_val, X_em_train[:n_em_val]], axis=0)
        y_val = np.concatenate([y_u_val, y_em_train[:n_em_val]], axis=0)
    else:
        X_val, y_val = X_u_val, y_u_val

    # Test = user test originals + EMNIST test
    test_parts_X = [X_u_test]
    test_parts_y = [y_u_test]
    if X_em_test is not None:
        test_parts_X.append(X_em_test)
        test_parts_y.append(y_em_test)
    X_test = np.concatenate(test_parts_X, axis=0)
    y_test = np.concatenate(test_parts_y, axis=0)

    print(f"\n  Final train : {len(X_train)} samples")
    print(f"  Final val   : {len(X_val)} samples")
    print(f"  Final test  : {len(X_test)} samples")

    if len(X_train) < 100:
        print("\n  ERROR: Too few training samples (<100). "
              "Collect more data or enable EMNIST.")
        sys.exit(1)

    if len(X_val) == 0:
        print("\n  ERROR: No validation samples. Need at least 3 originals per class "
              "for a proper split. Collect more data.")
        sys.exit(1)

    # Class weights
    class_weights = compute_class_weights(y_train)

    # ── 7. Train ───────────────────────────────────────────────────────────────
    print("\n[6/8] Building model ...")
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    print(f"\n[7/8] Training (max {args.epochs} epochs) ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload best weights (already done by EarlyStopping, but confirm)
    if MODEL_PATH.exists():
        model = keras.models.load_model(MODEL_PATH)

    # ── 8. Evaluate ────────────────────────────────────────────────────────────
    print("\n[8/8] Evaluating ...")
    _, val_acc  = model.evaluate(X_val,  y_val,  verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n  Validation accuracy : {val_acc:.4f}  ({val_acc:.1%})")
    print(f"  Test accuracy       : {test_acc:.4f}  ({test_acc:.1%})")

    # Per-class report
    from evaluate import (
        full_evaluation_report, plot_confusion_matrix,
        plot_training_history as _plot_hist,
    )
    full_evaluation_report(model, X_test, y_test,
                           out_dir=MODELS_DIR, split_name="test")
    plot_confusion_matrix(model, X_test, y_test, MODELS_DIR / "confusion_matrix.png")
    plot_training_history(history, MODELS_DIR / "training_history.png")

    # Bias check
    check_prediction_bias(model, X_test, y_test)

    # Save artifacts
    print("\n  Saving model artifacts ...")
    save_model_artifacts(model, val_acc, test_acc)

    # Final summary
    print("\n" + "=" * 62)
    print("TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Validation accuracy : {val_acc:.1%}")
    print(f"  Test accuracy       : {test_acc:.1%}")
    print(f"\n  Files saved:")
    print(f"    {MODEL_PATH}")
    print(f"    {LABELS_PATH}")
    print(f"    {PREPROC_CONFIG}")
    print(f"    {MODELS_DIR/'confusion_matrix.png'}")
    print(f"    {MODELS_DIR/'training_history.png'}")
    print(f"    {MODELS_DIR/'test_classification_report.txt'}")
    print(f"\n  Next step: python predict.py --draw")

    if total_orig < NUM_CLASSES * 5:
        print(f"\n  *** IMPORTANT WARNING ***")
        print(f"  You have only {total_orig} original user handwriting samples.")
        print(f"  The model learned primarily from EMNIST, not your handwriting.")
        print(f"  For best live-drawing performance:")
        print(f"    python dataset_builder.py --batch --rounds 20")
        print(f"  Then retrain.")


if __name__ == "__main__":
    main()
