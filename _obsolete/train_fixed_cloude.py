"""
Train Your Handwriting Recognition Model (FIXED VERSION)
==========================================================
This is train_simple.py with the overfitting-on-'A' bug fixed.

WHAT WAS WRONG:
----------------
1. Class imbalance: your dataset.csv had 3 drawings of 'A' but only
   1 drawing for many other letters. The old augment_simple.py
   multiplied every class by a fixed 9x, so that imbalance carried
   straight through (27 'A' images vs 9 for most other letters).
   The model saw 'A' three times more often, so it learned to guess
   'A' whenever it was unsure.
2. Some image paths in dataset.csv pointed to lowercase folders
   (e.g. dataset_images/a/...) that don't actually exist as separate
   folders on case-sensitive filesystems (Linux) - only the uppercase
   folder does. Depending on the OS you train on, this could silently
   drop entire classes.
3. Very little data overall (~1-3 drawings per class) with a fairly
   deep CNN -> the model has way more capacity than the data can
   support, which is a recipe for overfitting/memorization in general.

WHAT THIS SCRIPT DOES DIFFERENTLY:
------------------------------------
- Trains on balanced_dataset.csv (from balance_augment.py), where every
  class has the same number of images.
- Still computes class_weight='balanced' as a safety net in case the
  input CSV isn't perfectly balanced.
- Adds on-the-fly random augmentation INSIDE the model (rotation/zoom/
  translation), so the network sees a fresh variation every epoch
  instead of memorizing a fixed set of pre-generated images.
- Adds L2 weight regularization and slightly higher dropout, since the
  underlying dataset is still small.

HOW TO RUN:
-----------
   python balance_augment.py     # if you haven't already
   python train_simple_fixed.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────

INPUT_CSV     = "balanced_dataset.csv"    # from balance_augment.py
FALLBACK_CSV  = "augmented_dataset.csv"   # used if balanced_dataset.csv is missing
OUTPUT_DIR    = "my_model"
IMAGE_SIZE    = 28
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
L2            = 1e-4

# ──────────────────────────────────────────────────────────────


def resolve_path(raw_path: str):
    """Resolve a CSV image path, tolerating case differences between folder
    names (Windows/macOS are case-insensitive, Linux is not)."""
    p = Path(raw_path.replace("\\", "/"))
    if p.exists():
        return p
    parent, grandparent = p.parent, p.parent.parent
    if grandparent.exists():
        for candidate_dir in grandparent.iterdir():
            if candidate_dir.is_dir() and candidate_dir.name.lower() == parent.name.lower():
                candidate = candidate_dir / p.name
                if candidate.exists():
                    return candidate
                for f in candidate_dir.iterdir():
                    if f.name.lower() == p.name.lower():
                        return f
    return None


# ── 1. LOAD IMAGES INTO MEMORY ────────────────────────────────

def load_images(paths, labels):
    X, y = [], []
    for path, label in zip(paths, labels):
        try:
            img = Image.open(path).convert("L")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(label)
        except Exception:
            pass
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


# ── 2. BUILD THE MODEL ────────────────────────────────────────

def build_model(num_classes):
    """
    Same conv/dense structure as before, but with:
      - in-model random augmentation (new random variation every epoch)
      - L2 regularization on the dense layers
      - higher dropout, since the training set is still small
    """
    augmentation = keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = augmentation(inputs)

    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="HandwritingCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 3. PLOT RESULTS ───────────────────────────────────────────

def plot_results(history, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("How well the model is learning")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Chart saved -> {out_path}")


def plot_confusion(model, X_val, y_val, classes, out_path):
    """Extra diagnostic: shows exactly which letters get confused with
    which - use this to check whether 'A' is still stealing predictions."""
    from sklearn.metrics import confusion_matrix
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm = confusion_matrix(y_val, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, fontsize=7, rotation=90)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion matrix (look for a bright column = still biased)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Confusion matrix saved -> {out_path}")


# ── 4. MAIN ───────────────────────────────────────────────────

def main():
    print("\nHandwriting Trainer - TensorFlow/Keras (fixed)")
    print("=" * 48)

    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        print(f"'{INPUT_CSV}' not found, falling back to '{FALLBACK_CSV}'")
        print("(Run balance_augment.py first for a properly balanced dataset!)")
        csv_path = Path(FALLBACK_CSV)
        if not csv_path.exists():
            print(f"\nERROR: could not find '{INPUT_CSV}' or '{FALLBACK_CSV}'.")
            return

    paths, raw_labels = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            p = resolve_path(row["image_path"])
            if p is not None:
                paths.append(str(p))
                raw_labels.append(row["label"])

    print(f"\nLoaded {len(paths)} images")
    print(f"Characters: {sorted(set(raw_labels))}")

    if len(paths) < 20:
        print("\nERROR: too few images. Collect more data first!")
        return

    # Report per-class counts so imbalance is visible before training
    from collections import Counter
    counts = Counter(raw_labels)
    print(f"Per-class count range: min={min(counts.values())}, "
          f"max={max(counts.values())}")

    le = LabelEncoder()
    encoded = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    labels_path = Path(OUTPUT_DIR) / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)
    print(f"Label map saved -> {labels_path}")

    X_train_p, X_val_p, y_train, y_val = train_test_split(
        paths, encoded, test_size=0.15, stratify=encoded, random_state=42
    )
    print(f"Training: {len(X_train_p)}  |  Validation: {len(X_val_p)}\n")

    print("Loading images...")
    X_train, y_train = load_images(X_train_p, y_train)
    X_val, y_val = load_images(X_val_p, y_val)
    print(f"Images loaded - shape: {X_train.shape}\n")

    # Safety net: even if the CSV isn't perfectly balanced, weight the loss
    # so no class (e.g. 'A') dominates the gradient.
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {i: w for i, w in enumerate(class_weights_arr)}
    print(f"Class weights (1.0 = baseline): "
          f"min={min(class_weights_arr):.2f}, max={max(class_weights_arr):.2f}\n")

    model = build_model(num_classes)
    model.summary()
    print()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(Path(OUTPUT_DIR) / "model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    print(f"Training for up to {EPOCHS} epochs...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    plot_results(history, Path(OUTPUT_DIR) / "accuracy.png")
    plot_confusion(model, X_val, y_val, list(le.classes_),
                    Path(OUTPUT_DIR) / "confusion_matrix.png")

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nTraining complete!")
    print(f"   Best accuracy   : {val_acc:.1%}")
    print(f"   Model saved to  : {OUTPUT_DIR}/model.keras")
    print(f"   Labels saved to : {OUTPUT_DIR}/labels.json")
    print(f"   Accuracy chart  : {OUTPUT_DIR}/accuracy.png")
    print(f"   Confusion matrix: {OUTPUT_DIR}/confusion_matrix.png")
    print(f"\nNext step: run  python predict_tf.py --draw  to test your model!")


if __name__ == "__main__":
    main()