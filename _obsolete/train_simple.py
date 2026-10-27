"""
🧠  STEP 2: Train Your Handwriting Recognition Model (TensorFlow)
==================================================================
This script teaches a neural network to recognize
handwritten letters and numbers from your dataset.

Powered by TensorFlow + Keras — the easiest way to build neural networks!

HOW TO RUN:
-----------
1. Install requirements:
   pip install tensorflow pillow numpy scikit-learn matplotlib

2. Run this script:
   python train_tf.py

When it's done you'll have a saved model ready to recognize
new handwritten characters!
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────
# ⚙️  SETTINGS — change these if you want
# ──────────────────────────────────────────────────────────────

INPUT_CSV     = "augmented_dataset.csv"   # from augment_simple.py
OUTPUT_DIR    = "my_model"                # where model is saved
IMAGE_SIZE    = 28                        # resize all images to 28×28
EPOCHS        = 20                        # training rounds
BATCH_SIZE    = 32                        # images per training step
LEARNING_RATE = 0.001

# ──────────────────────────────────────────────────────────────


# ── 1. LOAD IMAGES INTO MEMORY ────────────────────────────────

def load_images(paths, labels):
    """Read images from disk and return as numpy arrays."""
    X, y = [], []
    for path, label in zip(paths, labels):
        try:
            img = Image.open(path).convert("L")          # grayscale
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))   # resize to 28×28
            arr = np.array(img, dtype=np.float32) / 255.0  # scale 0→1
            X.append(arr)
            y.append(label)
        except Exception:
            pass   # skip broken images silently
    X = np.array(X)[..., np.newaxis]   # shape: (N, 28, 28, 1)
    y = np.array(y)
    return X, y


# ── 2. BUILD THE MODEL ────────────────────────────────────────

def build_model(num_classes):
    """
    A simple CNN built with Keras.

    Input (28×28 grayscale image)
      → Conv2D  finds edges and strokes
      → Conv2D  finds curves and shapes
      → Conv2D  finds full letter patterns
      → Dense   makes the final prediction
    """
    model = keras.Sequential([
        # Tell Keras the input shape
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        # --- Convolutional layers ---
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(),           # shrink: 28→14

        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(),           # shrink: 14→7

        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(), # flatten smartly

        # --- Decision layers ---
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),             # prevent memorising (overfitting)
        layers.Dense(num_classes, activation="softmax"),  # final prediction
    ], name="HandwritingCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 3. PLOT RESULTS ───────────────────────────────────────────

def plot_results(history, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"],     label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("How well the model is learning")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊  Chart saved → {out_path}")


# ── 4. MAIN ───────────────────────────────────────────────────

def main():
    print("\n🧠  Handwriting Trainer — TensorFlow/Keras")
    print("=" * 45)

    # Check input file
    if not Path(INPUT_CSV).exists():
        print(f"\n❌  Could not find '{INPUT_CSV}'")
        print("    Run augment_simple.py first!")
        return

    # Read CSV
    paths, raw_labels = [], []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            if p.exists():
                paths.append(str(p))
                raw_labels.append(row["label"])

    print(f"\n📦  Loaded {len(paths)} images")
    print(f"🔤  Characters: {sorted(set(raw_labels))}")

    if len(paths) < 20:
        print("\n❌  Too few images. Collect more data first!")
        return

    # Encode labels  A→0, B→1, a→2 …
    le = LabelEncoder()
    encoded = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)
    print(f"🏷️   Number of classes : {num_classes}")

    # Save label mapping so we can decode predictions later
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    labels_path = Path(OUTPUT_DIR) / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)
    print(f"💾  Label map saved → {labels_path}")

    # Split: 85% training, 15% validation
    X_train_p, X_val_p, y_train, y_val = train_test_split(
        paths, encoded, test_size=0.15, stratify=encoded, random_state=42
    )
    print(f"📊  Training: {len(X_train_p)}  |  Validation: {len(X_val_p)}\n")

    # Load images into memory
    print("⏳  Loading images...")
    X_train, y_train = load_images(X_train_p, y_train)
    X_val,   y_val   = load_images(X_val_p,   y_val)
    print(f"✅  Images loaded — shape: {X_train.shape}\n")

    # Build & summarise model
    model = build_model(num_classes)
    model.summary()
    print()

    # Callbacks — things that happen automatically during training
    callbacks = [
        # Save the best model automatically
        keras.callbacks.ModelCheckpoint(
            filepath=str(Path(OUTPUT_DIR) / "model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if model stops improving (saves time)
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,            # stop after 5 rounds of no improvement
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    # 🚀 Train!
    print(f"🚀  Training for up to {EPOCHS} epochs...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,      # show progress bar per epoch
    )

    # Plot accuracy chart
    plot_results(history, Path(OUTPUT_DIR) / "accuracy.png")

    # Final score
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n🎉  Training complete!")
    print(f"   Best accuracy  : {val_acc:.1%}")
    print(f"   Model saved to : {OUTPUT_DIR}/model.keras")
    print(f"   Labels saved to: {OUTPUT_DIR}/labels.json")
    print(f"   Accuracy chart : {OUTPUT_DIR}/accuracy.png")
    print(f"\n👉  Next step: run  python predict_tf.py  to test your model!")


if __name__ == "__main__":
    main()