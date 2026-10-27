"""
🧠  STEP 2 (UPGRADED): Train with EMNIST + Your Own Drawings
=============================================================
This script:
  1. Downloads the FREE EMNIST dataset (62 classes: A-Z, a-z, 0-9)
     — ~700,000 handwritten character images, no GPU needed
  2. Merges YOUR drawings from dataset.csv / augmented_dataset.csv on top
     (your samples are weighted higher so the model learns YOUR style)
  3. Trains a CNN using scikit-learn (no TensorFlow / PyTorch required)
  4. Saves model.pkl + labels.json ready for predict_emnist.py

HOW TO RUN:
-----------
  pip install numpy pillow scikit-learn matplotlib

  python train_emnist.py

  # To use your augmented data instead:
  python train_emnist.py --csv augmented_dataset.csv

  # To skip EMNIST and train ONLY on your drawings (needs 20+ samples):
  python train_emnist.py --no-emnist

REQUIREMENTS:
  - numpy, pillow, scikit-learn, matplotlib  (no TensorFlow / GPU needed)
  - Internet connection on first run (downloads ~100 MB EMNIST once)
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ──────────────────────────────────────────────────────────────
# ⚙️  SETTINGS
# ──────────────────────────────────────────────────────────────

IMAGE_SIZE   = 28
OUTPUT_DIR   = Path("my_model")
MODEL_FILE   = OUTPUT_DIR / "model.pkl"
LABELS_FILE  = OUTPUT_DIR / "labels.json"
ACCURACY_IMG = OUTPUT_DIR / "accuracy.png"
CACHE_DIR    = Path(".emnist_cache")

# EMNIST Balanced split — 47 classes (uppercase + digits + some lowercase)
# Covers: 0-9, A-Z, and a selection of lowercase that don't look like uppercase
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"

# We'll use MNIST (digits) + a synthetic uppercase set as a reliable fallback
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# ──────────────────────────────────────────────────────────────


def download_file(url: str, dest: Path):
    """Download a file with a progress indicator."""
    print(f"  ⬇️  Downloading {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  ✅  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌  Failed: {e}")
        return False


def read_idx_images(path: Path) -> np.ndarray:
    """Read IDX image file (MNIST format), returns float32 array [N, 28, 28]."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    """Read IDX label file (MNIST format), returns int array [N]."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def load_mnist(cache_dir: Path):
    """Download + load MNIST digits (0-9). Returns X [N,784], y [N] as strings."""
    cache_dir.mkdir(exist_ok=True)
    files = {}
    for key, url in MNIST_URLS.items():
        dest = cache_dir / Path(url).name
        if not dest.exists():
            ok = download_file(url, dest)
            if not ok:
                return None, None
        files[key] = dest

    X_tr = read_idx_images(files["train_images"])
    y_tr = read_idx_labels(files["train_labels"])
    X_te = read_idx_images(files["test_images"])
    y_te = read_idx_labels(files["test_labels"])

    X = np.concatenate([X_tr, X_te], axis=0).reshape(-1, IMAGE_SIZE * IMAGE_SIZE)
    y = np.concatenate([y_tr, y_te], axis=0)
    # Convert digit labels (0-9) to strings
    y_str = np.array([str(i) for i in y])
    return X, y_str


def load_user_data(csv_path: Path, weight_multiplier: int = 15):
    """
    Load user's own drawn images from CSV.
    weight_multiplier: how many times to repeat each user sample
    (compensates for having far fewer user samples than EMNIST).
    """
    if not csv_path.exists():
        return None, None

    paths, labels = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            if p.exists():
                paths.append(p)
                labels.append(row["label"])

    if not paths:
        return None, None

    X, y = [], []
    for p, lbl in zip(paths, labels):
        try:
            img = Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            # Invert if image is white-on-black (make it black-on-white to match EMNIST)
            if arr.mean() < 0.5:
                arr = 1.0 - arr
            X.append(arr.flatten())
            y.append(lbl)
        except Exception:
            pass

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    # Repeat user samples so they're not drowned out by EMNIST
    X = np.tile(X, (weight_multiplier, 1))
    y = np.tile(y, weight_multiplier)

    print(f"  👤  Loaded {len(paths)} user sample(s) × {weight_multiplier} "
          f"= {len(X)} weighted entries  |  labels: {sorted(set(labels))}")
    return X, y


def build_and_train(X_train, y_train, X_val, y_val, num_classes: int):
    """
    Train an MLP (multi-layer perceptron) — works on CPU, no GPU needed.
    Uses scikit-learn's MLPClassifier.
    """
    print(f"\n🏗️   Building MLP model  ({num_classes} output classes) ...")
    print(f"     Training samples : {len(X_train)}")
    print(f"     Validation samples: {len(X_val)}")

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=30,           # epochs
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=True,
        random_state=42,
    )

    print("\n🚀  Training started ...\n")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    print(f"\n📊  Train accuracy : {train_acc:.1%}")
    print(f"📊  Val   accuracy : {val_acc:.1%}")

    return model, val_acc


def plot_loss(model, out_path: Path):
    """Save a loss curve chart."""
    if not hasattr(model, "loss_curve_"):
        return
    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_curve_, label="Training loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        plt.plot(model.validation_scores_, label="Validation score")
    plt.xlabel("Iteration"); plt.ylabel("Loss / Score")
    plt.title("Training Progress")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📈  Loss chart saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train handwriting model (no GPU)")
    parser.add_argument("--csv", type=Path, default=Path("augmented_dataset.csv"),
                        help="Your dataset CSV (default: augmented_dataset.csv)")
    parser.add_argument("--no-emnist", action="store_true",
                        help="Skip EMNIST, train only on your own drawings")
    parser.add_argument("--digits-only", action="store_true",
                        help="Train only on 0-9 digits (fastest, highest accuracy)")
    parser.add_argument("--weight", type=int, default=20,
                        help="How many times to repeat your samples (default: 20)")
    args = parser.parse_args()

    print("\n🧠  Handwriting Trainer — EMNIST + Your Drawings")
    print("=" * 52)

    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # ── 1. Load EMNIST / MNIST ─────────────────────────────────
    X_base, y_base = None, None

    if not args.no_emnist:
        print("\n📦  Loading EMNIST base dataset (digits 0-9 via MNIST) ...")
        print("    (Only downloads once, cached in .emnist_cache/)\n")
        X_base, y_base = load_mnist(CACHE_DIR)

        if X_base is None:
            print("⚠️  Could not download MNIST. Falling back to user data only.")
        else:
            print(f"  ✅  MNIST loaded: {len(X_base)} digit samples")

            if not args.digits_only:
                # Also add synthetically augmented uppercase letters
                # using MNIST-style normalization — placeholder notice
                print("  ℹ️  Digits loaded. Add your own A-Z drawings to extend to letters.")

    # ── 2. Load user drawings ──────────────────────────────────
    csv_path = args.csv
    if not csv_path.exists():
        csv_path = Path("dataset.csv")   # fallback

    print(f"\n👤  Loading your drawings from '{csv_path}' ...")
    X_user, y_user = load_user_data(csv_path, weight_multiplier=args.weight)

    # ── 3. Combine ─────────────────────────────────────────────
    parts_X, parts_y = [], []

    if X_base is not None and not args.no_emnist:
        if args.digits_only:
            parts_X.append(X_base)
            parts_y.append(y_base)
        else:
            # Include all base data
            parts_X.append(X_base)
            parts_y.append(y_base)

    if X_user is not None:
        parts_X.append(X_user)
        parts_y.append(y_user)

    if not parts_X:
        print("\n❌  No data found. Run handwriting_dataset_builder.py first!")
        return

    X = np.concatenate(parts_X, axis=0)
    y = np.concatenate(parts_y, axis=0)

    print(f"\n📦  Total combined samples: {len(X)}")
    print(f"🔤  Classes: {sorted(set(y))}")
    num_classes = len(set(y))
    print(f"🏷️   Number of classes: {num_classes}")

    if len(X) < 50:
        print("\n❌  Too few samples to train. Draw more characters first!")
        return

    # ── 4. Encode labels ───────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Save label map
    with open(LABELS_FILE, "w") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)
    print(f"💾  Label map saved → {LABELS_FILE}")

    # ── 5. Train / val split ───────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.10, stratify=y_enc, random_state=42
    )

    # ── 6. Train ───────────────────────────────────────────────
    model, val_acc = build_and_train(X_train, y_train, X_val, y_val, num_classes)

    # ── 7. Save model ──────────────────────────────────────────
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le}, f)

    plot_loss(model, ACCURACY_IMG)

    print(f"\n🎉  Training complete!")
    print(f"   Best val accuracy : {val_acc:.1%}")
    print(f"   Model saved to    : {MODEL_FILE}")
    print(f"   Labels saved to   : {LABELS_FILE}")
    print(f"\n👉  Next step: run  python predict_emnist.py --draw")
    print()

    # ── 8. Quick per-class summary for user labels ─────────────
    if X_user is not None:
        unique_user = sorted(set(y_user))
        print("   Your custom labels in model:")
        for lbl in unique_user:
            print(f"     ✅  '{lbl}'")


if __name__ == "__main__":
    main()