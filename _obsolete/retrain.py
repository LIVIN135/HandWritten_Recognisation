"""
retrain.py
==========
Controlled model retraining using accumulated user feedback.

PIPELINE:
  1. Load existing model
  2. Evaluate it (baseline metrics)
  3. Load original training data (EMNIST + user originals)
  4. Load and validate feedback data
  5. Check feedback balance → warn if highly skewed
  6. Build combined dataset (original + feedback, properly balanced)
  7. Fine-tune the model (low LR, fewer epochs)
  8. Evaluate new model on held-out test set
  9. Compare new vs old model
  10. Save new model only if it meets quality threshold
  11. Keep old model as backup (versioned)

Usage:
  python retrain.py                    — retrain using all feedback
  python retrain.py --full-retrain     — full retrain from scratch (like train.py)
  python retrain.py --dry-run          — show what would happen, don't train
  python retrain.py --min-samples 10   — require at least N feedback samples
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    CLASSES, NUM_CLASSES, CHAR_TO_IDX,
    IMAGE_SIZE, MODELS_DIR, MODEL_PATH, LABELS_PATH,
    LEARNING_RATE, EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR,
    MIN_IMPROVEMENT_THRESHOLD, RANDOM_SEED,
    LEGACY_DATASET_CSV, EMNIST_DIR, EMNIST_SAMPLES_PER_CLASS,
    ensure_dirs,
)
from preprocessing import preprocess_image
from feedback_manager import FeedbackManager

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL VERSIONING
# ──────────────────────────────────────────────────────────────────────────────

def get_next_version() -> int:
    """Scan models/ for existing versioned files and return next version number."""
    existing = list(MODELS_DIR.glob("handwriting_model_v*.keras"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions.append(v)
        except ValueError:
            pass
    return max(versions) + 1 if versions else 1


def backup_current_model() -> Path:
    """Copy current model to a versioned backup. Returns backup path."""
    if not MODEL_PATH.exists():
        return None
    v = get_next_version()
    backup = MODELS_DIR / f"handwriting_model_v{v}.keras"
    shutil.copy2(MODEL_PATH, backup)
    print(f"  Backed up current model → {backup.name}")
    return backup


def save_version_metadata(
    version: int, val_acc: float, test_acc: float,
    macro_f1: float, bal_acc: float,
    n_training: int, n_feedback: int,
    notes: str = "",
):
    """Append version metadata to model_versions.json."""
    meta_path = MODELS_DIR / "model_versions.json"
    versions = []
    if meta_path.exists():
        with open(meta_path) as f:
            try:
                versions = json.load(f)
            except json.JSONDecodeError:
                versions = []

    versions.append({
        "version"          : version,
        "model_file"       : f"handwriting_model_v{version}.keras",
        "trained_at"       : datetime.now().isoformat(),
        "val_accuracy"     : round(val_acc,  4),
        "test_accuracy"    : round(test_acc, 4),
        "macro_f1"         : round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc,  4),
        "training_samples" : n_training,
        "feedback_samples" : n_feedback,
        "notes"            : notes,
    })

    with open(meta_path, "w") as f:
        json.dump(versions, f, indent=2)
    print(f"  Version metadata saved → {meta_path}")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def quick_eval(model, X: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    """Return accuracy, balanced_accuracy, macro_f1 for a model."""
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score,
    )
    preds = np.argmax(model.predict(X, verbose=0), axis=1)
    acc      = float((preds == y).mean())
    bal_acc  = float(balanced_accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))

    tag = f"[{label}]" if label else ""
    print(f"  {tag} Accuracy={acc:.1%}  "
          f"Balanced={bal_acc:.1%}  MacroF1={macro_f1:.3f}")
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1}


# ──────────────────────────────────────────────────────────────────────────────
# PER-CLASS COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_per_class(
    model_old, model_new,
    X_test: np.ndarray, y_test: np.ndarray,
    corrected_labels: list,
):
    """
    Print before/after recall for the classes that were corrected.
    Also checks whether other classes degraded significantly.
    """
    from sklearn.metrics import recall_score

    preds_old = np.argmax(model_old.predict(X_test, verbose=0), axis=1)
    preds_new = np.argmax(model_new.predict(X_test, verbose=0), axis=1)

    print("\n  Per-class recall comparison (corrected classes):")
    print(f"  {'Class':<8} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'-'*36}")

    degraded = []
    for ch in CLASSES:
        idx  = CHAR_TO_IDX[ch]
        mask = (y_test == idx)
        if not mask.any():
            continue
        old_rec = float((preds_old[mask] == idx).mean())
        new_rec = float((preds_new[mask] == idx).mean())
        delta   = new_rec - old_rec
        flag    = ""

        if ch in corrected_labels:
            flag = " ← corrected"
        elif delta < -0.10:
            flag = " ← DEGRADED"
            degraded.append((ch, old_rec, new_rec))

        if ch in corrected_labels or abs(delta) >= 0.05:
            print(f"  {ch!r:<8} {old_rec:>8.1%} {new_rec:>8.1%} "
                  f"{delta:>+8.1%}  {flag}")

    if degraded:
        print(f"\n  WARNING: {len(degraded)} classes degraded by >10%:")
        for ch, old, new in degraded:
            print(f"    '{ch}': {old:.1%} → {new:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# FINE-TUNE TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    *,
    epochs:     int   = 20,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LEARNING_RATE * 0.1,  # lower LR for fine-tuning
) -> keras.callbacks.History:
    """Fine-tune the model on combined original + feedback data."""
    from collections import Counter
    from sklearn.utils.class_weight import compute_class_weight

    # Class weights — account for imbalance
    class_weights_arr = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}

    # Set a lower learning rate
    keras.backend.set_value(model.optimizer.learning_rate, lr)

    callbacks = [
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
            min_lr=MIN_LR * 0.1,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Controlled model retraining")
    parser.add_argument("--full-retrain",  action="store_true",
                        help="Full retrain from scratch (runs train.py)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Report what would happen without training")
    parser.add_argument("--min-samples",   type=int, default=5,
                        help="Minimum feedback samples required (default: 5)")
    parser.add_argument("--epochs",        type=int, default=20,
                        help="Max fine-tuning epochs (default: 20)")
    parser.add_argument("--no-emnist",     action="store_true",
                        help="Skip EMNIST in the combined dataset")
    args = parser.parse_args()

    ensure_dirs()

    if not TF_AVAILABLE:
        print("ERROR: TensorFlow required.  pip install tensorflow")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("CONTROLLED RETRAINING")
    print("=" * 60)

    # ── Full retrain shortcut ──────────────────────────────────────────────────
    if args.full_retrain:
        print("Full retrain requested — delegating to train.py ...")
        import subprocess
        result = subprocess.run([sys.executable, "train.py"])
        sys.exit(result.returncode)

    # ── 1. Check feedback ──────────────────────────────────────────────────────
    fb_mgr = FeedbackManager()
    fb_total = fb_mgr.total_count()

    print(f"\n[1/9] Feedback dataset: {fb_total} samples")
    if fb_total < args.min_samples:
        print(f"  Only {fb_total} feedback samples (minimum: {args.min_samples}).")
        print(f"  Use predict.py to collect more feedback before retraining.")
        sys.exit(0)

    if fb_mgr.is_imbalanced():
        print("  WARNING: Feedback dataset is highly imbalanced.")
        print("  Retraining may over-fit to frequently-corrected classes.")

    counts = fb_mgr.per_class_counts()
    corrected_classes = list(counts.keys())
    print(f"  Classes covered: {sorted(corrected_classes)}")

    # ── 2. Validate feedback ───────────────────────────────────────────────────
    print("\n[2/9] Validating feedback samples ...")
    val_result = fb_mgr.validate_all(verbose=False)
    print(f"  Valid: {val_result['valid']} / {val_result['total']}")
    if val_result["issues"]:
        print(f"  WARNING: {len(val_result['issues'])} invalid samples (will be skipped)")

    if args.dry_run:
        print("\n  [dry-run] Would fine-tune with feedback data.")
        from feedback_manager import cmd_stats
        cmd_stats(fb_mgr)
        sys.exit(0)

    # ── 3. Load existing model ─────────────────────────────────────────────────
    print("\n[3/9] Loading current model ...")
    if not MODEL_PATH.exists():
        print(f"  ERROR: No model found at {MODEL_PATH}")
        print("  Run  python train.py  first.")
        sys.exit(1)

    model_old = keras.models.load_model(MODEL_PATH)
    print(f"  Loaded: {MODEL_PATH}")

    # ── 4. Load original training data ─────────────────────────────────────────
    print("\n[4/9] Loading original training data ...")
    from train import (
        load_user_originals, split_originals_no_leakage,
        load_and_augment_user_train, load_user_split, load_emnist,
    )

    user_originals = load_user_originals(LEGACY_DATASET_CSV)
    train_bl, val_bl, test_bl = split_originals_no_leakage(user_originals)

    X_u_train, y_u_train = load_and_augment_user_train(train_bl)
    X_u_val,   y_u_val   = load_user_split(val_bl)
    X_u_test,  y_u_test  = load_user_split(test_bl)

    print(f"  User train: {len(X_u_train)}  val: {len(X_u_val)}  test: {len(X_u_test)}")

    # EMNIST
    X_em_train = X_em_val = X_em_test = None
    y_em_train = y_em_val = y_em_test = None

    if not args.no_emnist:
        try:
            X_em_train, y_em_train, X_em_test_f, y_em_test_f, _ = load_emnist(
                EMNIST_DIR, max_per_class=EMNIST_SAMPLES_PER_CLASS
            )
            # Use first 10% of EMNIST train as val supplement
            n_val = max(1, len(X_em_train) // 10)
            X_em_val   = X_em_train[:n_val]
            y_em_val   = y_em_train[:n_val]
            X_em_train = X_em_train[n_val:]
            y_em_train = y_em_train[n_val:]
            X_em_test  = X_em_test_f
            y_em_test  = y_em_test_f
            print(f"  EMNIST train: {len(X_em_train)}  val: {len(X_em_val)}  "
                  f"test: {len(X_em_test)}")
        except Exception as e:
            print(f"  WARNING: EMNIST load failed: {e}  (continuing without)")

    # ── 5. Load feedback data ──────────────────────────────────────────────────
    print("\n[5/9] Loading feedback data ...")
    X_fb, y_fb = fb_mgr.get_balanced_sample(max_per_class=50)
    print(f"  Feedback samples: {len(X_fb)}")

    if len(X_fb) == 0:
        print("  No valid feedback samples. Aborting.")
        sys.exit(0)

    # ── 6. Build combined dataset ──────────────────────────────────────────────
    print("\n[6/9] Combining datasets ...")

    train_parts_X = [X_u_train, X_fb]
    train_parts_y = [y_u_train, y_fb]
    val_parts_X   = [X_u_val]
    val_parts_y   = [y_u_val]
    test_parts_X  = [X_u_test]
    test_parts_y  = [y_u_test]

    if X_em_train is not None:
        train_parts_X.append(X_em_train)
        train_parts_y.append(y_em_train)
        val_parts_X.append(X_em_val)
        val_parts_y.append(y_em_val)
        test_parts_X.append(X_em_test)
        test_parts_y.append(y_em_test)

    X_train = np.concatenate([x for x in train_parts_X if len(x) > 0])
    y_train = np.concatenate([y for y in train_parts_y if len(y) > 0])
    X_val   = np.concatenate([x for x in val_parts_X if len(x) > 0])
    y_val   = np.concatenate([y for y in val_parts_y if len(y) > 0])
    X_test  = np.concatenate([x for x in test_parts_X if len(x) > 0])
    y_test  = np.concatenate([y for y in test_parts_y if len(y) > 0])

    # Shuffle train
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    if len(X_val) == 0:
        print("  ERROR: No validation data. Cannot retrain safely.")
        sys.exit(1)

    # ── 7. Baseline evaluation ─────────────────────────────────────────────────
    print("\n[7/9] Baseline evaluation (current model) ...")
    old_metrics = quick_eval(model_old, X_test, y_test, "OLD")

    # ── 8. Fine-tune ──────────────────────────────────────────────────────────
    print(f"\n[8/9] Fine-tuning (max {args.epochs} epochs, LR={LEARNING_RATE*0.1:.1e}) ...")
    model_new = keras.models.load_model(MODEL_PATH)  # fresh copy to fine-tune

    fine_tune(
        model_new, X_train, y_train, X_val, y_val,
        epochs=args.epochs,
    )

    # ── 9. Evaluate new model ──────────────────────────────────────────────────
    print("\n[9/9] Evaluating new model ...")
    new_metrics = quick_eval(model_new, X_test, y_test, "NEW")

    # Compare per-class for the corrected labels
    compare_per_class(model_old, model_new, X_test, y_test, corrected_classes)

    # ── Decision: save or reject ───────────────────────────────────────────────
    old_f1 = old_metrics["macro_f1"]
    new_f1 = new_metrics["macro_f1"]
    delta  = new_f1 - old_f1

    print(f"\n  Macro F1:  OLD={old_f1:.4f}  NEW={new_f1:.4f}  Δ={delta:+.4f}")

    if delta >= MIN_IMPROVEMENT_THRESHOLD:
        print(f"\n  ✓ New model meets quality threshold (Δ ≥ {MIN_IMPROVEMENT_THRESHOLD}).")
        print(f"  Saving new model ...")

        backup = backup_current_model()
        model_new.save(MODEL_PATH)
        print(f"  Saved: {MODEL_PATH}")

        v = get_next_version() - 1
        save_version_metadata(
            version=v,
            val_acc=new_metrics["accuracy"],
            test_acc=new_metrics["accuracy"],
            macro_f1=new_f1,
            bal_acc=new_metrics["balanced_accuracy"],
            n_training=len(X_train),
            n_feedback=len(X_fb),
            notes=f"Fine-tuned with {len(X_fb)} feedback samples",
        )
    else:
        print(f"\n  ✗ New model is WORSE than current (Δ={delta:+.4f} < {MIN_IMPROVEMENT_THRESHOLD}).")
        print(f"  Keeping the current model. No changes saved.")
        print(f"  TIP: Collect more diverse feedback and try again.")

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Before  Macro F1: {old_f1:.4f}")
    print(f"  After   Macro F1: {new_f1:.4f}")
    if delta >= MIN_IMPROVEMENT_THRESHOLD:
        print(f"  Result : MODEL UPDATED")
    else:
        print(f"  Result : OLD MODEL RETAINED")
    print(f"\n  Next step: python predict.py --draw")


if __name__ == "__main__":
    main()
