"""

evaluate.py
===========
Comprehensive model evaluation with per-class metrics.

Generates:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix PNG
  - Balanced accuracy, macro F1, per-class recall
  - Worst-performing class list
  - Prediction bias detection

Usage:
  python evaluate_kiro.py                          # evaluate saved model on test set
  python evaluate_kiro.py --split val             # evaluate on validation set
  python evaluate_kiro.py --csv my_test_data.csv  # evaluate on custom CSV
"""
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_kiro import (
    CLASSES, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR,
    MODEL_PATH, LABELS_PATH, MODELS_DIR,
    LEGACY_DATASET_CSV, ensure_dirs,
)
from preprocessing_kiro import preprocess_image


# ──────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT
# ──────────────────────────────────────────────────────────────────────────────

def full_evaluation_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    out_dir: Path = MODELS_DIR,
    split_name: str = "test",
) -> dict:
    """
    Compute and print a full per-class evaluation report.

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, macro_f1,
                    macro_precision, macro_recall, per_class, worst_classes
    """
    from sklearn.metrics import (
        classification_report, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
    )

    preds = np.argmax(model.predict(X, verbose=0), axis=1)

    acc      = float((preds == y).mean())
    bal_acc  = float(balanced_accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    macro_pr = float(precision_score(y, preds, average="macro", zero_division=0))
    macro_rc = float(recall_score(y, preds, average="macro", zero_division=0))

    # Per-class metrics
    labels_present = sorted(set(y.tolist()) | set(preds.tolist()))
    per_class = {}
    for idx in range(NUM_CLASSES):
        ch = CLASSES[idx]
        mask = (y == idx)
        if not mask.any():
            per_class[ch] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            continue
        tp = int(((preds == idx) & mask).sum())
        fp = int(((preds == idx) & ~mask).sum())
        fn = int((mask & (preds != idx)).sum())
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[ch] = {
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4),
            "support"  : int(mask.sum()),
        }

    # Worst classes (by F1, at least 5 support)
    worst = sorted(
        [(ch, d["f1"]) for ch, d in per_class.items() if d["support"] >= 5],
        key=lambda x: x[1]
    )[:10]

    # Print report
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"EVALUATION REPORT  [{split_name.upper()} SET]")
    print(sep)
    print(f"  Samples             : {len(y)}")
    print(f"  Accuracy            : {acc:.4f}  ({acc:.1%})")
    print(f"  Balanced accuracy   : {bal_acc:.4f}  ({bal_acc:.1%})")
    print(f"  Macro precision     : {macro_pr:.4f}")
    print(f"  Macro recall        : {macro_rc:.4f}")
    print(f"  Macro F1            : {macro_f1:.4f}")

    print(f"\n  {'Class':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*50}")
    for ch in CLASSES:
        d = per_class[ch]
        flag = ""
        if d["support"] > 0 and d["recall"] < 0.5:
            flag = " ← low"
        print(f"  {repr(ch):<8} {d['precision']:>10.3f} {d['recall']:>8.3f} "
              f"{d['f1']:>8.3f} {d['support']:>9}  {flag}")

    if worst:
        print(f"\n  Worst-performing classes (F1 score, min 5 samples):")
        for ch, f1 in worst:
            d = per_class[ch]
            print(f"    '{ch}'  F1={f1:.3f}  "
                  f"(precision={d['precision']:.3f}, recall={d['recall']:.3f})")

    # Prediction bias check
    pred_counter = Counter(preds.tolist())
    n = len(preds)
    biased_classes = [(CLASSES[i], cnt / n) for i, cnt in pred_counter.items()
                      if cnt / n > 0.20]
    if biased_classes:
        print(f"\n  *** WARNING: PREDICTION BIAS DETECTED ***")
        for ch, frac in biased_classes:
            print(f"  '{ch}' was predicted for {frac:.1%} of samples.")
        print(f"  The model is heavily biased. More diverse training data needed.")
    else:
        top3 = [(CLASSES[i], cnt / n) for i, cnt in pred_counter.most_common(3)]
        print(f"\n  Top 3 predicted classes: "
              + ", ".join(f"'{c}' {f:.1%}" for c, f in top3))
        print(f"  No significant prediction bias.")

    print(sep)

    # Save text report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{split_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report [{split_name}]\n")
        f.write(f"{'='*50}\n")
        f.write(f"Samples           : {len(y)}\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"Balanced accuracy : {bal_acc:.4f}\n")
        f.write(f"Macro precision   : {macro_pr:.4f}\n")
        f.write(f"Macro recall      : {macro_rc:.4f}\n")
        f.write(f"Macro F1          : {macro_f1:.4f}\n\n")
        f.write(f"{'Class':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}\n")
        f.write(f"{'-'*50}\n")
        for ch in CLASSES:
            d = per_class[ch]
            f.write(f"{repr(ch):<8} {d['precision']:>10.3f} {d['recall']:>8.3f} "
                    f"{d['f1']:>8.3f} {d['support']:>9}\n")
        if biased_classes:
            f.write(f"\nWARNING: prediction bias detected for: "
                    f"{[c for c,_ in biased_classes]}\n")
    print(f"  Report saved: {report_path}")

    return {
        "accuracy"           : acc,
        "balanced_accuracy"  : bal_acc,
        "macro_f1"           : macro_f1,
        "macro_precision"    : macro_pr,
        "macro_recall"       : macro_rc,
        "per_class"          : per_class,
        "worst_classes"      : worst,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X: np.ndarray, y: np.ndarray, out_path: Path):
    """Plot and save a 62×62 confusion matrix."""
    preds = np.argmax(model.predict(X, verbose=0), axis=1)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for true, pred in zip(y, preds):
        cm[int(true), int(pred)] += 1

    # Normalise rows (true class fractions)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums.astype(float), 0.0)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, fontsize=6, rotation=90)
    ax.set_yticklabels(CLASSES, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)\n"
                 "Bright column = model over-predicts that class", fontsize=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HISTORY PLOT  (re-exported so train.py can import it)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, out_path: Path):
    """Duplicate of the function in train.py — kept here for import consistency."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Training history saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate handwriting model")
    parser.add_argument("--csv",    type=Path, default=LEGACY_DATASET_CSV)
    parser.add_argument("--split",  type=str,  default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--model",  type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    ensure_dirs()

    try:
        from tensorflow import keras
    except ImportError:
        print("ERROR: TensorFlow required.  pip install tensorflow")
        sys.exit(1)

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("Run  python train_kiro.py  first.")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model)

    # Load data
    import csv as _csv
    from train_kiro import load_user_originals, split_originals_no_leakage, load_user_split

    user_originals = load_user_originals(args.csv)
    train_bl, val_bl, test_bl = split_originals_no_leakage(user_originals)

    split_map = {"train": train_bl, "val": val_bl, "test": test_bl}
    chosen = split_map[args.split]

    X, y = load_user_split(chosen)

    if len(X) == 0:
        print(f"No samples in the '{args.split}' split. Collect more data.")
        sys.exit(1)

    print(f"Evaluating on {len(X)} '{args.split}' samples ...")
    results = full_evaluation_report(model, X, y, split_name=args.split)
    plot_confusion_matrix(model, X, y, MODELS_DIR / f"{args.split}_confusion_matrix.png")


if __name__ == "__main__":
    main()
