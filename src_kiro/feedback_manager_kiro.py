"""

feedback_manager.py
===================
Manages the human-in-the-loop feedback dataset.

Responsibilities:
  - Save corrected samples to data/feedback/
  - Track per-class balance in the feedback buffer
  - Validate every incoming sample before storing
  - Prevent exact duplicate images from accumulating
  - Expose statistics for CLI reporting
  - Support the retrain pipeline

Usage:
  python feedback_manager_kiro.py stats
  python feedback_manager_kiro.py validate
  python feedback_manager_kiro.py clear
  python feedback_manager_kiro.py report
"""
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import csv
import hashlib
import json
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from config_kiro import (
    CLASSES, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR,
    IMAGE_SIZE, FEEDBACK_DIR, FEEDBACK_IMAGES_DIR, FEEDBACK_CSV,
    MAX_FEEDBACK_PER_CLASS, RETRAIN_AFTER_N_FEEDBACK_SAMPLES,
    ensure_dirs,
)
from preprocessing_kiro import preprocess_image


FEEDBACK_FIELDNAMES = [
    "image_path", "true_label", "predicted_label",
    "confidence", "timestamp", "session_id",
]


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK MANAGER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackManager:
    def __init__(self, feedback_csv: Path = FEEDBACK_CSV):
        self.csv_path   = feedback_csv
        self.images_dir = feedback_csv.parent / "images"
        ensure_dirs()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()
        self._hash_cache: set = self._load_existing_hashes()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FEEDBACK_FIELDNAMES).writeheader()

    def _load_existing_hashes(self) -> set:
        """Load MD5 hashes of all existing feedback images for dedup."""
        hashes = set()
        if not self.csv_path.exists():
            return hashes
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                p = Path(row.get("image_path", ""))
                if p.exists():
                    try:
                        hashes.add(_file_hash(p))
                    except Exception:
                        pass
        return hashes

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_sample(
        self,
        image_array: np.ndarray,
        true_label: str,
        predicted_label: str,
    ) -> tuple:
        """
        Validate a feedback sample before saving.

        Returns
        -------
        (ok: bool, reason: str)
        """
        # Label check
        if true_label not in CLASSES:
            return False, f"Invalid true_label: '{true_label}'"
        if predicted_label not in CLASSES and predicted_label != "":
            return False, f"Invalid predicted_label: '{predicted_label}'"

        # Image shape
        if image_array is None:
            return False, "image_array is None"
        arr = np.squeeze(image_array)
        if arr.ndim != 2:
            return False, f"Expected 2D image, got shape {image_array.shape}"
        if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
            return False, f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {arr.shape}"

        # Not empty
        if arr.max() < 0.05:
            return False, "Image appears to be blank (all background)"
        if arr.std() < 0.01:
            return False, "Image has no variation (blank or uniform)"

        return True, "ok"

    # ── Saving ─────────────────────────────────────────────────────────────────

    def save_feedback(
        self,
        image_array: np.ndarray,
        true_label: str,
        predicted_label: str,
        confidence: float,
        session_id: str = "",
    ) -> dict:
        """
        Validate and save one feedback sample.

        NEVER uses the model's prediction as ground-truth.
        Only user-confirmed labels are stored.

        Parameters
        ----------
        image_array      : preprocessed float32 array (28,28,1) or (28,28)
        true_label       : user-confirmed correct label (ground truth)
        predicted_label  : what the model predicted (for analysis only)
        confidence       : model confidence score
        session_id       : optional session identifier

        Returns
        -------
        dict with keys: saved, path, reason, total_feedback
        """
        ok, reason = self.validate_sample(image_array, true_label, predicted_label)
        if not ok:
            return {"saved": False, "reason": reason, "total_feedback": self.total_count()}

        # Check per-class cap
        counts = self.per_class_counts()
        if counts.get(true_label, 0) >= MAX_FEEDBACK_PER_CLASS:
            return {
                "saved": False,
                "reason": f"Class '{true_label}' already has {MAX_FEEDBACK_PER_CLASS} "
                          f"feedback samples (cap reached)",
                "total_feedback": self.total_count()
            }

        # Save image
        arr_8bit = (np.squeeze(image_array) * 255).clip(0, 255).astype(np.uint8)
        img_hash = hashlib.md5(arr_8bit.tobytes()).hexdigest()

        if img_hash in self._hash_cache:
            return {
                "saved": False,
                "reason": "Duplicate image already in feedback dataset",
                "total_feedback": self.total_count()
            }

        self.images_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{true_label}_{uuid.uuid4().hex[:8]}.png"
        dest = self.images_dir / filename

        from PIL import Image
        img_pil = Image.fromarray(arr_8bit, mode="L")
        img_pil.save(dest)
        self._hash_cache.add(img_hash)

        # Append to CSV
        row = {
            "image_path"      : str(dest),
            "true_label"      : true_label,
            "predicted_label" : predicted_label,
            "confidence"      : f"{confidence:.4f}",
            "timestamp"       : datetime.now().isoformat(),
            "session_id"      : session_id,
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FEEDBACK_FIELDNAMES).writerow(row)

        total = self.total_count()
        print(f"  [Feedback] Saved '{true_label}' (pred='{predicted_label}' "
              f"conf={confidence:.0%})  total={total}")

        return {"saved": True, "path": str(dest), "reason": "ok",
                "total_feedback": total}

    # ── Queries ────────────────────────────────────────────────────────────────

    def load_all_rows(self) -> list:
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def total_count(self) -> int:
        return len(self.load_all_rows())

    def per_class_counts(self) -> Counter:
        counts: Counter = Counter()
        for row in self.load_all_rows():
            counts[row["true_label"]] += 1
        return counts

    def correction_counts(self) -> Counter:
        """Count (predicted→true) pairs for confusion analysis."""
        counts: Counter = Counter()
        for row in self.load_all_rows():
            pred  = row.get("predicted_label", "")
            true  = row["true_label"]
            if pred and pred != true:
                counts[(pred, true)] += 1
        return counts

    def correct_prediction_count(self) -> int:
        return sum(1 for r in self.load_all_rows()
                   if r.get("predicted_label") == r["true_label"])

    def high_conf_wrong_count(self, threshold: float = 0.85) -> int:
        """Count samples where model was confidently wrong."""
        n = 0
        for r in self.load_all_rows():
            pred = r.get("predicted_label", "")
            true = r["true_label"]
            try:
                conf = float(r.get("confidence", 0))
            except ValueError:
                conf = 0.0
            if pred != true and conf >= threshold:
                n += 1
        return n

    def is_imbalanced(self, ratio_threshold: float = 5.0) -> bool:
        counts = self.per_class_counts()
        if len(counts) < 2:
            return False
        vals = list(counts.values())
        return max(vals) / max(min(vals), 1) > ratio_threshold

    # ── Validation of saved files ──────────────────────────────────────────────

    def validate_all(self, verbose: bool = True) -> dict:
        rows   = self.load_all_rows()
        issues = []
        valid  = 0

        for row in rows:
            p = Path(row.get("image_path", ""))
            label = row.get("true_label", "")

            if not p.exists():
                issues.append(f"Missing: {p}")
                continue
            if label not in CLASSES:
                issues.append(f"Invalid label '{label}': {p.name}")
                continue
            try:
                arr = preprocess_image(p)
                assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 1)
                valid += 1
            except Exception as exc:
                issues.append(f"Preproc error {p.name}: {exc}")

        if verbose:
            print(f"\n  Feedback validation: {valid}/{len(rows)} valid")
            for iss in issues[:10]:
                print(f"    WARNING: {iss}")
            if len(issues) > 10:
                print(f"    ... and {len(issues)-10} more.")

        return {"total": len(rows), "valid": valid, "issues": issues}

    def get_balanced_sample(
        self, max_per_class: int = 50
    ) -> tuple:
        """
        Return balanced arrays (X, y) from feedback data for retraining.
        Up to max_per_class samples per class.
        """
        rows_by_label: dict = {}
        for row in self.load_all_rows():
            lbl = row["true_label"]
            if lbl not in CLASSES:
                continue
            rows_by_label.setdefault(lbl, []).append(row)

        X_list, y_list = [], []
        for lbl, rows in rows_by_label.items():
            # Limit per class
            selected = rows[-max_per_class:]  # most recent first
            idx = CHAR_TO_IDX[lbl]
            for row in selected:
                p = Path(row["image_path"])
                if not p.exists():
                    continue
                try:
                    arr = preprocess_image(p)
                    X_list.append(arr)
                    y_list.append(idx)
                except Exception:
                    pass

        if not X_list:
            return (np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))

        return (np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.int32))


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def cmd_stats(mgr: FeedbackManager):
    rows   = mgr.load_all_rows()
    counts = mgr.per_class_counts()
    corr   = mgr.correction_counts()
    n_ok   = mgr.correct_prediction_count()
    n_hcw  = mgr.high_conf_wrong_count()
    total  = len(rows)

    print("\n" + "=" * 55)
    print("FEEDBACK DATASET STATISTICS")
    print("=" * 55)
    print(f"  Total feedback samples     : {total}")
    print(f"  Correct predictions (YES)  : {n_ok}")
    print(f"  Incorrect predictions (NO) : {total - n_ok}")
    print(f"  High-confidence errors     : {n_hcw}  (conf ≥ 85%)")
    print(f"  Until next retrain prompt  : "
          f"{RETRAIN_AFTER_N_FEEDBACK_SAMPLES - (total % RETRAIN_AFTER_N_FEEDBACK_SAMPLES)}")

    if mgr.is_imbalanced():
        print(f"\n  WARNING: Feedback dataset is highly imbalanced.")

    # Per-class table
    print(f"\n  {'Class':<8} {'Samples':>8}")
    print(f"  {'-'*20}")
    for ch in CLASSES:
        n = counts.get(ch, 0)
        if n > 0:
            print(f"  {ch!r:<8} {n:>8}")

    missing = [ch for ch in CLASSES if counts.get(ch, 0) == 0]
    if missing:
        print(f"\n  Classes with no feedback: {len(missing)}")

    # Most common corrections
    if corr:
        print(f"\n  Most common user corrections:")
        print(f"  {'Predicted':>10}  →  {'True':>5}  {'Count':>8}")
        print(f"  {'-'*35}")
        for (pred, true), cnt in corr.most_common(15):
            print(f"  {pred!r:>10}  →  {true!r:>5}  {cnt:>8}")

    print("=" * 55)


def cmd_validate(mgr: FeedbackManager):
    print("\nValidating feedback dataset ...")
    result = mgr.validate_all(verbose=True)
    print(f"\n  Valid: {result['valid']} / {result['total']}")
    if result["issues"]:
        print(f"  Issues found: {len(result['issues'])}")
        sys.exit(1)
    else:
        print("  All feedback samples are valid.")


def cmd_clear(mgr: FeedbackManager):
    total = mgr.total_count()
    if total == 0:
        print("Feedback dataset is already empty.")
        return
    confirm = input(
        f"This will delete {total} feedback samples. Type YES to confirm: "
    ).strip()
    if confirm != "YES":
        print("Aborted.")
        return

    import shutil
    if mgr.images_dir.exists():
        shutil.rmtree(mgr.images_dir)
    mgr.images_dir.mkdir(parents=True, exist_ok=True)
    mgr.csv_path.unlink(missing_ok=True)
    mgr._ensure_csv_header()
    mgr._hash_cache.clear()
    print(f"Cleared {total} feedback samples.")


def cmd_report(mgr: FeedbackManager):
    corr = mgr.correction_counts()
    if not corr:
        print("No correction data yet.")
        return

    print("\nMost Common Model Errors (user corrections):")
    print(f"  {'Predicted':>10} → {'True':>5}  {'Count':>8}  Note")
    print(f"  {'-'*50}")

    from config_kiro import CONFUSABLE_PAIRS
    confusable = set(CONFUSABLE_PAIRS) | {(b, a) for a, b in CONFUSABLE_PAIRS}

    for (pred, true), cnt in corr.most_common(20):
        note = " ← visually similar" if (pred, true) in confusable else ""
        print(f"  {pred!r:>10} → {true!r:>5}  {cnt:>8}  {note}")


def main():
    parser = argparse.ArgumentParser(description="Manage the feedback dataset")
    parser.add_argument("command", choices=["stats", "validate", "clear", "report"],
                        help="Command to run")
    args = parser.parse_args()

    ensure_dirs()
    mgr = FeedbackManager()

    if args.command == "stats":
        cmd_stats(mgr)
    elif args.command == "validate":
        cmd_validate(mgr)
    elif args.command == "clear":
        cmd_clear(mgr)
    elif args.command == "report":
        cmd_report(mgr)


if __name__ == "__main__":
    main()
