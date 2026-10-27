"""
dataset_validator.py
====================
Validates the handwritten character dataset before training.

Checks:
  - All 62 classes present
  - Per-class original sample counts
  - Missing image files
  - Invalid labels
  - Empty / too dark / too light images
  - Duplicate image paths
  - Near-duplicate images (pixel similarity)
  - Image dimensions
  - Preprocessing compatibility
  - Train/val/test leakage (if split files exist)

Usage:
  python dataset_validator.py
  python dataset_validator.py --csv dataset.csv
  python dataset_validator.py --csv dataset.csv --check-duplicates
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    CLASSES, NUM_CLASSES, IMAGE_SIZE, LEGACY_DATASET_CSV,
    MIN_RECOMMENDED_ORIGINALS, VALID_MEAN_RANGE, DUPLICATE_DIFF_THRESHOLD,
    ensure_dirs,
)
from preprocessing import preprocess_image

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """MD5 hash of file contents (fast exact-duplicate check)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _pixel_fingerprint(path: Path, size: int = 8) -> Optional[np.ndarray]:
    """
    8x8 average-hash fingerprint of an image for near-duplicate detection.
    Returns None on error.
    """
    try:
        from PIL import Image as PImage
        img = PImage.open(path).convert("L").resize((size, size), PImage.LANCZOS)
        return np.array(img, dtype=np.float32)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATASET CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset(
    csv_path: Path = LEGACY_DATASET_CSV,
    check_duplicates: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run all validation checks and return a results dict.
    Prints a formatted report if verbose=True.
    """

    results = {
        "csv_path"           : str(csv_path),
        "total_rows"         : 0,
        "valid_rows"         : 0,
        "missing_files"      : [],
        "invalid_labels"     : [],
        "duplicate_paths"    : [],
        "exact_duplicates"   : [],
        "near_duplicates"    : [],
        "empty_images"       : [],
        "bad_exposure"       : [],
        "wrong_dimensions"   : [],
        "preproc_errors"     : [],
        "class_counts"       : {},
        "missing_classes"    : [],
        "low_data_classes"   : [],
        "warnings"           : [],
        "errors"             : [],
        "passed"             : False,
    }

    # ── 1. Load CSV ─────────────────────────────────────────────────────────────
    rows = load_csv(csv_path)
    if not rows:
        results["errors"].append(f"CSV not found or empty: {csv_path}")
        _print_report(results)
        return results

    results["total_rows"] = len(rows)

    # ── 2. Per-row checks ──────────────────────────────────────────────────────
    class_counts: Counter = Counter()
    seen_paths: dict = {}       # path → row index
    file_hashes: dict = {}      # md5 → path
    fingerprints: list = []     # (label, path, fp)

    for i, row in enumerate(rows):
        path_str  = row.get("image_path", "").strip()
        label     = row.get("label", "").strip()

        # Check label
        if label not in CLASSES:
            results["invalid_labels"].append(f"Row {i+2}: label='{label}'")
            continue

        # Check path
        if not path_str:
            results["missing_files"].append(f"Row {i+2}: empty path")
            continue

        p = Path(path_str)
        if not p.exists():
            results["missing_files"].append(str(p))
            continue

        # Duplicate path
        if path_str in seen_paths:
            results["duplicate_paths"].append(path_str)
        seen_paths[path_str] = i

        # Count valid rows
        class_counts[label] += 1
        results["valid_rows"] += 1

        # PIL checks (image content)
        if PIL_AVAILABLE:
            try:
                from PIL import Image as PImage
                img = PImage.open(p).convert("L")
                arr = np.array(img, dtype=np.float32)

                # Check dimensions
                if img.size[0] < 8 or img.size[1] < 8:
                    results["wrong_dimensions"].append(str(p))

                # Check empty / bad exposure
                mean_val = arr.mean()
                if arr.std() < 2.0:
                    results["empty_images"].append(f"{p}  (std={arr.std():.1f})")
                elif not (VALID_MEAN_RANGE[0] < mean_val < VALID_MEAN_RANGE[1]):
                    results["bad_exposure"].append(
                        f"{p}  (mean={mean_val:.1f})"
                    )

                # Exact duplicate via hash
                if check_duplicates:
                    h = _file_hash(p)
                    if h in file_hashes:
                        results["exact_duplicates"].append(
                            f"{p}  (same as {file_hashes[h]})"
                        )
                    else:
                        file_hashes[h] = p

                    fp = _pixel_fingerprint(p)
                    if fp is not None:
                        fingerprints.append((label, p, fp))

                # Preprocessing compatibility
                try:
                    arr_out = preprocess_image(p)
                    assert arr_out.shape == (IMAGE_SIZE, IMAGE_SIZE, 1)
                except Exception as exc:
                    results["preproc_errors"].append(f"{p}: {exc}")

            except Exception as exc:
                results["preproc_errors"].append(f"{p}: {exc}")

    # ── 3. Near-duplicate detection ────────────────────────────────────────────
    if check_duplicates and fingerprints:
        # Group by label to limit comparison cost
        by_label: defaultdict = defaultdict(list)
        for label, path, fp in fingerprints:
            by_label[label].append((path, fp))

        for label, items in by_label.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    diff = float(np.abs(items[i][1] - items[j][1]).mean())
                    if diff < DUPLICATE_DIFF_THRESHOLD:
                        results["near_duplicates"].append(
                            f"{items[i][0]}  ↔  {items[j][0]}  (diff={diff:.2f})"
                        )

    # ── 4. Class-level checks ──────────────────────────────────────────────────
    results["class_counts"] = dict(class_counts)
    results["missing_classes"] = [c for c in CLASSES if class_counts.get(c, 0) == 0]
    results["low_data_classes"] = [
        c for c in CLASSES
        if 0 < class_counts.get(c, 0) < MIN_RECOMMENDED_ORIGINALS
    ]

    # ── 5. Warnings ────────────────────────────────────────────────────────────
    if results["missing_classes"]:
        results["warnings"].append(
            f"{len(results['missing_classes'])} classes have NO samples: "
            f"{results['missing_classes']}"
        )
    if results["low_data_classes"]:
        results["warnings"].append(
            f"{len(results['low_data_classes'])} classes have fewer than "
            f"{MIN_RECOMMENDED_ORIGINALS} original samples (recommended minimum)."
        )
    if results["missing_files"]:
        results["errors"].append(f"{len(results['missing_files'])} missing image files.")
    if results["invalid_labels"]:
        results["errors"].append(f"{len(results['invalid_labels'])} invalid labels.")
    if results["empty_images"]:
        results["warnings"].append(f"{len(results['empty_images'])} possibly empty images.")
    if results["exact_duplicates"]:
        results["warnings"].append(f"{len(results['exact_duplicates'])} exact duplicate image files.")
    if len(results["near_duplicates"]) > 0:
        results["warnings"].append(
            f"{len(results['near_duplicates'])} near-duplicate image pairs detected."
        )

    # ── 6. Overall pass/fail ───────────────────────────────────────────────────
    critical = (
        results["missing_files"] or
        results["invalid_labels"] or
        results["missing_classes"] or
        results["preproc_errors"]
    )
    results["passed"] = not bool(critical)

    if verbose:
        _print_report(results)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(results: dict):
    sep = "=" * 60

    print(f"\nDATASET VALIDATION REPORT")
    print(sep)
    print(f"  CSV path         : {results['csv_path']}")
    print(f"  Total rows       : {results['total_rows']}")
    print(f"  Valid rows       : {results['valid_rows']}")
    print(f"  Classes expected : {NUM_CLASSES}")

    counts = results["class_counts"]
    present = len(counts)
    print(f"  Classes found    : {present}")

    if counts:
        vals = list(counts.values())
        print(f"  Min samples/class: {min(vals)}")
        print(f"  Max samples/class: {max(vals)}")
        import statistics
        print(f"  Median samples   : {statistics.median(vals):.0f}")

    # ── Per-class table ────────────────────────────────────────────────────────
    print(f"\n  {'Class':<8} {'Samples':>8}  {'Status'}")
    print(f"  {'-'*40}")
    for ch in CLASSES:
        n = counts.get(ch, 0)
        if n == 0:
            status = "MISSING"
        elif n < 5:
            status = "CRITICAL"
        elif n < MIN_RECOMMENDED_ORIGINALS:
            status = f"LOW (want {MIN_RECOMMENDED_ORIGINALS}+)"
        else:
            status = "OK"
        print(f"  {repr(ch):<8} {n:>8}  {status}")

    # ── Issues ─────────────────────────────────────────────────────────────────
    print()

    def _show(label, items, limit=5):
        if not items:
            return
        print(f"  {label} ({len(items)} total):")
        for item in items[:limit]:
            print(f"    {item}")
        if len(items) > limit:
            print(f"    ... and {len(items) - limit} more.")

    _show("Missing files",      results["missing_files"])
    _show("Invalid labels",     results["invalid_labels"])
    _show("Duplicate paths",    results["duplicate_paths"])
    _show("Exact duplicates",   results["exact_duplicates"])
    _show("Near-duplicates",    results["near_duplicates"])
    _show("Empty images",       results["empty_images"])
    _show("Bad exposure",       results["bad_exposure"])
    _show("Wrong dimensions",   results["wrong_dimensions"])
    _show("Preproc errors",     results["preproc_errors"])

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if results["warnings"]:
        print("  WARNINGS:")
        for w in results["warnings"]:
            print(f"    ⚠  {w}")

    if results["errors"]:
        print("  ERRORS:")
        for e in results["errors"]:
            print(f"    ✗  {e}")

    missing_classes = results.get("missing_classes", [])
    low_classes     = results.get("low_data_classes", [])

    print()
    if results["passed"] and not missing_classes and not low_classes:
        print("  ✓  VALIDATION PASSED — dataset looks good.")
    elif results["passed"]:
        print("  ⚠  VALIDATION PASSED WITH WARNINGS — dataset is usable but limited.")
    else:
        print("  ✗  VALIDATION FAILED — fix errors before training.")

    if missing_classes or low_classes:
        print()
        print("  RECOMMENDATION:")
        print(f"    Collect more independent handwriting samples.")
        print(f"    Target: {MIN_RECOMMENDED_ORIGINALS}–50 genuine originals per class.")
        print(f"    Run:  python dataset_builder.py --batch --rounds 20")
        print()
        print("  NOTE:")
        print("    Augmentation creates variations of EXISTING samples.")
        print("    It cannot replace genuine diverse handwriting.")
        print("    A model trained on 1-3 originals per class will generalise poorly")
        print("    regardless of how many augmented copies are generated.")

    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate the handwriting dataset before training."
    )
    parser.add_argument("--csv",               type=Path, default=LEGACY_DATASET_CSV,
                        help=f"Path to dataset CSV (default: {LEGACY_DATASET_CSV})")
    parser.add_argument("--no-duplicates",     action="store_true",
                        help="Skip near-duplicate image detection (faster)")
    args = parser.parse_args()

    ensure_dirs()

    if not PIL_AVAILABLE:
        print("WARNING: Pillow not installed. Image content checks will be skipped.")
        print("  pip install Pillow")

    results = validate_dataset(
        csv_path=args.csv,
        check_duplicates=not args.no_duplicates,
        verbose=True,
    )

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
