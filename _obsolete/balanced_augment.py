"""
Balanced Dataset Augmenter
===========================
Fixes class-imbalance overfitting (e.g. model always guessing 'A')
by generating the SAME number of augmented images for every class,
regardless of how many original drawings you collected for it.

Why this matters:
------------------
Your old augment_simple.py multiplied every class by a fixed 9x.
If you drew 'A' three times and 'B' once, you ended up with
27 'A' images vs 9 'B' images -> the model sees 'A' three times
more often during training -> it learns to guess 'A' whenever
it's unsure.

This script instead picks a single TARGET_PER_CLASS number and
tops every class up to that number, generating more variations
for classes that have fewer originals.

HOW TO RUN:
-----------
   python balance_augment.py
"""

import csv
import random
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def resolve_path(raw_path: str):
    """
    Resolve an image path from the CSV, tolerating case differences between
    folder names (e.g. 'dataset_images/a/x.png' vs the real
    'dataset_images/A/x.png'). This happens because the CSV was generated on
    a case-insensitive filesystem (Windows/macOS) where 'A' and 'a' folders
    silently merge into one, but on Linux they are separate directories.
    """
    p = Path(raw_path.replace("\\", "/"))
    if p.exists():
        return p

    # Fall back to a case-insensitive search within the parent directory tree.
    parent = p.parent
    grandparent = parent.parent
    if grandparent.exists():
        for candidate_dir in grandparent.iterdir():
            if candidate_dir.is_dir() and candidate_dir.name.lower() == parent.name.lower():
                candidate = candidate_dir / p.name
                if candidate.exists():
                    return candidate
                # also try case-insensitive filename match
                for f in candidate_dir.iterdir():
                    if f.name.lower() == p.name.lower():
                        return f
    return None

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────

INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "balanced_dataset.csv"
OUTPUT_FOLDER = "balanced_images"
TARGET_PER_CLASS = 150   # every class ends up with (about) this many images

# ──────────────────────────────────────────────────────────────


def make_variation(img: Image.Image) -> Image.Image:
    """Return a randomly perturbed version of img (rotation/shift/brightness/noise)."""
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=0)

    arr = np.array(img)
    shift_x = random.randint(-3, 3)
    shift_y = random.randint(-3, 3)
    arr = np.roll(arr, shift_y, axis=0)
    arr = np.roll(arr, shift_x, axis=1)
    img = Image.fromarray(arr)

    brightness = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # mild random scale via crop+resize, keeps 28x28 output
    if random.random() < 0.5:
        scale = random.uniform(0.85, 1.15)
        size = img.size
        new_size = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
        img = img.resize(new_size).resize(size)

    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def main():
    print("\nBalanced Dataset Augmenter")
    print("=" * 40)

    if not Path(INPUT_CSV).exists():
        print(f"\nERROR: could not find '{INPUT_CSV}'. Collect data first.")
        return

    rows_by_label = defaultdict(list)
    unresolved = 0
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            p = resolve_path(row["image_path"])
            if p is not None:
                rows_by_label[row["label"]].append(p)
            else:
                unresolved += 1

    if unresolved:
        print(f"\nWARNING: {unresolved} row(s) in {INPUT_CSV} point to files "
              f"that could not be found (even case-insensitively) and were skipped.")

    if not rows_by_label:
        print("ERROR: dataset.csv has no valid image paths.")
        return

    print(f"\nFound {len(rows_by_label)} classes")
    counts = {k: len(v) for k, v in rows_by_label.items()}
    print(f"Original counts per class: min={min(counts.values())}, "
          f"max={max(counts.values())}, target={TARGET_PER_CLASS}")

    if min(counts.values()) < 2:
        print("\nWARNING: some classes have only 1 original drawing.")
        print("Augmentation can only create variations of that ONE sample -")
        print("it cannot invent new handwriting styles. For best results,")
        print("go back to handwriting_dataset_builder.py and collect at")
        print("least 5-10 original drawings for every class.\n")

    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    output_rows = []

    for label, paths in sorted(rows_by_label.items()):
        label_folder = Path(OUTPUT_FOLDER) / label
        label_folder.mkdir(exist_ok=True)

        # keep the originals
        for p in paths:
            output_rows.append({"image_path": str(p), "label": label})

        needed = max(0, TARGET_PER_CLASS - len(paths))
        for i in range(needed):
            src = paths[i % len(paths)]  # cycle through originals evenly
            img = Image.open(src).convert("L")
            varied = make_variation(img)
            filename = f"{label}_bal_{uuid.uuid4().hex[:6]}.png"
            save_path = label_folder / filename
            varied.save(save_path)
            output_rows.append({"image_path": str(save_path), "label": label})

        print(f"  {label:>3s}: {len(paths)} originals -> "
              f"{len(paths) + needed} total")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone! Balanced dataset saved to '{OUTPUT_CSV}'")
    print(f"Total images: {len(output_rows)}")
    print("\nNext step: python train_simple.py")


if __name__ == "__main__":
    main()