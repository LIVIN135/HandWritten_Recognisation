"""
✏️  STEP 1: Make More Training Images (Augmentation)
======================================================
This script takes your collected handwriting images and
creates many variations of each one — so your model learns better.

Think of it like practising different styles of the same letter!

HOW TO RUN:
-----------
1. Install requirements:
   pip install Pillow numpy

2. Run this script:
   python augment_simple.py

That's it! It will read your dataset.csv and create augmented_dataset.csv
"""

import csv
import os
import uuid
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

# ──────────────────────────────────────────────────────────────
# ⚙️  SETTINGS — change these if you want
# ──────────────────────────────────────────────────────────────

INPUT_CSV       = "dataset.csv"          # your collected data
OUTPUT_CSV      = "augmented_dataset.csv"  # output (bigger dataset)
OUTPUT_FOLDER   = "augmented_images"     # where new images are saved
COPIES_PER_IMAGE = 8                     # how many versions per image

# ──────────────────────────────────────────────────────────────


def make_variation(img):
    """Take one image and return a slightly different version of it."""

    # 1. Rotate a little (like tilting your pen slightly)
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, fillcolor=0)

    # 2. Shift the character left/right/up/down slightly
    arr = np.array(img)
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    arr = np.roll(arr, shift_y, axis=0)
    arr = np.roll(arr, shift_x, axis=1)
    img = Image.fromarray(arr)

    # 3. Make it slightly brighter or darker
    brightness = random.uniform(0.75, 1.25)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 4. Add a tiny bit of random noise (like paper texture)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def main():
    print("\n✏️   Handwriting Dataset Augmenter")
    print("=" * 40)

    # Check input file exists
    if not Path(INPUT_CSV).exists():
        print(f"\n❌  Could not find '{INPUT_CSV}'")
        print("    Make sure you have collected data first using the dataset builder.")
        return

    # Read all rows from dataset.csv
    rows = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if len(rows) == 0:
        print("❌  Your dataset.csv is empty. Collect some data first!")
        return

    print(f"\n📂  Found {len(rows)} original images")
    print(f"🔁  Creating {COPIES_PER_IMAGE} variations each")
    print(f"📦  Total after augmentation: ~{len(rows) * COPIES_PER_IMAGE} images\n")

    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    output_rows = []

    # Keep original images in the output too
    output_rows.extend(rows)

    # Create variations
    skipped = 0
    for i, row in enumerate(rows):
        img_path = Path(row["image_path"])
        label    = row["label"]

        if not img_path.exists():
            skipped += 1
            continue

        img = Image.open(img_path).convert("L")  # open as grayscale

        # Make a folder for this label
        label_folder = Path(OUTPUT_FOLDER) / label
        label_folder.mkdir(exist_ok=True)

        for _ in range(COPIES_PER_IMAGE):
            varied = make_variation(img)
            filename = f"{label}_aug_{uuid.uuid4().hex[:6]}.png"
            save_path = label_folder / filename
            varied.save(save_path)
            output_rows.append({"image_path": str(save_path), "label": label})

        # Show progress every 20 images
        if (i + 1) % 20 == 0:
            print(f"  ⏳  {i + 1}/{len(rows)} done...")

    # Save the new CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✅  Done!")
    print(f"   Original images  : {len(rows)}")
    if skipped:
        print(f"   Skipped (missing): {skipped}")
    print(f"   New total        : {len(output_rows)}")
    print(f"   Saved to         : {OUTPUT_CSV}")
    print(f"\n👉  Next step: run  python train_simple.py")


if __name__ == "__main__":
    main()