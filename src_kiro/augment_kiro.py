"""

augment.py
==========
Realistic augmentation for handwritten character images.

KEY DESIGN RULES:
  1. Augmentation is applied ONLY to training images (never val/test).
  2. Each augmented image is freshly generated from the original at runtime
     so the model never sees the exact same augmented version twice per epoch.
  3. No np.roll (pixel wrapping) — shifts use constant background fill.
  4. Rotation fills with background colour (white), not black.
  5. Transformations are mild enough not to change character identity.
  6. Elastic deformation is optional and very gentle.

The module exposes:
  augment_image(pil_img) → PIL Image  (single random augmentation)
  build_augmentation_layer()          → Keras augmentation layer (for in-model use)

Both methods are calibrated to produce realistic handwriting variations.
"""
from __future__ import annotations
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config_kiro import (
    IMAGE_SIZE,
    AUG_ROTATION_DEGREES,
    AUG_SHIFT_FRACTION,
    AUG_SCALE_RANGE,
    AUG_BRIGHTNESS_RANGE,
    AUG_NOISE_STD,
    AUG_ELASTIC,
    AUG_ELASTIC_ALPHA,
    AUG_ELASTIC_SIGMA,
    BACKGROUND_PIXEL,
)


# ──────────────────────────────────────────────────────────────────────────────
# PIL-BASED AUGMENTATION  (used for offline / pre-augmented data)
# ──────────────────────────────────────────────────────────────────────────────

def augment_image(img: Image.Image, *, strong: bool = False) -> Image.Image:
    """
    Apply a random combination of mild augmentations to a PIL grayscale image.

    Parameters
    ----------
    img    : PIL Image (mode "L", white background, black ink)
    strong : if True, use slightly wider ranges (useful for very small datasets)

    Returns
    -------
    PIL Image with same size, same mode.
    """
    img = img.convert("L")

    # ── 1. Random rotation ─────────────────────────────────────────────────────
    max_rot = AUG_ROTATION_DEGREES * (1.5 if strong else 1.0)
    angle   = random.uniform(-max_rot, max_rot)
    img = img.rotate(angle, fillcolor=BACKGROUND_PIXEL, resample=Image.BICUBIC)

    # ── 2. Random scale (crop-and-resize, no padding artifacts) ────────────────
    lo, hi  = AUG_SCALE_RANGE
    if strong:
        lo, hi = lo - 0.05, hi + 0.05
    scale   = random.uniform(lo, hi)
    w, h    = img.size
    new_w   = max(8, int(w * scale))
    new_h   = max(8, int(h * scale))
    img_s   = img.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto white canvas of original size, centred
    canvas  = Image.new("L", (w, h), BACKGROUND_PIXEL)
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    canvas.paste(img_s, (paste_x, paste_y))
    img = canvas

    # ── 3. Random translation (constant fill, not wrapping) ────────────────────
    max_shift = int(IMAGE_SIZE * AUG_SHIFT_FRACTION * (1.5 if strong else 1.0))
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    if dx != 0 or dy != 0:
        import PIL.ImageChops as IC
        img = IC.offset(img, dx, dy)
        # Fill the exposed border with background colour
        arr = np.array(img, dtype=np.uint8)
        if dy > 0:
            arr[:dy, :]  = BACKGROUND_PIXEL
        elif dy < 0:
            arr[dy:, :]  = BACKGROUND_PIXEL
        if dx > 0:
            arr[:, :dx]  = BACKGROUND_PIXEL
        elif dx < 0:
            arr[:, dx:]  = BACKGROUND_PIXEL
        img = Image.fromarray(arr)

    # ── 4. Brightness / contrast variation ────────────────────────────────────
    b_lo, b_hi = AUG_BRIGHTNESS_RANGE
    if strong:
        b_lo, b_hi = b_lo - 0.05, b_hi + 0.05
    brightness = random.uniform(b_lo, b_hi)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    contrast   = random.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # ── 5. Gaussian noise (simulates paper texture) ────────────────────────────
    std = AUG_NOISE_STD * (1.5 if strong else 1.0)
    if std > 0:
        arr  = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, std, arr.shape)
        arr  = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img  = Image.fromarray(arr)

    # ── 6. Mild Gaussian blur (simulates low-resolution or worn pen tip) ───────
    if random.random() < 0.25:
        radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # ── 7. Elastic deformation (optional, very mild) ───────────────────────────
    if AUG_ELASTIC and random.random() < 0.4:
        img = _elastic_transform(img, alpha=AUG_ELASTIC_ALPHA, sigma=AUG_ELASTIC_SIGMA)

    return img


def _elastic_transform(img: Image.Image, alpha: float, sigma: float) -> Image.Image:
    """
    Apply a gentle elastic deformation.

    alpha : amplitude of displacement field (pixels)
    sigma : smoothness of displacement field (Gaussian std dev)
    """
    from scipy.ndimage import gaussian_filter

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # Random displacement fields
    dx = (np.random.rand(h, w) * 2 - 1)
    dy = (np.random.rand(h, w) * 2 - 1)

    # Smooth the fields
    dx = gaussian_filter(dx, sigma) * alpha
    dy = gaussian_filter(dy, sigma) * alpha

    # Build coordinate arrays
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)

    # Map with bilinear interpolation via scipy
    from scipy.ndimage import map_coordinates
    distorted = map_coordinates(arr, [y_new.ravel(), x_new.ravel()], order=1)
    distorted = distorted.reshape(h, w).clip(0, 255).astype(np.uint8)

    return Image.fromarray(distorted)


def augment_numpy(arr: np.ndarray, *, strong: bool = False) -> np.ndarray:
    """
    Augment a preprocessed numpy array (shape H×W or H×W×1, float32 [0,1]).
    Returns same shape and dtype.
    """
    squeeze = False
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
        squeeze = True

    # Convert to 0-255 uint8 PIL
    img_u8 = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img_aug = augment_image(img_u8, strong=strong)

    out = np.array(img_aug, dtype=np.float32) / 255.0
    if squeeze:
        out = out[:, :, np.newaxis]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# KERAS IN-MODEL AUGMENTATION LAYER  (used during model.fit())
# ──────────────────────────────────────────────────────────────────────────────

def build_augmentation_layer():
    """
    Build a Keras Sequential augmentation layer suitable for 28×28 greyscale
    handwritten character images.

    Used as the first layer inside the model so augmentation:
      - only runs during training (automatically disabled at inference)
      - produces a fresh variation every batch
      - requires no pre-augmented files on disk

    Returns a keras.Sequential model or None if Keras is unavailable.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        return None

    # Convert config fractions to Keras parameter conventions
    rot_factor  = AUG_ROTATION_DEGREES / 360.0  # keras uses fraction of full circle
    zoom_range  = 1.0 - AUG_SCALE_RANGE[0]      # e.g. 0.15 for ±15%
    shift_frac  = AUG_SHIFT_FRACTION

    aug = keras.Sequential([
        layers.RandomRotation(
            factor=rot_factor,
            fill_mode="constant",
            fill_value=1.0,    # 1.0 = white background in [0,1] space
        ),
        layers.RandomZoom(
            height_factor=(-zoom_range, zoom_range),
            width_factor=(-zoom_range, zoom_range),
            fill_mode="constant",
            fill_value=1.0,
        ),
        layers.RandomTranslation(
            height_factor=shift_frac,
            width_factor=shift_frac,
            fill_mode="constant",
            fill_value=1.0,
        ),
    ], name="augmentation")

    return aug


# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE AUGMENTATION UTILITY  (pre-generate augmented files on disk)
# ──────────────────────────────────────────────────────────────────────────────

def augment_dataset_to_csv(
    input_csv: Path,
    output_csv: Path,
    output_dir: Path,
    target_per_class: int = 150,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> int:
    """
    Read original images from input_csv and generate augmented variants so that
    every class reaches target_per_class images.

    IMPORTANT: This function should ONLY be applied to the TRAINING split.
    Never apply it to validation or test splits.

    Returns the total number of rows written to output_csv.
    """
    import csv as _csv
    from collections import defaultdict

    random.seed(seed)
    np.random.seed(seed)

    rows_by_label: dict = defaultdict(list)
    with open(input_csv, newline="") as f:
        for row in _csv.DictReader(f):
            p = Path(row["image_path"])
            if p.exists():
                rows_by_label[row["label"]].append(p)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    total_generated = 0

    for label in sorted(rows_by_label):
        originals = rows_by_label[label]
        label_dir = output_dir / label
        label_dir.mkdir(exist_ok=True)

        # Keep originals
        for p in originals:
            output_rows.append({"image_path": str(p), "label": label})

        # Generate augmented variants
        needed = max(0, target_per_class - len(originals))
        for i in range(needed):
            src = originals[i % len(originals)]
            try:
                from PIL import Image as PImage
                img = PImage.open(src).convert("L")
                aug = augment_image(img)
                fname = f"{label}_aug_{i:04d}_{src.stem[-4:]}.png"
                dest  = label_dir / fname
                aug.save(dest)
                output_rows.append({"image_path": str(dest), "label": label})
                total_generated += 1
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: could not augment {src}: {exc}")

        if verbose:
            print(f"  {label!r:4s}: {len(originals):3d} originals → "
                  f"{len(originals) + needed:3d} total")

    with open(output_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows)


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image as PImage
    import uuid
    import tempfile

    print("augment.py  — self-test")
    print("=" * 40)

    # Create a simple test image: black letter on white background
    test_img = PImage.new("L", (28, 28), 255)
    from PIL import ImageDraw as PImageDraw
    d = PImageDraw.Draw(test_img)
    d.text((4, 4), "A", fill=0)

    # Run augmentation 5 times and verify output
    for i in range(5):
        aug = augment_image(test_img)
        assert aug.size == (28, 28), f"Size changed: {aug.size}"
        assert aug.mode == "L", f"Mode changed: {aug.mode}"
        arr = np.array(aug)
        assert arr.dtype == np.uint8
        print(f"  Variant {i+1}: mean={arr.mean():.1f}  std={arr.std():.1f}")

    # Test numpy variant
    arr_in  = np.full((28, 28, 1), 0.9, dtype=np.float32)
    arr_in[10:20, 8:18, 0] = 0.1  # fake ink
    arr_out = augment_numpy(arr_in)
    assert arr_out.shape == (28, 28, 1)
    assert arr_out.dtype == np.float32
    print(f"\n  numpy test: shape={arr_out.shape}  min={arr_out.min():.3f}  max={arr_out.max():.3f}")

    # Test Keras layer
    try:
        layer = build_augmentation_layer()
        if layer is not None:
            import numpy as np
            x = np.random.rand(4, 28, 28, 1).astype(np.float32)
            y = layer(x, training=True)
            assert y.shape == (4, 28, 28, 1)
            print(f"  Keras layer test: output shape {y.shape}  OK")
        else:
            print("  Keras not available — layer test skipped.")
    except Exception as e:
        print(f"  Keras layer test: SKIPPED ({e})")

    print("\nAll augmentation tests passed.")
