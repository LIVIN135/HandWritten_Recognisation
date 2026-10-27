"""

preprocessing.py
================
Single shared preprocessing function used during:
  - dataset loading (train/val/test)
  - live prediction
  - feedback collection

Pipeline:
  1. Convert to grayscale
  2. Correct polarity (ensure white background, black ink)
  3. Find character bounding box
  4. Add small margin
  5. Crop to bounding box (preserving aspect ratio)
  6. Pad to square
  7. Resize to IMAGE_SIZE × IMAGE_SIZE
  8. Apply autocontrast (normalise contrast)
  9. Scale pixel values to [0, 1]
  10. Return float32 array shape (IMAGE_SIZE, IMAGE_SIZE, 1)

Every path that feeds data into the model MUST use preprocess_image().
"""
from __future__ import annotations
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

from config_kiro import IMAGE_SIZE, BACKGROUND_PIXEL, INK_PIXEL, CROP_MARGIN, PREPROC_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# CORE PREPROCESSING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    source: Union[Image.Image, np.ndarray, str, Path],
    *,
    image_size: int = IMAGE_SIZE,
    crop_margin: int = CROP_MARGIN,
    save_preproc_config: bool = False,
) -> np.ndarray:
    """
    Preprocess a handwritten character image for model input.

    Parameters
    ----------
    source : PIL Image, numpy array, file path (str or Path)
    image_size : target side length in pixels (default IMAGE_SIZE from config)
    crop_margin : extra white-space pixels added around bounding box
    save_preproc_config : if True, write params to PREPROC_CONFIG path

    Returns
    -------
    numpy array, shape (image_size, image_size, 1), dtype float32, values in [0, 1]
    """
    # --- Step 1: Load as PIL grayscale -----------------------------------------
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("L")
    elif isinstance(source, np.ndarray):
        # Accept (H,W), (H,W,1), (H,W,3)
        if source.ndim == 3 and source.shape[2] == 1:
            source = source[:, :, 0]
        if source.ndim == 3:
            img = Image.fromarray(source.astype(np.uint8)).convert("L")
        else:
            img = Image.fromarray(source.astype(np.uint8))
    elif isinstance(source, Image.Image):
        img = source.convert("L")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # --- Step 2: Correct polarity -----------------------------------------------
    # We want WHITE background (255), BLACK ink (0).
    # If the mean pixel is dark (< 128), the image is inverted – flip it.
    arr = np.array(img, dtype=np.float32)
    if arr.mean() < 128:
        arr = 255.0 - arr
        img = Image.fromarray(arr.astype(np.uint8))

    # --- Step 3: Find bounding box of ink pixels --------------------------------
    # Ink pixels are dark (< threshold). We look for pixels below a threshold.
    threshold = 200  # pixels darker than this are considered ink
    ink_mask = arr < threshold

    if ink_mask.any():
        rows = np.where(ink_mask.any(axis=1))[0]
        cols = np.where(ink_mask.any(axis=0))[0]
        r_min, r_max = int(rows[0]),  int(rows[-1])
        c_min, c_max = int(cols[0]),  int(cols[-1])

        # --- Step 4: Add margin -------------------------------------------------
        h, w = arr.shape
        r_min = max(0, r_min - crop_margin)
        r_max = min(h - 1, r_max + crop_margin)
        c_min = max(0, c_min - crop_margin)
        c_max = min(w - 1, c_max + crop_margin)

        # --- Step 5: Crop -------------------------------------------------------
        img = img.crop((c_min, r_min, c_max + 1, r_max + 1))
    # If no ink found, keep the original (empty canvas)

    # --- Step 6: Pad to square (preserve aspect ratio) -------------------------
    img = _pad_to_square(img, fill=BACKGROUND_PIXEL)

    # --- Step 7: Resize ---------------------------------------------------------
    img = img.resize((image_size, image_size), Image.LANCZOS)

    # --- Step 8: Autocontrast ---------------------------------------------------
    # Only when there is actual ink; skips blank images.
    arr2 = np.array(img, dtype=np.uint8)
    if arr2.min() < 200:    # some dark pixels present
        img = ImageOps.autocontrast(img)

    # --- Step 9: Scale to [0, 1] ------------------------------------------------
    out = np.array(img, dtype=np.float32) / 255.0

    # --- Step 10: Add channel dimension -----------------------------------------
    out = out[:, :, np.newaxis]   # (H, W, 1)

    if save_preproc_config:
        _save_config(image_size, crop_margin)

    return out


def _pad_to_square(img: Image.Image, fill: int = 255) -> Image.Image:
    """Pad an image to a square by adding background-coloured borders."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    new_img = Image.new("L", (side, side), fill)
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def _save_config(image_size: int, crop_margin: int):
    """Persist preprocessing parameters so prediction uses identical settings."""
    cfg = {
        "image_size": image_size,
        "crop_margin": crop_margin,
        "background_pixel": BACKGROUND_PIXEL,
        "ink_pixel": INK_PIXEL,
        "polarity": "white_bg_black_ink",
        "normalisation": "divide_by_255",
    }
    PREPROC_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROC_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)


def load_preproc_config() -> dict:
    """Load preprocessing parameters saved during training."""
    if PREPROC_CONFIG.exists():
        with open(PREPROC_CONFIG) as f:
            return json.load(f)
    return {
        "image_size": IMAGE_SIZE,
        "crop_margin": CROP_MARGIN,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PREPROCESSING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_batch(
    paths: list,
    labels: list,
    *,
    image_size: int = IMAGE_SIZE,
    crop_margin: int = CROP_MARGIN,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a list of image paths.

    Parameters
    ----------
    paths  : list of str or Path
    labels : list of int (encoded class indices)
    image_size, crop_margin : passed to preprocess_image()

    Returns
    -------
    X : float32 array (N, image_size, image_size, 1)
    y : int array (N,)
    """
    X, y = [], []
    skipped = 0
    for path, label in zip(paths, labels):
        try:
            arr = preprocess_image(
                path,
                image_size=image_size,
                crop_margin=crop_margin,
            )
            X.append(arr)
            y.append(label)
        except Exception as exc:
            if verbose:
                print(f"  WARNING: skipped {path} — {exc}")
            skipped += 1

    if skipped and verbose:
        print(f"  {skipped} images skipped during batch preprocessing.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# CANVAS-TO-ARRAY  (used by the Tkinter drawing window)
# ──────────────────────────────────────────────────────────────────────────────

def canvas_to_array(canvas, canvas_size: int = 280) -> np.ndarray:
    """
    Render a Tkinter canvas's drawn items onto a PIL image and preprocess it.

    This method does NOT use screenshots or PostScript – it redraws each
    canvas item (lines, ovals) onto a PIL Image using ImageDraw.
    This matches exactly how handwriting_dataset_builder.py saves images.

    Returns
    -------
    float32 array shape (IMAGE_SIZE, IMAGE_SIZE, 1) ready for model.predict()
    """
    from PIL import ImageDraw

    img = Image.new("L", (canvas_size, canvas_size), color=BACKGROUND_PIXEL)
    draw = ImageDraw.Draw(img)

    for item in canvas.find_all():
        coords    = canvas.coords(item)
        item_type = canvas.type(item)
        cfg       = canvas.itemconfig(item)
        opts      = {k: v[-1] for k, v in cfg.items()}
        fill      = INK_PIXEL  # black ink

        if item_type == "oval" and len(coords) == 4:
            draw.ellipse(coords, fill=fill)

        elif item_type == "line" and len(coords) >= 4:
            try:
                width = int(float(opts.get("width", 2)))
            except (ValueError, TypeError):
                width = 2
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(points) >= 2:
                draw.line(points, fill=fill, width=width)

    return preprocess_image(img)


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("preprocessing.py  — self-test")
    print("=" * 40)

    # Test 1: synthetic white-on-black image (should be auto-inverted)
    test_arr = np.zeros((100, 100), dtype=np.uint8)
    test_arr[20:80, 30:70] = 255   # white letter on black background
    result = preprocess_image(test_arr)
    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 1), f"Bad shape: {result.shape}"
    assert result.dtype == np.float32, f"Bad dtype: {result.dtype}"
    assert 0.0 <= result.min() and result.max() <= 1.0, "Values out of [0,1]"
    print(f"  Test 1 PASSED  shape={result.shape}  min={result.min():.3f}  max={result.max():.3f}")

    # Test 2: already-correct polarity (black ink on white)
    test_arr2 = np.full((100, 100), 255, dtype=np.uint8)
    test_arr2[20:80, 30:70] = 0    # black letter on white background
    result2 = preprocess_image(test_arr2)
    assert result2.shape == (IMAGE_SIZE, IMAGE_SIZE, 1)
    print(f"  Test 2 PASSED  shape={result2.shape}")

    # Test 3: test from a real file if provided as CLI arg
    if len(sys.argv) > 1:
        path = sys.argv[1]
        result3 = preprocess_image(path)
        print(f"  Test 3 PASSED  file={path}  shape={result3.shape}")

    print("\nAll tests passed.")
