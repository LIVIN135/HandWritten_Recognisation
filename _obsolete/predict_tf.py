"""
🔍  STEP 3: Recognize Handwritten Characters (Predict)
=======================================================
This script loads your trained model and recognizes
handwritten letters and numbers from new images.

THREE WAYS TO USE IT:
---------------------
  1. Test a single image:
     python predict_tf.py --image my_photo.png

  2. Test a whole folder of images:
     python predict_tf.py --folder my_test_images/

  3. Draw and predict live (interactive window):
     python predict_tf.py --draw

HOW TO RUN:
-----------
  pip install tensorflow pillow numpy matplotlib

  python predict_tf.py --draw
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────
# ⚙️  SETTINGS
# ──────────────────────────────────────────────────────────────

MODEL_PATH  = "my_model/model.keras"
LABELS_PATH = "my_model/labels.json"
IMAGE_SIZE  = 28

# ──────────────────────────────────────────────────────────────


# ── LOAD MODEL & LABELS ───────────────────────────────────────

def load_model_and_labels():
    if not Path(MODEL_PATH).exists():
        print(f"❌  Model not found at '{MODEL_PATH}'")
        print("    Run train_tf.py first!")
        exit(1)
    if not Path(LABELS_PATH).exists():
        print(f"❌  Labels not found at '{LABELS_PATH}'")
        exit(1)

    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        classes = json.load(f)["classes"]

    print(f"✅  Model loaded  ({len(classes)} classes: {classes})")
    return model, classes


# ── PREPROCESS IMAGE ──────────────────────────────────────────

def preprocess(img: Image.Image) -> np.ndarray:
    """Resize, grayscale, and normalize an image for the model."""
    img = img.convert("L")                            # grayscale
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))        # 28×28
    arr = np.array(img, dtype=np.float32) / 255.0    # scale 0→1
    return arr.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)  # add batch + channel dims


# ── PREDICT ONE IMAGE ─────────────────────────────────────────

def predict_image(model, classes, img: Image.Image):
    """Run the model and return (predicted_label, confidence, all_scores)."""
    x = preprocess(img)
    scores = model.predict(x, verbose=0)[0]          # shape: (num_classes,)
    idx    = int(np.argmax(scores))
    return classes[idx], float(scores[idx]), scores


# ── MODE 1: SINGLE IMAGE ──────────────────────────────────────

def predict_single(model, classes, image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"❌  Image not found: {image_path}")
        return

    img   = Image.open(path)
    label, confidence, scores = predict_image(model, classes, img)

    print(f"\n🖼️   Image     : {path.name}")
    print(f"🔤  Prediction : {label}")
    print(f"📊  Confidence : {confidence:.1%}")

    # Show top 5 guesses
    top5_idx = np.argsort(scores)[::-1][:5]
    print(f"\n   Top 5 guesses:")
    for i, idx in enumerate(top5_idx):
        bar = "█" * int(scores[idx] * 20)
        print(f"   {i+1}. {classes[idx]:>4s}  {bar}  {scores[idx]:.1%}")

    # Show the image with prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(img.convert("L"), cmap="gray")
    plt.title(f"Predicted: '{label}'  ({confidence:.1%} confident)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ── MODE 2: FOLDER OF IMAGES ──────────────────────────────────

def predict_folder(model, classes, folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌  Folder not found: {folder_path}")
        return

    image_files = list(folder.glob("*.png")) + \
                  list(folder.glob("*.jpg")) + \
                  list(folder.glob("*.jpeg"))

    if not image_files:
        print(f"❌  No images found in {folder_path}")
        return

    print(f"\n📂  Found {len(image_files)} images in '{folder}'\n")

    correct, total = 0, 0
    results = []

    for img_path in sorted(image_files):
        img = Image.open(img_path)
        label, confidence, _ = predict_image(model, classes, img)
        results.append((img_path.name, label, confidence))

        # Try to check if filename contains the true label (e.g. "A_001.png")
        true_label = img_path.stem.split("_")[0]
        is_correct = true_label == label
        mark = "✅" if is_correct else "❌"

        if true_label in classes:
            correct += is_correct
            total += 1
            print(f"  {mark}  {img_path.name:<25}  →  '{label}'  ({confidence:.1%})")
        else:
            print(f"  🔍  {img_path.name:<25}  →  '{label}'  ({confidence:.1%})")

    if total > 0:
        print(f"\n📊  Accuracy on labelled images: {correct}/{total} = {correct/total:.1%}")

    # Show a grid of results
    cols = min(5, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).flatten()

    for i, (img_path_full) in enumerate(sorted(image_files)):
        img = Image.open(img_path_full).convert("L")
        name, label, conf = results[i]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"'{label}' {conf:.0%}", fontsize=9)
        axes[i].axis("off")

    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Handwriting Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── MODE 3: DRAW & PREDICT LIVE ───────────────────────────────

def predict_draw(model, classes):
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        print("❌  Tkinter not available. Try --image or --folder instead.")
        return

    CANVAS_PX = 280   # drawing area size
    last = {"x": None, "y": None}

    root = tk.Tk()
    root.title("✏️  Draw a character — I'll guess it!")
    root.resizable(False, False)

    # ── top label ──────────────────────────────────────────────
    result_var = tk.StringVar(value="Draw a letter or number below ✍️")
    tk.Label(root, textvariable=result_var,
             font=("Helvetica", 16, "bold"), pady=8).pack()

    conf_var = tk.StringVar(value="")
    tk.Label(root, textvariable=conf_var,
             font=("Helvetica", 11), fg="#555").pack()

    # ── canvas ─────────────────────────────────────────────────
    canvas = tk.Canvas(root, width=CANVAS_PX, height=CANVAS_PX,
                       bg="white", cursor="crosshair",
                       highlightthickness=2, highlightbackground="#aaa")
    canvas.pack(padx=12, pady=8)

    # ── top-5 bar display ──────────────────────────────────────
    bar_frame = tk.Frame(root)
    bar_frame.pack(fill="x", padx=16, pady=(0, 4))
    bar_labels = []
    for _ in range(5):
        lbl = tk.Label(bar_frame, text="", font=("Courier", 10), anchor="w")
        lbl.pack(fill="x")
        bar_labels.append(lbl)

    # ── buttons ────────────────────────────────────────────────
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)

    def clear():
        canvas.delete("all")
        result_var.set("Draw a letter or number below ✍️")
        conf_var.set("")
        for lbl in bar_labels:
            lbl.config(text="")
        last["x"] = last["y"] = None

    def run_predict():
        # Capture canvas as image via PostScript → PIL
        ps = canvas.postscript(colormode="gray")
        try:
            import io
            from PIL import EpsImagePlugin
            img = Image.open(io.BytesIO(ps.encode("utf-8"))).convert("L")
        except Exception:
            img = Image.new("L", (CANVAS_PX, CANVAS_PX), 255)

        img = ImageOps.invert(img)   # white background → black background
        label, confidence, scores = predict_image(model, classes, img)

        result_var.set(f"I think that's:  '{label}'")
        conf_var.set(f"Confidence: {confidence:.1%}")

        # Update top-5 bars
        top5 = np.argsort(scores)[::-1][:5]
        for i, idx in enumerate(top5):
            bar = "█" * int(scores[idx] * 25)
            bar_labels[i].config(
                text=f"  {classes[idx]:>4s}  {bar:<25}  {scores[idx]:.1%}"
            )

    tk.Button(btn_frame, text="🔍 Predict  [P]", command=run_predict,
              bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
              padx=12, pady=4).pack(side="left", padx=8)

    tk.Button(btn_frame, text="🗑 Clear  [C]", command=clear,
              bg="#f44336", fg="white", font=("Helvetica", 12),
              padx=12, pady=4).pack(side="left", padx=8)

    # keyboard shortcuts
    root.bind("<p>", lambda e: run_predict())
    root.bind("<P>", lambda e: run_predict())
    root.bind("<c>", lambda e: clear())
    root.bind("<C>", lambda e: clear())

    # drawing
    def paint(event):
        r = 10
        x, y = event.x, event.y
        if last["x"]:
            canvas.create_line(last["x"], last["y"], x, y,
                               fill="black", width=20,
                               capstyle=tk.ROUND, smooth=True)
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        last["x"], last["y"] = x, y

    def reset(event):
        last["x"] = last["y"] = None

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<ButtonRelease-1>", reset)

    print("\n✏️   Drawing window open!")
    print("   Draw a character → click Predict (or press P)")
    print("   Press C to clear and draw again\n")
    root.mainloop()


# ── ENTRY POINT ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Handwriting Predictor")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a single image to predict")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to a folder of images to predict")
    parser.add_argument("--draw",   action="store_true",
                        help="Open a drawing window and predict live")
    args = parser.parse_args()

    if not args.image and not args.folder and not args.draw:
        print("\n✏️   Handwriting Predictor")
        print("=" * 35)
        print("Usage:")
        print("  python predict_tf.py --image  path/to/image.png")
        print("  python predict_tf.py --folder path/to/folder/")
        print("  python predict_tf.py --draw")
        return

    model, classes = load_model_and_labels()

    if args.image:
        predict_single(model, classes, args.image)

    if args.folder:
        predict_folder(model, classes, args.folder)

    if args.draw:
        predict_draw(model, classes)


if __name__ == "__main__":
    main()