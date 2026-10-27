"""
🔍  STEP 3 (UPGRADED): Predict with your trained model
=======================================================
Works with model.pkl from train_emnist.py

THREE WAYS TO USE:
  python predict_emnist.py --draw              ← draw in a window and predict live
  python predict_emnist.py --image my.png      ← predict a single image
  python predict_emnist.py --folder ./images/  ← predict a whole folder

REQUIREMENTS:
  pip install numpy pillow scikit-learn matplotlib
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_PATH  = "my_model/model.pkl"
LABELS_PATH = "my_model/labels.json"
IMAGE_SIZE  = 28


# ── Load model ────────────────────────────────────────────────

def load_model():
    if not Path(MODEL_PATH).exists():
        print(f"❌  Model not found at '{MODEL_PATH}'")
        print("    Run train_emnist.py first!")
        exit(1)
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    le    = data["label_encoder"]
    print(f"✅  Model loaded  ({len(le.classes_)} classes: {list(le.classes_)})")
    return model, le


# ── Preprocess ────────────────────────────────────────────────

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Make sure background is light, ink is dark (same as training)
    if arr.mean() < 0.5:
        arr = 1.0 - arr
    return arr.flatten().reshape(1, -1)


def predict_image(model, le, img: Image.Image):
    x = preprocess(img)
    proba = model.predict_proba(x)[0]
    idx   = int(np.argmax(proba))
    return le.classes_[idx], float(proba[idx]), proba, le.classes_


# ── Mode 1: Single image ──────────────────────────────────────

def predict_single(model, le, image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"❌  Image not found: {image_path}")
        return
    img = Image.open(path)
    label, conf, proba, classes = predict_image(model, le, img)
    print(f"\n🖼️   Image     : {path.name}")
    print(f"🔤  Prediction : {label}")
    print(f"📊  Confidence : {conf:.1%}")
    top5 = np.argsort(proba)[::-1][:5]
    print(f"\n   Top 5 guesses:")
    for i, idx in enumerate(top5):
        bar = "█" * int(proba[idx] * 20)
        print(f"   {i+1}. {classes[idx]:>4s}  {bar}  {proba[idx]:.1%}")


# ── Mode 2: Folder ────────────────────────────────────────────

def predict_folder(model, le, folder_path: str):
    import matplotlib.pyplot as plt
    folder = Path(folder_path)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    if not image_files:
        print(f"❌  No images in {folder_path}")
        return

    results = []
    for p in sorted(image_files):
        img = Image.open(p)
        label, conf, _, _ = predict_image(model, le, img)
        results.append((p, label, conf))
        print(f"  🔍  {p.name:<25}  →  '{label}'  ({conf:.1%})")

    cols = min(5, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).flatten()
    for i, (p, lbl, conf) in enumerate(results):
        axes[i].imshow(Image.open(p).convert("L"), cmap="gray")
        axes[i].set_title(f"'{lbl}' {conf:.0%}", fontsize=9)
        axes[i].axis("off")
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Handwriting Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Mode 3: Draw & predict live ───────────────────────────────

def predict_draw(model, le):
    try:
        import tkinter as tk
        from PIL import ImageDraw as PilDraw
    except ImportError:
        print("❌  Tkinter not available.")
        return

    CANVAS_PX = 280
    last = {"x": None, "y": None}
    canvas_items = []   # track drawn items for PIL replay

    root = tk.Tk()
    root.title("✏️  Draw a character — I'll guess it!")
    root.resizable(False, False)

    result_var = tk.StringVar(value="Draw a letter or number below ✍️")
    tk.Label(root, textvariable=result_var,
             font=("Helvetica", 16, "bold"), pady=8).pack()

    conf_var = tk.StringVar(value="")
    tk.Label(root, textvariable=conf_var,
             font=("Helvetica", 11), fg="#555").pack()

    canvas = tk.Canvas(root, width=CANVAS_PX, height=CANVAS_PX,
                       bg="white", cursor="crosshair",
                       highlightthickness=2, highlightbackground="#aaa")
    canvas.pack(padx=12, pady=8)

    bar_frame = tk.Frame(root)
    bar_frame.pack(fill="x", padx=16, pady=(0, 4))
    bar_labels = []
    for _ in range(5):
        lbl = tk.Label(bar_frame, text="", font=("Courier", 10), anchor="w")
        lbl.pack(fill="x")
        bar_labels.append(lbl)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)

    def clear():
        canvas.delete("all")
        canvas_items.clear()
        result_var.set("Draw a letter or number below ✍️")
        conf_var.set("")
        for lbl in bar_labels:
            lbl.config(text="")
        last["x"] = last["y"] = None

    def canvas_to_pil():
        """Replay canvas items onto PIL image — no screen grab needed."""
        img = Image.new("L", (CANVAS_PX, CANVAS_PX), color=255)
        draw = PilDraw.Draw(img)
        for item in canvas.find_all():
            coords = canvas.coords(item)
            item_type = canvas.type(item)
            cfg = canvas.itemconfig(item)
            opts = {k: v[-1] for k, v in cfg.items()}
            if item_type == "oval" and len(coords) == 4:
                draw.ellipse(coords, fill=0)
            elif item_type == "line" and len(coords) >= 4:
                try:
                    w = int(float(opts.get("width", 2)))
                except Exception:
                    w = 2
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                if len(points) >= 2:
                    draw.line(points, fill=0, width=w)
        return img

    def run_predict():
        img = canvas_to_pil()
        label, conf, proba, classes = predict_image(model, le, img)
        result_var.set(f"I think that's:  '{label}'")
        conf_var.set(f"Confidence: {conf:.1%}")
        top5 = np.argsort(proba)[::-1][:5]
        for i, idx in enumerate(top5):
            bar = "█" * int(proba[idx] * 25)
            bar_labels[i].config(
                text=f"  {classes[idx]:>4s}  {bar:<25}  {proba[idx]:.1%}"
            )

    tk.Button(btn_frame, text="🔍 Predict  [P]", command=run_predict,
              bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
              padx=12, pady=4).pack(side="left", padx=8)
    tk.Button(btn_frame, text="🗑 Clear  [C]", command=clear,
              bg="#f44336", fg="white", font=("Helvetica", 12),
              padx=12, pady=4).pack(side="left", padx=8)

    root.bind("<p>", lambda e: run_predict())
    root.bind("<P>", lambda e: run_predict())
    root.bind("<c>", lambda e: clear())
    root.bind("<C>", lambda e: clear())

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
    print("   Press C to clear\n")
    root.mainloop()


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Handwriting Predictor (sklearn)")
    parser.add_argument("--image",  type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--draw",   action="store_true")
    args = parser.parse_args()

    if not args.image and not args.folder and not args.draw:
        print("Usage:")
        print("  python predict_emnist.py --draw")
        print("  python predict_emnist.py --image path/to/image.png")
        print("  python predict_emnist.py --folder path/to/folder/")
        return

    model, le = load_model()

    if args.image:  predict_single(model, le, args.image)
    if args.folder: predict_folder(model, le, args.folder)
    if args.draw:   predict_draw(model, le)


if __name__ == "__main__":
    main()