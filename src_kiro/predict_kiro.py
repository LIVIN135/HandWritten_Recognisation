"""

predict.py
==========
Live prediction with feedback UI for the 62-class handwriting recognizer.

Supports three modes:
  python predict_kiro.py --draw              Draw in Tkinter window (with feedback UI)
  python predict_kiro.py --image path.png   Predict a single image file
  python predict_kiro.py --folder ./imgs/   Predict all images in a folder

The drawing window includes:
  - Real-time prediction with confidence score
  - Top-5 predictions bar chart
  - Confidence threshold guard (shows "Uncertain" below threshold)
  - YES / NO feedback buttons
  - Correct-label entry when model is wrong
  - Feedback saved to data/feedback/ for controlled retraining
  - Retrain prompt when buffer reaches threshold
"""
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config_kiro import (
    CLASSES, NUM_CLASSES, CHAR_TO_IDX,
    MODEL_PATH, LABELS_PATH,
    CANVAS_SIZE, IMAGE_SIZE,
    CONFIDENCE_THRESHOLD, TOP_K_PREDICTIONS,
    RETRAIN_AFTER_N_FEEDBACK_SAMPLES, AUTO_RETRAIN,
    ensure_dirs,
)
from preprocessing_kiro import preprocess_image, canvas_to_array


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_labels():
    """Load the trained Keras model and verify label list."""
    try:
        from tensorflow import keras
    except ImportError:
        print("ERROR: TensorFlow not installed.  pip install tensorflow")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run  python train_kiro.py  first.")
        sys.exit(1)

    model = keras.models.load_model(MODEL_PATH)

    # Verify labels
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            saved = json.load(f)["classes"]
        if saved != CLASSES:
            print("WARNING: labels.json differs from config.CLASSES")
            print("  Using config.CLASSES (the canonical source).")

    print(f"Model loaded: {MODEL_PATH}")
    print(f"  Classes: {NUM_CLASSES}  |  Input: {IMAGE_SIZE}x{IMAGE_SIZE}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CORE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_array(model, arr: np.ndarray) -> tuple:
    """
    Run inference on a preprocessed array.

    Parameters
    ----------
    arr : float32 (IMAGE_SIZE, IMAGE_SIZE, 1)  OR  (1, IMAGE_SIZE, IMAGE_SIZE, 1)

    Returns
    -------
    label       : str   top-1 predicted character
    confidence  : float top-1 probability
    scores      : np.ndarray shape (NUM_CLASSES,) all class probabilities
    top_k       : list of (label, score) tuples, sorted descending
    """
    if arr.ndim == 3:
        x = arr[np.newaxis]
    else:
        x = arr

    scores = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(scores))
    label = CLASSES[top_idx]
    conf  = float(scores[top_idx])

    top_k = sorted(
        [(CLASSES[i], float(scores[i])) for i in range(NUM_CLASSES)],
        key=lambda t: t[1], reverse=True
    )[:TOP_K_PREDICTIONS]

    return label, conf, scores, top_k


def predict_image_file(model, path: Path) -> tuple:
    arr = preprocess_image(path)
    return predict_array(model, arr)


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-IMAGE MODE
# ──────────────────────────────────────────────────────────────────────────────

def mode_single_image(model, image_path: str):
    p = Path(image_path)
    if not p.exists():
        print(f"ERROR: File not found: {p}")
        return

    label, conf, _, top_k = predict_image_file(model, p)

    print(f"\nImage      : {p.name}")
    if conf < CONFIDENCE_THRESHOLD:
        print(f"Prediction : UNCERTAIN (confidence {conf:.1%} < threshold {CONFIDENCE_THRESHOLD:.0%})")
    else:
        print(f"Prediction : '{label}'")
        print(f"Confidence : {conf:.1%}")

    print(f"\nTop {TOP_K_PREDICTIONS} predictions:")
    for ch, sc in top_k:
        bar = "█" * int(sc * 30)
        print(f"  {ch!r:>4}  {bar:<30}  {sc:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# FOLDER MODE
# ──────────────────────────────────────────────────────────────────────────────

def mode_folder(model, folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    if not files:
        print(f"No image files found in {folder}")
        return

    print(f"\nFound {len(files)} images in '{folder}'\n")
    correct = total = 0

    for f in files:
        label, conf, _, _ = predict_image_file(model, f)
        uncertain = conf < CONFIDENCE_THRESHOLD

        # Try to infer true label from filename prefix
        true_label = f.stem.split("_")[0]
        if true_label in CLASSES:
            total += 1
            is_right = (label == true_label)
            correct  += is_right
            mark = "✓" if is_right else "✗"
        else:
            mark = "?"

        status = f"UNCERTAIN ({conf:.0%})" if uncertain else f"'{label}' {conf:.0%}"
        print(f"  {mark}  {f.name:<30}  →  {status}")

    if total > 0:
        print(f"\nAccuracy on labelled images: {correct}/{total} = {correct/total:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# DRAW MODE  — Tkinter drawing window with feedback UI
# ──────────────────────────────────────────────────────────────────────────────

def mode_draw(model):
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("ERROR: Tkinter not available.")
        sys.exit(1)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow required.  pip install Pillow")
        sys.exit(1)

    from feedback_manager_kiro import FeedbackManager
    fb_mgr = FeedbackManager()

    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "last_pred"  : None,
        "last_conf"  : 0.0,
        "last_scores": None,
        "last_image" : None,   # numpy array (IMAGE_SIZE, IMAGE_SIZE, 1)
    }

    # ── Root window ────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Handwriting Recognizer  —  Draw & Predict")
    root.resizable(False, False)
    FONT_BIG   = ("Helvetica", 16, "bold")
    FONT_MED   = ("Helvetica", 11)
    FONT_SMALL = ("Courier",   10)

    # ── Prediction display ────────────────────────────────────────────────────
    pred_var = tk.StringVar(value="Draw a character, then press Predict")
    tk.Label(root, textvariable=pred_var, font=FONT_BIG, pady=6).pack()

    conf_var = tk.StringVar(value="")
    tk.Label(root, textvariable=conf_var, font=FONT_MED, fg="#555").pack()

    # ── Drawing canvas ────────────────────────────────────────────────────────
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                       bg="white", cursor="crosshair",
                       highlightthickness=2, highlightbackground="#aaa")
    canvas.pack(padx=12, pady=8)

    # Brush size
    brush_row = tk.Frame(root)
    brush_row.pack(pady=(0, 4))
    tk.Label(brush_row, text="Brush:", font=("Helvetica", 9)).pack(side="left")
    brush_var = tk.IntVar(value=18)
    tk.Scale(brush_row, from_=6, to=36, orient="horizontal",
             variable=brush_var, length=120).pack(side="left")

    # ── Top-K bar display ─────────────────────────────────────────────────────
    bar_frame = tk.LabelFrame(root, text=" Top predictions ", font=("Helvetica", 9))
    bar_frame.pack(fill="x", padx=16, pady=(0, 4))
    bar_labels = []
    for _ in range(TOP_K_PREDICTIONS):
        lbl = tk.Label(bar_frame, text="", font=FONT_SMALL, anchor="w",
                       justify="left")
        lbl.pack(fill="x", padx=6)
        bar_labels.append(lbl)

    # ── Predict / Clear buttons ───────────────────────────────────────────────
    btn_row1 = tk.Frame(root)
    btn_row1.pack(pady=4)

    last = {"x": None, "y": None}

    def do_clear():
        canvas.delete("all")
        pred_var.set("Draw a character, then press Predict")
        conf_var.set("")
        for bl in bar_labels:
            bl.config(text="")
        state["last_pred"] = None
        for w in [yes_btn, no_btn, correct_frame]:
            w.pack_forget()
        last["x"] = last["y"] = None

    def do_predict():
        arr = canvas_to_array(canvas, CANVAS_SIZE)
        state["last_image"] = arr

        label, conf, scores, top_k = predict_array(model, arr)
        state["last_pred"]   = label
        state["last_conf"]   = conf
        state["last_scores"] = scores

        if conf < CONFIDENCE_THRESHOLD:
            pred_var.set(f"Uncertain  (try redrawing)")
            conf_var.set(f"Highest confidence: {conf:.1%}  (threshold: {CONFIDENCE_THRESHOLD:.0%})")
        else:
            pred_var.set(f"Prediction:  '{label}'")
            conf_var.set(f"Confidence: {conf:.1%}")

        for i, (ch, sc) in enumerate(top_k):
            bar = "█" * int(sc * 28)
            bar_labels[i].config(text=f"  {ch!r:>4}  {bar:<28}  {sc:.1%}")

        # Show feedback buttons
        yes_btn.pack(side="left", padx=8)
        no_btn.pack(side="left",  padx=8)

    tk.Button(btn_row1, text="Predict  [P]", command=do_predict,
              bg="#4CAF50", fg="white", font=FONT_MED,
              padx=12, pady=4).pack(side="left", padx=8)
    tk.Button(btn_row1, text="Clear  [C]", command=do_clear,
              bg="#f44336", fg="white", font=FONT_MED,
              padx=12, pady=4).pack(side="left", padx=8)

    # ── Feedback row ─────────────────────────────────────────────────────────
    fb_title = tk.Label(root, text="Is this correct?", font=FONT_MED)
    fb_title.pack()

    fb_row = tk.Frame(root)
    fb_row.pack(pady=4)

    def _save_and_show_status(true_label: str):
        """Save feedback sample and update status display."""
        img_arr = state["last_image"]
        pred    = state["last_pred"]
        conf    = state["last_conf"]

        result = fb_mgr.save_feedback(
            image_array=img_arr,
            true_label=true_label,
            predicted_label=pred,
            confidence=conf,
        )
        # Hide feedback controls
        yes_btn.pack_forget()
        no_btn.pack_forget()
        correct_frame.pack_forget()

        total_fb = fb_mgr.total_count()
        fb_status.config(text=f"Feedback saved: '{true_label}'  "
                              f"(total: {total_fb})",
                         fg="#2196F3")

        # Check retrain threshold
        if total_fb > 0 and total_fb % RETRAIN_AFTER_N_FEEDBACK_SAMPLES == 0:
            if AUTO_RETRAIN:
                fb_status.config(
                    text=f"{total_fb} feedback samples — auto-retraining ...",
                    fg="#FF5722")
            else:
                answer = messagebox.askyesno(
                    "Retrain?",
                    f"{total_fb} new feedback samples collected.\n\n"
                    f"Would you like to retrain the model now?\n"
                    f"(This may take several minutes.)",
                )
                if answer:
                    root.destroy()
                    import subprocess
                    subprocess.Popen([sys.executable, "retrain.py"])
                    return

    def on_yes():
        if state["last_pred"] is None:
            return
        _save_and_show_status(state["last_pred"])

    def on_no():
        if state["last_pred"] is None:
            return
        # Show correct-label entry
        correct_frame.pack(pady=4)
        correct_entry.focus()

    yes_btn = tk.Button(fb_row, text="✓  YES", command=on_yes,
                        bg="#4CAF50", fg="white", font=FONT_MED,
                        padx=14, pady=4)
    no_btn  = tk.Button(fb_row, text="✗  NO", command=on_no,
                        bg="#f44336", fg="white", font=FONT_MED,
                        padx=14, pady=4)
    # Hide initially
    yes_btn.pack_forget()
    no_btn.pack_forget()

    # ── Correct-label entry (shown after NO) ──────────────────────────────────
    correct_frame = tk.Frame(root)

    tk.Label(correct_frame, text="Correct character:", font=FONT_MED).pack(side="left")
    correct_var = tk.StringVar()
    correct_entry = tk.Entry(correct_frame, textvariable=correct_var,
                             width=4, font=("Helvetica", 14, "bold"))
    correct_entry.pack(side="left", padx=6)

    def on_submit_correction():
        ch = correct_var.get().strip()
        if not ch:
            return
        if ch not in CLASSES:
            from tkinter import messagebox as mb
            mb.showwarning("Invalid",
                f"'{ch}' is not a valid class.\nValid: 0-9, A-Z, a-z")
            return
        correct_var.set("")
        _save_and_show_status(ch)

    tk.Button(correct_frame, text="Submit", command=on_submit_correction,
              bg="#2196F3", fg="white", font=FONT_MED,
              padx=10, pady=3).pack(side="left", padx=4)

    correct_frame.pack_forget()

    # Quick-pick grid inside correct_frame
    qf = tk.Frame(root)
    qf.pack(pady=(0, 2))

    # ── Feedback status label ─────────────────────────────────────────────────
    fb_status = tk.Label(root, text="", font=("Helvetica", 9), fg="#2196F3")
    fb_status.pack(pady=(0, 6))

    # Show current feedback count
    total_now = fb_mgr.total_count()
    needed = RETRAIN_AFTER_N_FEEDBACK_SAMPLES - (total_now % RETRAIN_AFTER_N_FEEDBACK_SAMPLES)
    if total_now > 0:
        fb_status.config(
            text=f"Feedback buffer: {total_now} samples  "
                 f"({needed} more until retrain prompt)",
            fg="#888"
        )

    # ── Drawing bindings ──────────────────────────────────────────────────────
    def paint(event):
        r = brush_var.get() // 2
        x, y = event.x, event.y
        if last["x"] is not None:
            canvas.create_line(last["x"], last["y"], x, y,
                               fill="black", width=brush_var.get(),
                               capstyle=tk.ROUND, smooth=True)
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        last["x"], last["y"] = x, y

    def reset_brush(event):
        last["x"] = last["y"] = None

    canvas.bind("<B1-Motion>",      paint)
    canvas.bind("<ButtonRelease-1>", reset_brush)

    root.bind("<p>", lambda _: do_predict())
    root.bind("<P>", lambda _: do_predict())
    root.bind("<c>", lambda _: do_clear())
    root.bind("<C>", lambda _: do_clear())
    root.bind("<Return>", lambda _: on_submit_correction())

    print("\nDrawing window open.")
    print("  Draw a character → press P to predict")
    print("  Then press YES or NO to provide feedback")
    print("  Press C to clear\n")

    root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Handwriting Predictor — 62-class character recognition"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--draw",   action="store_true",
                       help="Open interactive drawing window")
    group.add_argument("--image",  type=str, default=None,
                       help="Predict a single image file")
    group.add_argument("--folder", type=str, default=None,
                       help="Predict all images in a folder")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help=f"Model path (default: {MODEL_PATH})")
    args = parser.parse_args()

    if not args.draw and not args.image and not args.folder:
        parser.print_help()
        print("\nExamples:")
        print("  python predict_kiro.py --draw")
        print("  python predict_kiro.py --image my_char.png")
        print("  python predict_kiro.py --folder ./test_images/")
        return

    ensure_dirs()
    model = load_model_and_labels()

    if args.image:
        mode_single_image(model, args.image)
    elif args.folder:
        mode_folder(model, args.folder)
    elif args.draw:
        mode_draw(model)


if __name__ == "__main__":
    main()
