"""
dataset_builder.py
==================
Guided, balanced handwriting data collection tool.

This is an enhanced version of the existing batch_collector_cloude.py,
updated to use the shared preprocessing pipeline and save to the new
data/raw/ directory structure.

Two modes:
  --draw     Open Tkinter GUI for drawing characters
  --import   Import images from a folder (with subfolders named by label)

HOW TO RUN:
-----------
  # Collect 5 rounds of all 62 characters (= 310 new samples)
  python dataset_builder.py --draw --rounds 5

  # Collect 20 rounds (recommended minimum for meaningful training)
  python dataset_builder.py --draw --rounds 20 --vary-brush

  # Import existing images
  python dataset_builder.py --import-dir ./my_scans

OUTPUT:
  data/raw/<label>/<label>_<uuid>.png
  dataset.csv  (appended, not overwritten)

TIPS FOR BETTER DATA:
  - Vary your writing style slightly each round (slant, size, stroke width).
  - Get 1-2 other people to contribute if possible.
  - Don't rush - a careful stroke teaches the model more than a quick scribble.
  - 20-50 genuinely different originals per class is the minimum for good results.
"""

import argparse
import csv
import random
import sys
import uuid
from collections import Counter
from pathlib import Path

from config import (
    CLASSES, NUM_CLASSES, IMAGE_SIZE, CANVAS_SIZE,
    LEGACY_DATASET_CSV, LEGACY_DATASET_IMAGES_DIR,
    RAW_DIR, DATA_DIR, MIN_RECOMMENDED_ORIGINALS,
    ensure_dirs,
)
from preprocessing import preprocess_image

try:
    from PIL import Image, ImageDraw, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

EXPORT_SIZE = IMAGE_SIZE  # 28 px


# ──────────────────────────────────────────────────────────────────────────────
# CSV MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class CSVManager:
    FIELDNAMES = ["image_path", "label", "source"]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def append(self, image_path: str, label: str, source: str = "user_draw"):
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(
                {"image_path": image_path, "label": label, "source": source}
            )

    def count(self) -> int:
        if not self.csv_path.exists():
            return 0
        with open(self.csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)

    def label_counts(self) -> Counter:
        counts: Counter = Counter()
        if not self.csv_path.exists():
            return counts
        with open(self.csv_path) as f:
            for row in csv.DictReader(f):
                counts[row["label"]] += 1
        return counts


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE CAPTURE (canvas → PIL) – shared method used in all tools
# ──────────────────────────────────────────────────────────────────────────────

def canvas_to_pil(canvas, canvas_size: int = CANVAS_SIZE) -> "Image.Image":
    """
    Redraw canvas strokes onto a PIL Image directly.
    Black ink (0) on white background (255) — no screen grab, no Ghostscript.
    This exactly matches how prediction preprocessing works.
    """
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    for item in canvas.find_all():
        coords    = canvas.coords(item)
        item_type = canvas.type(item)
        cfg       = canvas.itemconfig(item)
        opts      = {k: v[-1] for k, v in cfg.items()}

        if item_type == "oval" and len(coords) == 4:
            draw.ellipse(coords, fill=0)

        elif item_type == "line" and len(coords) >= 4:
            try:
                width = int(float(opts.get("width", 2)))
            except (ValueError, TypeError):
                width = 2
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(points) >= 2:
                draw.line(points, fill=0, width=width)

    # Resize to export size — apply same preprocessing as training
    img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)
    if img.getextrema()[0] < 200:
        img = ImageOps.autocontrast(img)
    return img


def save_image(img: "Image.Image", label: str, out_dir: Path) -> Path:
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
    dest = label_dir / filename
    img.save(dest)
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# DATASET QUALITY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_report(csv_path: Path):
    """Print a per-class summary of original sample counts."""
    if not csv_path.exists():
        print("  No dataset CSV found.")
        return

    counts: Counter = Counter()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            counts[row["label"]] += 1

    print(f"\n{'Character':<12} {'Original Samples':<18} {'Status'}")
    print("-" * 55)
    for ch in CLASSES:
        n = counts.get(ch, 0)
        if n == 0:
            status = "MISSING"
        elif n < 5:
            status = "CRITICAL - needs more data"
        elif n < MIN_RECOMMENDED_ORIGINALS:
            status = f"LOW - recommend {MIN_RECOMMENDED_ORIGINALS}+"
        else:
            status = "OK"
        print(f"  {ch!r:<10} {n:<18} {status}")

    total = sum(counts.values())
    present = sum(1 for ch in CLASSES if counts.get(ch, 0) > 0)
    missing = [ch for ch in CLASSES if counts.get(ch, 0) == 0]
    low = [ch for ch in CLASSES if 0 < counts.get(ch, 0) < MIN_RECOMMENDED_ORIGINALS]

    print(f"\n  Total samples   : {total}")
    print(f"  Classes present : {present} / {NUM_CLASSES}")
    if missing:
        print(f"  Missing classes : {missing}")
    if low:
        print(f"  Low-data classes: {len(low)} classes have fewer than {MIN_RECOMMENDED_ORIGINALS} samples")
    if missing or low:
        print(f"\n  RECOMMENDATION: Use  python dataset_builder.py --draw  to collect more data.")
        print(f"  Target: {MIN_RECOMMENDED_ORIGINALS}–50 genuinely different originals per class.")


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE IMPORT MODE
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def import_images(import_dir: Path, out_dir: Path, csv_mgr: CSVManager):
    """
    Walk import_dir. If images are inside sub-folders named after labels
    (e.g. import_dir/A/img1.png) the label is inferred from the folder name.
    Otherwise the user is prompted interactively.
    """
    if not PIL_AVAILABLE:
        print("ERROR: Pillow required.  pip install Pillow")
        sys.exit(1)

    image_files = []
    for ext in VALID_EXTS:
        image_files.extend(import_dir.rglob(f"*{ext}"))
        image_files.extend(import_dir.rglob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {import_dir}")
        return

    print(f"Found {len(image_files)} images in {import_dir}")
    added = 0

    for img_path in sorted(image_files):
        parent_name = img_path.parent.name
        if parent_name != import_dir.name and parent_name in CLASSES:
            label = parent_name
        else:
            label = input(f"  Label for '{img_path.name}' (or ENTER to skip): ").strip()
            if not label or label not in CLASSES:
                if label:
                    print(f"    '{label}' is not a valid class. Skipping.")
                continue

        try:
            img = Image.open(img_path).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)
            dest = save_image(img, label, out_dir)
            csv_mgr.append(str(dest), label, source="import")
            added += 1
            print(f"    Saved '{label}' → {dest.name}")
        except Exception as exc:
            print(f"    ERROR processing {img_path}: {exc}")

    print(f"\nImported {added} images.  Total in CSV: {csv_mgr.count()}")


# ──────────────────────────────────────────────────────────────────────────────
# BATCH DRAWING COLLECTOR  (guided round-robin through all 62 classes)
# ──────────────────────────────────────────────────────────────────────────────

def build_queue(rounds: int, shuffle: bool = True) -> list:
    """One shuffled pass through every label, repeated `rounds` times."""
    queue = []
    for _ in range(rounds):
        batch = list(CLASSES)
        if shuffle:
            random.shuffle(batch)
        queue.extend(batch)
    return queue


class BatchCollectorApp:
    def __init__(self, root: "tk.Tk", queue: list, out_dir: Path,
                 csv_mgr: CSVManager, vary_brush: bool = False):
        self.root       = root
        self.queue      = queue
        self.total      = len(queue)
        self.index      = 0
        self.out_dir    = out_dir
        self.csv_mgr    = csv_mgr
        self.vary_brush = vary_brush
        self.brush_size = 14
        self.last       = {"x": None, "y": None}
        self.saved_count: Counter = Counter()

        root.title("Handwriting Dataset Builder")
        root.resizable(False, False)

        # Status
        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 11), fg="#555").pack(pady=(10, 2))

        # Big character prompt
        self.prompt_var = tk.StringVar()
        tk.Label(root, textvariable=self.prompt_var,
                 font=("Helvetica", 48, "bold")).pack(pady=(0, 8))

        # Count display
        self.count_var = tk.StringVar()
        tk.Label(root, textvariable=self.count_var,
                 font=("Helvetica", 9), fg="#888").pack(pady=(0, 4))

        # Canvas
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", cursor="crosshair",
            highlightthickness=2, highlightbackground="#888",
        )
        self.canvas.pack(padx=10, pady=4)

        # Brush size slider
        brush_row = tk.Frame(root)
        brush_row.pack(pady=(2, 0))
        tk.Label(brush_row, text="Brush:", font=("Helvetica", 9)).pack(side="left")
        self.brush_var = tk.IntVar(value=14)
        tk.Scale(brush_row, from_=6, to=30, orient="horizontal",
                 variable=self.brush_var, length=120).pack(side="left")

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Clear  [C]",    command=self.clear,
                  bg="#f0f0f0", font=("Helvetica", 11), padx=10).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Save & Next  [S]", command=self.save_and_next,
                  bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
                  padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Skip  [K]",    command=self.skip,
                  bg="#ff9800", fg="white", font=("Helvetica", 11), padx=10).pack(side="left", padx=6)

        # Keyboard shortcuts
        self.canvas.bind("<B1-Motion>",      self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_brush)
        root.bind("<s>", lambda _: self.save_and_next())
        root.bind("<S>", lambda _: self.save_and_next())
        root.bind("<c>", lambda _: self.clear())
        root.bind("<C>", lambda _: self.clear())
        root.bind("<k>", lambda _: self.skip())
        root.bind("<K>", lambda _: self.skip())

        self._show_current()

    # ── drawing ────────────────────────────────────────────────────────────────
    def _paint(self, event):
        r = self.brush_var.get() // 2
        x, y = event.x, event.y
        if self.last["x"] is not None:
            self.canvas.create_line(
                self.last["x"], self.last["y"], x, y,
                fill="black", width=self.brush_var.get(),
                capstyle=tk.ROUND, smooth=True,
            )
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.last["x"], self.last["y"] = x, y

    def _reset_brush(self, _event):
        self.last["x"] = self.last["y"] = None

    def clear(self):
        self.canvas.delete("all")
        self.last["x"] = self.last["y"] = None

    # ── navigation ─────────────────────────────────────────────────────────────
    def _show_current(self):
        if self.index >= self.total:
            self._finish()
            return
        label = self.queue[self.index]
        done = self.saved_count[label]
        self.status_var.set(f"Prompt {self.index + 1} / {self.total}  —  draw then press S to save")
        self.prompt_var.set(f"Draw:  {label}")
        self.count_var.set(f"Saved so far for '{label}': {done}")
        if self.vary_brush:
            self.brush_var.set(random.randint(10, 22))

    def save_and_next(self):
        if not self.canvas.find_all():
            messagebox.showwarning("Empty canvas", "Please draw something first!")
            return
        label = self.queue[self.index]
        img = canvas_to_pil(self.canvas)
        dest = save_image(img, label, self.out_dir)
        self.csv_mgr.append(str(dest), label, source="user_draw")
        self.saved_count[label] += 1
        self.index += 1
        self.clear()
        self._show_current()

    def skip(self):
        self.index += 1
        self.clear()
        self._show_current()

    def _finish(self):
        total = self.csv_mgr.count()
        messagebox.showinfo(
            "Collection complete!",
            f"Collected {sum(self.saved_count.values())} new drawings.\n"
            f"Total in dataset: {total} samples.\n\n"
            f"Next step: python dataset_validator.py",
        )
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-CHARACTER DRAWING APP  (more flexible, from existing dataset_builder)
# ──────────────────────────────────────────────────────────────────────────────

class SingleDrawApp:
    """Freeform drawing tool where the user selects the label manually."""

    def __init__(self, root: "tk.Tk", out_dir: Path, csv_mgr: CSVManager):
        self.root    = root
        self.out_dir = out_dir
        self.csv_mgr = csv_mgr
        self.last    = {"x": None, "y": None}

        root.title("Handwriting Dataset Builder — Freeform")

        # Label selector
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=4)
        tk.Label(top, text="Label:", font=("Helvetica", 12)).pack(side="left")
        self.label_var = tk.StringVar(value="A")
        tk.Entry(top, textvariable=self.label_var, width=5,
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=4)

        # Quick-pick grid
        qf = tk.Frame(root)
        qf.pack(fill="x", padx=8, pady=2)
        tk.Label(qf, text="Quick:", font=("Helvetica", 9)).pack(side="left")
        for ch in CLASSES:
            tk.Button(qf, text=ch, width=2, font=("Helvetica", 7),
                      command=lambda c=ch: self.label_var.set(c)
                      ).pack(side="left", padx=1)

        # Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="crosshair",
                                highlightthickness=2, highlightbackground="#888")
        self.canvas.pack(padx=8, pady=4)

        # Brush size
        ctrl = tk.Frame(root)
        ctrl.pack(fill="x", padx=8)
        tk.Label(ctrl, text="Brush:", font=("Helvetica", 9)).pack(side="left")
        self.brush_var = tk.IntVar(value=14)
        tk.Scale(ctrl, from_=4, to=30, orient="horizontal",
                 variable=self.brush_var, length=100).pack(side="left")
        tk.Button(ctrl, text="Clear",    command=self._clear,
                  bg="#f0f0f0").pack(side="left", padx=6)
        tk.Button(ctrl, text="Save [S]", command=self._save,
                  bg="#4CAF50", fg="white",
                  font=("Helvetica", 11, "bold"), padx=10).pack(side="left", padx=4)

        # Status
        self.status_var = tk.StringVar(value="Draw a character and press Save.")
        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 9), fg="#555", anchor="w").pack(
            fill="x", padx=8, pady=(0, 6))

        self.canvas.bind("<B1-Motion>",      self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_brush)
        root.bind("<s>", lambda _: self._save())
        root.bind("<S>", lambda _: self._save())
        root.bind("<c>", lambda _: self._clear())
        root.bind("<C>", lambda _: self._clear())

        self._update_status()

    def _paint(self, event):
        r = self.brush_var.get() // 2
        x, y = event.x, event.y
        if self.last["x"] is not None:
            self.canvas.create_line(self.last["x"], self.last["y"], x, y,
                                    fill="black", width=self.brush_var.get(),
                                    capstyle=tk.ROUND, smooth=True)
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.last["x"], self.last["y"] = x, y

    def _reset_brush(self, _event):
        self.last["x"] = self.last["y"] = None

    def _clear(self):
        self.canvas.delete("all")

    def _save(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("No label", "Enter a label first.")
            return
        if label not in CLASSES:
            messagebox.showwarning("Invalid label",
                f"'{label}' is not a valid class.\nValid: 0-9, A-Z, a-z")
            return
        img  = canvas_to_pil(self.canvas)
        dest = save_image(img, label, self.out_dir)
        self.csv_mgr.append(str(dest), label, source="user_draw")
        self._clear()
        self._update_status(f"Saved '{label}' → {dest.name}")

    def _update_status(self, extra: str = ""):
        counts = self.csv_mgr.label_counts()
        msg = f"Total: {self.csv_mgr.count()} samples | Classes: {len(counts)}"
        if extra:
            msg += f"  |  {extra}"
        self.status_var.set(msg)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Handwriting Dataset Builder — collect labeled character images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--draw",        action="store_true",
                        help="Open drawing GUI")
    parser.add_argument("--batch",       action="store_true",
                        help="Guided batch mode (round-robin through all 62 characters)")
    parser.add_argument("--import-dir",  type=Path, default=None,
                        help="Import images from this folder")
    parser.add_argument("--rounds",      type=int, default=5,
                        help="Rounds in batch mode (default: 5)")
    parser.add_argument("--vary-brush",  action="store_true",
                        help="Randomise brush size each prompt (adds stroke-width variety)")
    parser.add_argument("--out-dir",     type=Path, default=None,
                        help="Override output directory (default: data/raw/)")
    parser.add_argument("--csv",         type=Path, default=None,
                        help="Override CSV output path (default: dataset.csv)")
    parser.add_argument("--report",      action="store_true",
                        help="Print dataset quality report and exit")
    args = parser.parse_args()

    ensure_dirs()
    out_dir  = args.out_dir  or RAW_DIR
    csv_path = args.csv      or LEGACY_DATASET_CSV

    csv_mgr = CSVManager(csv_path)

    if args.report:
        print("\nDATASET QUALITY REPORT")
        print("=" * 55)
        print_dataset_report(csv_path)
        return

    if not args.draw and not args.batch and not args.import_dir:
        parser.print_help()
        print("\nExamples:")
        print("  python dataset_builder.py --batch --rounds 20   # recommended")
        print("  python dataset_builder.py --draw                 # freeform")
        print("  python dataset_builder.py --report               # show counts")
        return

    print(f"\nDataset Builder")
    print("=" * 40)
    print(f"  CSV       : {csv_path}")
    print(f"  Output    : {out_dir}")
    print(f"  Existing  : {csv_mgr.count()} samples")
    print_dataset_report(csv_path)
    print()

    # CLI import
    if args.import_dir:
        if not PIL_AVAILABLE:
            print("ERROR: Pillow required.  pip install Pillow")
            sys.exit(1)
        import_images(args.import_dir, out_dir, csv_mgr)

    # GUI modes
    if args.draw or args.batch:
        if not TK_AVAILABLE:
            print("ERROR: Tkinter not available.")
            sys.exit(1)
        if not PIL_AVAILABLE:
            print("ERROR: Pillow required.  pip install Pillow")
            sys.exit(1)

        root = tk.Tk()

        if args.batch:
            print(f"\nBatch mode: {args.rounds} round(s) × {NUM_CLASSES} characters "
                  f"= {args.rounds * NUM_CLASSES} prompts")
            print("Draw each character → press S to save → automatic progression\n")
            queue = build_queue(args.rounds)
            BatchCollectorApp(root, queue, out_dir, csv_mgr, vary_brush=args.vary_brush)
        else:
            print("\nFreeform mode: select a label and draw freely.\n")
            SingleDrawApp(root, out_dir, csv_mgr)

        root.mainloop()

        # Final summary
        counts = csv_mgr.label_counts()
        print(f"\nSession complete.  {csv_mgr.count()} total samples, "
              f"{len(counts)} classes.\n")
        print_dataset_report(csv_path)


if __name__ == "__main__":
    main()
