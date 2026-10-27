"""
Handwriting Dataset Builder
============================
Builds a labeled CSV dataset of handwritten letters and numbers.

Two input modes:
  1. Image import   — point to a folder of scanned/photographed images
  2. Drawing canvas — draw characters directly on screen using a Tkinter GUI

Output: dataset.csv  with columns  [image_path, label]
        Images saved to  ./dataset_images/<label>/

Usage
-----
  # Mode 1 – import images from a folder
  python handwriting_dataset_builder.py --import-dir ./my_photos

  # Mode 2 – draw characters on screen
  python handwriting_dataset_builder.py --draw

  # Both at once
  python handwriting_dataset_builder.py --import-dir ./my_photos --draw

  # Custom output
  python handwriting_dataset_builder.py --draw --out-dir ./out --csv dataset.csv
"""

import argparse
import csv
import os
import shutil
import sys
import uuid
from pathlib import Path
from datetime import datetime

# ── optional heavy imports (only needed for image-import mode) ─────────────
try:
    from PIL import Image, ImageOps, ImageFilter,ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Tkinter (drawing mode) ─────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CANVAS_SIZE = 280          # drawing canvas pixels
EXPORT_SIZE = 28           # saved image size (28×28 like MNIST)

LABELS_ALPHANUMERIC = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz") +
    [str(d) for d in range(10)]
)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class CSVManager:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])

    def append(self, image_path: str, label: str):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([image_path, label])

    def count(self) -> int:
        with open(self.csv_path, "r") as f:
            return max(0, sum(1 for _ in f) - 1)   # subtract header

    def label_counts(self) -> dict:
        counts = {}
        if not self.csv_path.exists():
            return counts
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                counts[label] = counts.get(label, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE IMPORT MODE
# ═══════════════════════════════════════════════════════════════════════════

def import_images(import_dir: Path, out_dir: Path, csv_mgr: CSVManager):
    """
    Walk import_dir. If images are inside sub-folders named after labels
    (e.g.  import_dir/A/img1.png) the label is inferred from the folder name.
    Otherwise the user is prompted interactively.
    """
    if not PIL_AVAILABLE:
        print("❌  Pillow is required for image import.  Run:  pip install Pillow")
        sys.exit(1)

    image_files = []
    for ext in VALID_IMAGE_EXTS:
        image_files.extend(import_dir.rglob(f"*{ext}"))
        image_files.extend(import_dir.rglob(f"*{ext.upper()}"))

    if not image_files:
        print(f"⚠️  No images found in {import_dir}")
        return

    print(f"\n📂  Found {len(image_files)} image(s) in {import_dir}")
    added = 0

    for img_path in sorted(image_files):
        # Try to infer label from parent folder name
        parent_name = img_path.parent.name
        if parent_name != import_dir.name and len(parent_name) <= 3:
            label = parent_name
            print(f"  ↳  {img_path.name}  →  label: '{label}'  (from folder)")
        else:
            label = input(f"  Label for '{img_path.name}' (or ENTER to skip): ").strip()
            if not label:
                continue

        dest = _save_processed_image(img_path, label, out_dir)
        csv_mgr.append(str(dest), label)
        added += 1

    print(f"\n✅  Imported {added} image(s).  Total in CSV: {csv_mgr.count()}")


def _save_processed_image(src: Path, label: str, out_dir: Path) -> Path:
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("L")          # grayscale
    img = ImageOps.autocontrast(img)
    img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)

    filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
    dest = label_dir / filename
    img.save(dest)
    return dest


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWING MODE  (Tkinter GUI)
# ═══════════════════════════════════════════════════════════════════════════

class DrawingApp:
    def __init__(self, root: tk.Tk, out_dir: Path, csv_mgr: CSVManager):
        self.root = root
        self.out_dir = out_dir
        self.csv_mgr = csv_mgr

        self.root.title("✏️  Handwriting Dataset Builder")
        self.root.resizable(False, True)

        self._build_ui()
        self._bind_events()
        self._update_status()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ── label selector ─────────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(fill="x", **pad)

        tk.Label(top, text="Label:", font=("Helvetica", 12)).pack(side="left")
        self.label_var = tk.StringVar(value="A")
        self.label_entry = tk.Entry(top, textvariable=self.label_var, width=5,
                                    font=("Helvetica", 14, "bold"))
        self.label_entry.pack(side="left", padx=4)

        # quick-pick buttons
        qf = tk.Frame(self.root)
        qf.pack(fill="x", padx=8, pady=2)
        tk.Label(qf, text="Quick pick:", font=("Helvetica", 9)).pack(side="left")
        for ch in list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
            btn = tk.Button(qf, text=ch, width=2, font=("Helvetica", 7),
                            command=lambda c=ch: self.label_var.set(c))
            btn.pack(side="left", padx=1)

        # ── canvas ─────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="crosshair",
                                highlightthickness=2, highlightbackground="#888")
        self.canvas.pack(padx=8, pady=4)

        # ── controls ───────────────────────────────────────────────────────
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill="x", **pad)

        tk.Label(ctrl, text="Brush:", font=("Helvetica", 9)).pack(side="left")
        self.brush_var = tk.IntVar(value=14)
        tk.Scale(ctrl, from_=4, to=30, orient="horizontal",
                 variable=self.brush_var, length=100).pack(side="left")

        tk.Button(ctrl, text="🗑 Clear", command=self._clear,
                  bg="#f0f0f0").pack(side="left", padx=6)
        tk.Button(ctrl, text="💾 Save  [S]", command=self._save,
                  bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"),
                  padx=10).pack(side="left", padx=4)
        tk.Button(ctrl, text="📂 Import folder", command=self._import_folder,
                  bg="#2196F3", fg="white").pack(side="left", padx=4)

        # ── status bar ─────────────────────────────────────────────────────
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Helvetica", 9), fg="#555", anchor="w").pack(
            fill="x", padx=8, pady=(0, 6))

        self._last_x = None
        self._last_y = None

    # ── event bindings ─────────────────────────────────────────────────────
    def _bind_events(self):
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_brush)
        self.root.bind("<s>", lambda e: self._save())
        self.root.bind("<S>", lambda e: self._save())
        self.root.bind("<c>", lambda e: self._clear())
        self.root.bind("<C>", lambda e: self._clear())

    # ── drawing ────────────────────────────────────────────────────────────
    def _paint(self, event):
        r = self.brush_var.get() // 2
        x, y = event.x, event.y
        if self._last_x is not None:
            self.canvas.create_line(self._last_x, self._last_y, x, y,
                                    fill="black", width=self.brush_var.get(),
                                    capstyle=tk.ROUND, smooth=True)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="black", outline="black")
        self._last_x, self._last_y = x, y

    def _reset_brush(self, event):
        self._last_x = None
        self._last_y = None

    def _clear(self):
        self.canvas.delete("all")

    # ── save ───────────────────────────────────────────────────────────────
    def _save(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("No label", "Please enter a label before saving.")
            return

        # Render canvas to PIL image
        img = self._canvas_to_pil()
        dest = self._write_image(img, label)
        self.csv_mgr.append(str(dest), label)
        self._clear()
        self._update_status(f"Saved '{label}' → {dest.name}")

    # def _canvas_to_pil(self) -> "Image.Image":
    #     """Capture canvas content reliably across platforms."""
    #     img = None

    # # Method 1: ImageGrab (Windows / macOS — requires canvas to be visible)
    #     try:
    #         from PIL import ImageGrab
    #         self.root.update_idletasks()  # flush pending draws
    #         x = self.canvas.winfo_rootx() + 2
    #         y = self.canvas.winfo_rooty() + 2
    #         w = x + CANVAS_SIZE - 4
    #         h = y + CANVAS_SIZE - 4
    #         img = ImageGrab.grab(bbox=(x, y, w, h)).convert("L")
    #     except Exception:
    #         img = None

    # # Method 2: Redraw via ImageDraw using canvas item coords
    #     if img is None:
    #         try:
    #             img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)  # white bg
    #             draw = ImageDraw.Draw(img)
    #             for item in self.canvas.find_all():
    #                 coords = self.canvas.coords(item)
    #                 item_type = self.canvas.type(item)
    #                 opts = {k: v for k, v in
    #                         zip(self.canvas.itemconfig(item).keys(),
    #                             [v[-1] for v in self.canvas.itemconfig(item).values()])}
    #                 fill = 0  # black

    #                 if item_type == "oval" and len(coords) == 4:
    #                     draw.ellipse(coords, fill=fill)
    #                 elif item_type == "line" and len(coords) >= 4:
    #                     width = int(opts.get("width", 2))
    #                     for i in range(0, len(coords) - 2, 2):
    #                         draw.line(
    #                             [coords[i], coords[i+1], coords[i+2], coords[i+3]],
    #                             fill=fill, width=width
    #                         )
    #         except Exception:
    #         # Last resort: blank white — at least not inverted black
    #             img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)

    # # Resize and normalise — NO invert needed (drawing is already dark on white)
    #     img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)
    #     img = ImageOps.autocontrast(img)
    #     return img

    def _canvas_to_pil(self) -> "Image.Image":
        # """Redraw canvas items onto a PIL image directly — no screen grab needed."""
        img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)  # white background
        draw = ImageDraw.Draw(img)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            item_type = self.canvas.type(item)
            cfg = self.canvas.itemconfig(item)
            opts = {k: v[-1] for k, v in cfg.items()}
            fill = 0  # black ink

            if item_type == "oval" and len(coords) == 4:
                draw.ellipse(coords, fill=fill)

            elif item_type == "line" and len(coords) >= 4:
                try:
                    width = int(float(opts.get("width", 2)))
                except (ValueError, TypeError):
                    width = 2
            # Collect all points and draw as one polyline
                points = [(coords[i], coords[i + 1])
                          for i in range(0, len(coords), 2)]
                if len(points) >= 2:
                    draw.line(points, fill=fill, width=width)

    # Resize to 28×28
        img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)

    # Only sharpen contrast if actual content exists
        if img.getextrema()[0] < 200:   # some dark pixels present
            img = ImageOps.autocontrast(img)

        return img

    def _write_image(self, img: "Image.Image", label: str) -> Path:
        label_dir = self.out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
        dest = label_dir / filename
        img.save(dest)
        return dest

    # ── folder import (inside GUI) ─────────────────────────────────────────
    def _import_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        folder_path = Path(folder)
        import_images(folder_path, self.out_dir, self.csv_mgr)
        self._update_status(f"Import done. Total: {self.csv_mgr.count()} samples")

    # ── status ─────────────────────────────────────────────────────────────
    def _update_status(self, extra: str = ""):
        total = self.csv_mgr.count()
        counts = self.csv_mgr.label_counts()
        unique = len(counts)
        msg = f"Total samples: {total}  |  Unique labels: {unique}"
        if extra:
            msg += f"  |  {extra}"
        self.status_var.set(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Handwriting Dataset Builder — letters & numbers"
    )
    parser.add_argument("--import-dir", type=Path, default=None,
                        help="Folder of images to import (CLI mode)")
    parser.add_argument("--draw", action="store_true",
                        help="Open drawing GUI")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_images"),
                        help="Where to save images (default: dataset_images/)")
    parser.add_argument("--csv", type=Path, default=Path("dataset.csv"),
                        help="Output CSV file (default: dataset.csv)")
    args = parser.parse_args()

    if not args.import_dir and not args.draw:
        parser.print_help()
        print("\n⚠️  Specify --import-dir, --draw, or both.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_mgr = CSVManager(args.csv)

    print(f"\n📄  CSV  : {args.csv.resolve()}")
    print(f"🖼   Images: {args.out_dir.resolve()}")
    print(f"📊  Existing samples: {csv_mgr.count()}\n")

    # ── CLI image import ───────────────────────────────────────────────────
    if args.import_dir:
        if not PIL_AVAILABLE:
            print("❌  Pillow is required.  Run:  pip install Pillow")
            sys.exit(1)
        import_images(args.import_dir, args.out_dir, csv_mgr)

    # ── GUI drawing mode ───────────────────────────────────────────────────
    if args.draw:
        if not TK_AVAILABLE:
            print("❌  Tkinter is not available on this system.")
            sys.exit(1)
        if not PIL_AVAILABLE:
            print("❌  Pillow is required.  Run:  pip install Pillow")
            sys.exit(1)

        root = tk.Tk()
        app = DrawingApp(root, args.out_dir, csv_mgr)
        root.mainloop()

        # Final summary
        total = csv_mgr.count()
        counts = csv_mgr.label_counts()
        print(f"\n✅  Session complete.  {total} total samples across {len(counts)} labels.")
        if counts:
            print("   Label breakdown:")
            for label, n in sorted(counts.items()):
                print(f"     {label:>4s} : {n}")


if __name__ == "__main__":
    main()