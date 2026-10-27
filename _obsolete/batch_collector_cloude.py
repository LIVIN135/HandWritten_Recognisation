"""
Batch Data Collector - guided, balanced data collection
==========================================================
Your biggest accuracy lever right now isn't the model or the training
script - it's that you only have 1-3 real drawings per character.
Augmentation (rotate/shift/noise) can only reshuffle what already
exists; it can't invent new handwriting styles, slants, or stroke
widths. More REAL drawings is the highest-leverage thing you can do.

This tool walks you through drawing every character (0-9, A-Z, a-z)
in ROUNDS. Each round asks for one drawing of every class, in random
order, then starts the next round. That keeps the dataset perfectly
balanced by construction (every class always has the same count) and
avoids the fatigue/boredom of drawing the same letter 20 times in a
row.

HOW TO RUN:
-----------
   python batch_collect.py                  # 20 rounds (recommended minimum)
   python batch_collect.py --rounds 30      # more rounds = more accuracy
   python batch_collect.py --rounds 10 --vary-brush

TIPS FOR BETTER DATA (read before you start):
-----------------------------------------------
- Vary your writing style a bit each round: change the slant, make it
  slightly bigger/smaller, use print vs. cursive-ish strokes for
  letters like 'a' 'g' 'q' where people write two different ways.
- If possible, get 1-2 OTHER people to run this tool too and append to
  the same dataset.csv (different handwriting = the single best thing
  you can add). Just run this script on their machine pointed at the
  same dataset.csv/dataset_images folder (e.g. share via USB/cloud
  drive), or literally hand them the keyboard.
- Don't rush - a careless scribble teaches the model less than a
  clean, deliberate stroke.
"""

import argparse
import csv
import random
import uuid
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("Tkinter is not available on this system. This tool needs a "
          "desktop Python install with Tkinter (standard on Windows/macOS).")
    raise SystemExit(1)

from PIL import Image, ImageDraw, ImageOps

CANVAS_SIZE = 280
EXPORT_SIZE = 28
OUT_DIR = Path("dataset_images")
CSV_PATH = Path("dataset.csv")

LABELS = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)


def ensure_csv_header():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["image_path", "label"])


def build_queue(rounds: int):
    """One shuffled pass through every label, repeated `rounds` times."""
    queue = []
    for _ in range(rounds):
        round_labels = LABELS.copy()
        random.shuffle(round_labels)
        queue.extend(round_labels)
    return queue


class BatchCollector:
    def __init__(self, root, rounds, vary_brush):
        self.root = root
        self.queue = build_queue(rounds)
        self.total = len(self.queue)
        self.index = 0
        self.vary_brush = vary_brush
        self.last = {"x": None, "y": None}
        self.saved_count = {}

        root.title("Batch Data Collector")
        root.resizable(False, False)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 13), fg="#555").pack(pady=(10, 2))

        self.prompt_var = tk.StringVar()
        tk.Label(root, textvariable=self.prompt_var,
                 font=("Helvetica", 40, "bold")).pack(pady=(0, 8))

        brush_default = random.randint(10, 18) if vary_brush else 14
        self.brush_size = brush_default

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg="white", cursor="crosshair",
                                 highlightthickness=2, highlightbackground="#888")
        self.canvas.pack(padx=10, pady=4)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Clear  [C]", command=self.clear,
                  bg="#f0f0f0", font=("Helvetica", 11), padx=10).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Save & Next  [S]", command=self.save_and_next,
                  bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
                  padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Skip this one  [K]", command=self.skip,
                  bg="#ff9800", fg="white", font=("Helvetica", 11), padx=10).pack(side="left", padx=6)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_brush)
        root.bind("<s>", lambda e: self.save_and_next())
        root.bind("<S>", lambda e: self.save_and_next())
        root.bind("<c>", lambda e: self.clear())
        root.bind("<C>", lambda e: self.clear())
        root.bind("<k>", lambda e: self.skip())
        root.bind("<K>", lambda e: self.skip())

        self.show_current()

    def current_label(self):
        return self.queue[self.index]

    def show_current(self):
        if self.index >= self.total:
            messagebox.showinfo("Done!", f"Collected drawings for all "
                                 f"{self.total} prompts. Great work!\n\n"
                                 f"Next step: python balanced_augment.py")
            self.root.destroy()
            return
        label = self.current_label()
        self.status_var.set(f"Prompt {self.index + 1} / {self.total}")
        self.prompt_var.set(f"Draw:  {label}")
        if self.vary_brush:
            self.brush_size = random.randint(10, 18)

    def paint(self, event):
        r = self.brush_size // 2
        x, y = event.x, event.y
        if self.last["x"] is not None:
            self.canvas.create_line(self.last["x"], self.last["y"], x, y,
                                     fill="black", width=self.brush_size,
                                     capstyle=tk.ROUND, smooth=True)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                 fill="black", outline="black")
        self.last["x"], self.last["y"] = x, y

    def reset_brush(self, event):
        self.last["x"] = None
        self.last["y"] = None

    def clear(self):
        self.canvas.delete("all")
        self.last["x"] = self.last["y"] = None

    def canvas_to_pil(self):
        """Same capture method as handwriting_dataset_builder.py: redraw
        canvas strokes onto a PIL image directly (black ink, white bg)."""
        img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            item_type = self.canvas.type(item)
            cfg = self.canvas.itemconfig(item)
            opts = {k: v[-1] for k, v in cfg.items()}
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
        img = img.resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)
        if img.getextrema()[0] < 200:
            img = ImageOps.autocontrast(img)
        return img

    def save_and_next(self):
        if not self.canvas.find_all():
            messagebox.showwarning("Empty", "Draw something before saving!")
            return
        label = self.current_label()
        img = self.canvas_to_pil()
        label_dir = OUT_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
        dest = label_dir / filename
        img.save(dest)
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([str(dest), label])
        self.saved_count[label] = self.saved_count.get(label, 0) + 1
        self.index += 1
        self.clear()
        self.show_current()

    def skip(self):
        self.index += 1
        self.clear()
        self.show_current()


def main():
    parser = argparse.ArgumentParser(description="Guided balanced data collector")
    parser.add_argument("--rounds", type=int, default=20,
                         help="How many drawings to collect per class (default: 20)")
    parser.add_argument("--vary-brush", action="store_true",
                         help="Randomize brush thickness each prompt for more stroke-width variety")
    args = parser.parse_args()

    ensure_csv_header()

    print(f"\nCollecting {args.rounds} drawing(s) for each of {len(LABELS)} "
          f"characters = {args.rounds * len(LABELS)} new images total.")
    print("Draw naturally, vary your style a little each round, and take breaks!\n")

    root = tk.Tk()
    BatchCollector(root, args.rounds, args.vary_brush)
    root.mainloop()


if __name__ == "__main__":
    main()