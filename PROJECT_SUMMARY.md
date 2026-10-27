# Handwritten Character Recognition — Project Summary

## Project Structure

```
Handwritten Recognization/
│
├── src_kiro/                        ← All pipeline scripts (active)
│   ├── config_kiro.py               Central config: classes, paths, hyperparams
│   ├── preprocessing_kiro.py        Shared preprocessing (polarity fix, crop, resize)
│   ├── augment_kiro.py              Realistic augmentation (rotation, shift, elastic)
│   ├── dataset_builder_kiro.py      GUI tool to collect handwriting samples
│   ├── dataset_validator_kiro.py    Dataset quality checker
│   ├── train_kiro.py                Main training script (EMNIST + user data)
│   ├── evaluate_kiro.py             Per-class metrics, confusion matrix
│   ├── predict_kiro.py              Live drawing UI with feedback buttons
│   ├── feedback_manager_kiro.py     Manages user correction feedback dataset
│   └── retrain_kiro.py              Controlled retraining with model comparison
│
├── data/
│   ├── raw/                         Original user handwriting (212 images, 62 classes)
│   ├── emnist/                      EMNIST ByClass cache (downloaded on first train)
│   ├── feedback/                    User correction samples (images + feedback.csv)
│   ├── dataset.csv                  Original samples manifest
│   └── balanced_dataset.csv         Legacy balanced manifest (kept for reference)
│
├── models/
│   ├── handwriting_model.keras      Active model (created after training)
│   ├── labels.json                  Class list (62 characters)
│   ├── preproc_config.json          Preprocessing settings snapshot
│   ├── training_history.png         Accuracy/loss curves
│   ├── confusion_matrix.png         62×62 confusion matrix
│   └── test_classification_report.txt  Per-class F1/precision/recall
│
├── _obsolete/                       Old broken scripts (kept for reference)
├── requirements.txt                 Python dependencies
└── PROJECT_SUMMARY.md               This file
```

---

## Classes

62 total: `0–9` (10) + `A–Z` (26) + `a–z` (26)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Commands

### 1. Collect more handwriting data (recommended: 20+ rounds)
```bash
python src_kiro/dataset_builder_kiro.py --batch --rounds 20
```

### 2. Validate the dataset
```bash
python src_kiro/dataset_validator_kiro.py --csv data/dataset.csv
```

### 3. Train the model
```bash
python src_kiro/train_kiro.py
```
- Downloads EMNIST ByClass (~500 MB) on first run (stored in `data/emnist/`)
- Splits user originals into train/val/test **before** augmentation (no leakage)
- Combines EMNIST + user data, trains with class weights + early stopping
- Saves model to `models/handwriting_model.keras`

Skip EMNIST (user data only):
```bash
python src_kiro/train_kiro.py --no-emnist
```

### 4. Evaluate the model
```bash
python src_kiro/evaluate_kiro.py
```

### 5. Predict — live drawing window
```bash
python src_kiro/predict_kiro.py --draw
```

Predict a single image file:
```bash
python src_kiro/predict_kiro.py --image path/to/image.png
```

### 6. View feedback statistics
```bash
python src_kiro/feedback_manager_kiro.py stats
```

### 7. Retrain with accumulated feedback
```bash
python src_kiro/retrain_kiro.py
```

---

## Why the old model always predicted "A" — Root Causes

| # | Problem | Fix |
|---|---------|-----|
| 1 | **Train/val leakage** — augmented copies of the same image split across train and val | Split ORIGINALS first, augment only the training portion |
| 2 | **Tiny real data** — only 2–5 genuine samples per class; augmenting 1 image to 150 copies teaches nothing new | Use EMNIST ByClass (814k samples) as the primary data source |
| 3 | **Wrong EMNIST** — old scripts loaded MNIST digits cache (10 classes) not EMNIST ByClass (62 classes) | Download and verify EMNIST ByClass with correct label mapping |
| 4 | **Polarity mismatch** — one predict script captured inverted images vs training | Single shared `preprocess_image()` used everywhere |
| 5 | **No bounding-box crop** — `img.resize(28,28)` on an off-centre character | Crop to ink bounding box, pad to square, then resize |
| 6 | **Bad augmentation** — `np.roll` wraps pixels around edges; rotation filled with black | Constant white fill; proper scipy/PIL transforms |
| 7 | **No test set** — only 85/15 train/val split; no held-out test | Proper 70/15/15 stratified split of originals |

---

## Dataset Limitations

The current user dataset has only **2–5 original handwriting samples per class**.
This is critically low. The model will rely primarily on EMNIST for generalisation.

**Recommendation:** Run at least 20 rounds of data collection:
```bash
python src_kiro/dataset_builder_kiro.py --batch --rounds 20
```
Target: **20–50 genuine originals per class** for reliable live-drawing performance.

---

## Continual Learning (Feedback Loop)

```
User draws → Model predicts → User confirms YES/NO
                                        ↓
                              NO → User enters correct label
                                        ↓
                              Saved to data/feedback/
                                        ↓
                    After 20 corrections → retrain prompt
                                        ↓
                    python src_kiro/retrain_kiro.py
                                        ↓
                    New model compared to old model
                                        ↓
                    Better → save  |  Worse → keep old
```

Auto-retraining is **disabled by default** (`AUTO_RETRAIN = False` in `config_kiro.py`).
Change `RETRAIN_AFTER_N_FEEDBACK_SAMPLES` to adjust the threshold.
