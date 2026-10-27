"""
config.py
=========
Single source of truth for all settings used across the pipeline.
Import from here instead of hardcoding values in individual scripts.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CHARACTER CLASSES  (62 total: digits + uppercase + lowercase)
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz")
)
NUM_CLASSES = len(CLASSES)                    # 62
CHAR_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CLASSES)}

# ──────────────────────────────────────────────────────────────────────────────
# IMAGE SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE    = 28          # pixels (width = height)
CANVAS_SIZE   = 280         # Tkinter drawing canvas size

# ──────────────────────────────────────────────────────────────────────────────
# DIRECTORY / FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.resolve()

# --- data ---
DATA_DIR             = BASE_DIR / "data"
RAW_DIR              = DATA_DIR / "raw"          # user-drawn originals
PROCESSED_DIR        = DATA_DIR / "processed"    # train/val/test split artefacts
EMNIST_DIR           = DATA_DIR / "emnist"       # downloaded EMNIST files
FEEDBACK_DIR         = DATA_DIR / "feedback"
FEEDBACK_IMAGES_DIR  = FEEDBACK_DIR / "images"
FEEDBACK_CSV         = FEEDBACK_DIR / "feedback.csv"

# Legacy paths (existing data – kept for backwards compatibility)
LEGACY_DATASET_CSV        = BASE_DIR / "dataset.csv"
LEGACY_BALANCED_CSV       = BASE_DIR / "balanced_dataset.csv"
LEGACY_DATASET_IMAGES_DIR = BASE_DIR / "dataset_images"
LEGACY_BALANCED_IMAGES_DIR= BASE_DIR / "balanced_images"
LEGACY_EMNIST_CACHE       = BASE_DIR / ".emnist_cache"

# --- models ---
MODELS_DIR     = BASE_DIR / "models"
MODEL_PATH     = MODELS_DIR / "handwriting_model.keras"
LABELS_PATH    = MODELS_DIR / "labels.json"
PREPROC_CONFIG = MODELS_DIR / "preproc_config.json"   # saved preprocessing params

# ──────────────────────────────────────────────────────────────────────────────
# EMNIST BYCLASS DOWNLOAD URLS
# ──────────────────────────────────────────────────────────────────────────────

# Primary mirror (NIST official zip – very large ~500 MB)
EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# Individual-file mirrors (ByClass split)
EMNIST_BYCLASS_URLS = {
    "train_images": "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images" : "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels" : "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist-byclass-test-labels-idx1-ubyte.gz",
}

# Alternative mirror (often faster)
EMNIST_BYCLASS_ALT_URLS = {
    "train_images": "https://storage.googleapis.com/emnist/emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/emnist/emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images" : "https://storage.googleapis.com/emnist/emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels" : "https://storage.googleapis.com/emnist/emnist-byclass-test-labels-idx1-ubyte.gz",
}

# EMNIST ByClass label → character mapping
# Labels 0-9  → '0'-'9'
# Labels 10-35 → 'A'-'Z'
# Labels 36-61 → 'a'-'z'
EMNIST_LABEL_TO_CHAR = {i: c for i, c in enumerate(CLASSES)}

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT (applied to ORIGINAL samples only, before augmentation)
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# If a class has fewer than MIN_SAMPLES_FOR_SPLIT originals, all go to train.
MIN_SAMPLES_FOR_SPLIT = 3

RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZE      = 64
EPOCHS          = 50           # max; early stopping will cut short
LEARNING_RATE   = 0.001
L2_REG          = 1e-4
DROPOUT_RATE    = 0.5

EARLY_STOPPING_PATIENCE    = 8
REDUCE_LR_PATIENCE         = 4
REDUCE_LR_FACTOR           = 0.5
MIN_LR                     = 1e-6

# How many EMNIST samples per class to use (None = use all)
# Reduce this if RAM is limited. 2000 per class = 124 000 total.
EMNIST_SAMPLES_PER_CLASS   = 2000

# User samples are up-weighted so they are not drowned out by EMNIST.
# Each original user image is duplicated this many times BEFORE augmentation.
USER_UPSAMPLE_FACTOR = 10

# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATION SETTINGS  (applied only to training data)
# ──────────────────────────────────────────────────────────────────────────────

AUG_ROTATION_DEGREES  = 10    # ± degrees
AUG_SHIFT_FRACTION    = 0.10  # ± fraction of image size
AUG_SCALE_RANGE       = (0.85, 1.15)
AUG_BRIGHTNESS_RANGE  = (0.8, 1.2)
AUG_NOISE_STD         = 5.0   # pixel std dev (0-255 scale)
AUG_ELASTIC           = True  # apply mild elastic deformation
AUG_ELASTIC_ALPHA     = 8.0
AUG_ELASTIC_SIGMA     = 3.0

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

# Background is WHITE (255), ink is BLACK (0) in all images.
# This is the polarity used by the drawing canvas and the EMNIST images
# after orientation correction.
BACKGROUND_PIXEL = 255
INK_PIXEL        = 0

# Bounding-box crop margin (pixels around the tightest bounding box)
CROP_MARGIN = 2

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.30   # below this → show "uncertain"
TOP_K_PREDICTIONS    = 5

# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK / CONTINUAL LEARNING
# ──────────────────────────────────────────────────────────────────────────────

RETRAIN_AFTER_N_FEEDBACK_SAMPLES = 20
AUTO_RETRAIN                     = False

# Max feedback samples per class kept in the buffer.
# Older samples are evicted (FIFO) when this is exceeded.
MAX_FEEDBACK_PER_CLASS = 200

# Minimum model improvement (absolute) required to save a new version.
# e.g. 0.001 means new model must score ≥ old_macro_f1 − 0.001.
MIN_IMPROVEMENT_THRESHOLD = -0.005   # small tolerance for acceptable regression

# ──────────────────────────────────────────────────────────────────────────────
# DATASET VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

# Warn when a class has fewer than this many ORIGINAL samples.
MIN_RECOMMENDED_ORIGINALS = 20

# Image is considered "too dark / too light" if mean pixel is outside this range.
VALID_MEAN_RANGE = (10, 245)   # out of 255

# Two images are considered potential duplicates if their pixel-level
# mean absolute difference is below this threshold.
DUPLICATE_DIFF_THRESHOLD = 5.0

# ──────────────────────────────────────────────────────────────────────────────
# MODEL VERSIONING
# ──────────────────────────────────────────────────────────────────────────────

MODEL_METADATA_FILE = MODELS_DIR / "model_versions.json"
CURRENT_MODEL_LINK  = MODEL_PATH   # always points to the best model

# ──────────────────────────────────────────────────────────────────────────────
# VISUALLY SIMILAR CLASS PAIRS  (tracked in confusion analysis)
# ──────────────────────────────────────────────────────────────────────────────

CONFUSABLE_PAIRS = [
    ("0", "O"), ("0", "o"),
    ("1", "I"), ("1", "l"), ("1", "i"),
    ("A", "a"),
    ("B", "b"),
    ("C", "c"),
    ("D", "d"),
    ("F", "f"),
    ("G", "g"),
    ("I", "i"), ("I", "l"),
    ("J", "j"),
    ("K", "k"),
    ("L", "l"),
    ("M", "m"),
    ("N", "n"),
    ("O", "o"),
    ("P", "p"),
    ("Q", "q"),
    ("S", "s"),
    ("T", "t"),
    ("U", "u"),
    ("V", "v"),
    ("W", "w"),
    ("X", "x"),
    ("Y", "y"),
    ("Z", "z"),
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: ensure required directories exist
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all required directories if they do not already exist."""
    dirs = [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, EMNIST_DIR,
        FEEDBACK_DIR, FEEDBACK_IMAGES_DIR, MODELS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Config loaded successfully.")
    print(f"  Classes   : {NUM_CLASSES}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Data dir  : {DATA_DIR}")
