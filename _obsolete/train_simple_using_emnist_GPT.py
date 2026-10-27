import gzip
import struct
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# CONFIG
# -----------------------------

CACHE_DIR = Path(".emnist_cache")
MODEL_DIR = Path("my_model")

MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 15

# -----------------------------
# LOAD IDX FILES
# -----------------------------

def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)

    data = data.reshape(count, rows, cols)

    # normalize
    data = data.astype(np.float32) / 255.0

    # channel dimension
    data = np.expand_dims(data, -1)

    return data


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


# -----------------------------
# LOAD DATASET
# -----------------------------

print("Loading EMNIST...")

X_train = read_images(
    CACHE_DIR / "train-images-idx3-ubyte.gz"
)

y_train = read_labels(
    CACHE_DIR / "train-labels-idx1-ubyte.gz"
)

X_test = read_images(
    CACHE_DIR / "t10k-images-idx3-ubyte.gz"
)

y_test = read_labels(
    CACHE_DIR / "t10k-labels-idx1-ubyte.gz"
)

num_classes = len(np.unique(y_train))

print(f"Train images : {len(X_train)}")
print(f"Test images  : {len(X_test)}")
print(f"Classes      : {num_classes}")

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# -----------------------------
# MODEL
# -----------------------------

inputs = keras.Input(shape=(28, 28, 1))

x = augmentation(inputs)

x = layers.Conv2D(
    32, 3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(
    64, 3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(
    128, 3,
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# -----------------------------
# COMPILE
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# -----------------------------
# EVALUATE
# -----------------------------

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {acc:.4f}")

# -----------------------------
# SAVE LABELS
# -----------------------------

labels = {
    str(i): str(i)
    for i in range(num_classes)
}

with open(
    MODEL_DIR / "labels.json",
    "w"
) as f:
    json.dump(labels, f, indent=2)

print("\nModel saved:")
print("  my_model/model.keras")
print("  my_model/labels.json")