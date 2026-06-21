"""
Create a starter pneumonia model when no weights file is available.

This script builds a MobileNetV2 classifier and fine-tunes it on any images
found in backend/uploads (virus -> PNEUMONIA, otherwise NORMAL). It is intended
for local development and smoke testing. For production-quality results, train
with scripts/train_model.py on the full Kaggle Chest X-Ray dataset.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import tensorflow as tf

BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BACKEND_DIR / "uploads"
WEIGHTS_DIR = BACKEND_DIR / "weights"
MODEL_OUTPUT = WEIGHTS_DIR / "pneumonia_model.h5"
IMAGE_SIZE = (224, 224)


def _collect_bootstrap_dataset() -> Path | None:
    if not UPLOADS_DIR.exists():
        return None

    images = list(UPLOADS_DIR.glob("*.jpeg")) + list(UPLOADS_DIR.glob("*.jpg")) + list(UPLOADS_DIR.glob("*.png"))
    if not images:
        return None

    temp_root = Path(tempfile.mkdtemp(prefix="pneumonia_bootstrap_"))
    for label in ("NORMAL", "PNEUMONIA"):
        (temp_root / label).mkdir(parents=True, exist_ok=True)

    for image_path in images:
        label = "PNEUMONIA" if "virus" in image_path.name.lower() else "NORMAL"
        shutil.copy2(image_path, temp_root / label / image_path.name)

    return temp_root


def _build_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    dataset_dir = _collect_bootstrap_dataset()
    if dataset_dir is None:
        print("No bootstrap images found in backend/uploads.")
        print("Place sample X-rays there or run scripts/train_model.py with the full dataset.")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=2,
        class_mode="binary",
        shuffle=True,
    )

    model = _build_model()
    model.fit(train_data, epochs=12, verbose=1)
    model.save(str(MODEL_OUTPUT))

    # Export lightweight TFLite model for production (Render/Vercel backends)
    try:
        import subprocess
        subprocess.run([sys.executable, str(BACKEND_DIR / "scripts" / "export_tflite.py")], check=True)
    except Exception as exc:
        print(f"Warning: TFLite export failed ({exc}). Run scripts/export_tflite.py manually.")

    shutil.rmtree(dataset_dir, ignore_errors=True)
    print(f"Bootstrap model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
