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
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_builder import WEIGHTS_DIR, build_transfer_model, compile_classifier
from services.model_registry import get_weights_path

UPLOADS_DIR = BACKEND_DIR / "uploads"
MODEL_OUTPUT = get_weights_path("MobileNetV2")


def _collect_bootstrap_dataset() -> tuple[Path, int, int] | None:
    if not UPLOADS_DIR.exists():
        return None

    images = (
        list(UPLOADS_DIR.glob("*.jpeg"))
        + list(UPLOADS_DIR.glob("*.jpg"))
        + list(UPLOADS_DIR.glob("*.png"))
    )
    if not images:
        return None

    temp_root = Path(tempfile.mkdtemp(prefix="pneumonia_bootstrap_"))
    for label in ("NORMAL", "PNEUMONIA"):
        (temp_root / label).mkdir(parents=True, exist_ok=True)

    n_normal = 0
    n_pneumonia = 0
    for image_path in images:
        # Skip UUID-prefixed uploads (user-uploaded at runtime, not training data)
        if len(image_path.stem.split("_")[0]) == 32:
            continue
        label = "PNEUMONIA" if "virus" in image_path.name.lower() else "NORMAL"
        shutil.copy2(image_path, temp_root / label / image_path.name)
        if label == "NORMAL":
            n_normal += 1
        else:
            n_pneumonia += 1

    print(f"Bootstrap dataset: {n_normal} NORMAL, {n_pneumonia} PNEUMONIA")

    if n_normal == 0 or n_pneumonia == 0:
        print("ERROR: Need at least one image per class. Check uploads/ folder.")
        shutil.rmtree(temp_root, ignore_errors=True)
        sys.exit(1)

    return temp_root, n_normal, n_pneumonia


def _build_model() -> tf.keras.Model:
    model = build_transfer_model("MobileNetV2", trainable_base=True, learning_rate=1e-5)
    compile_classifier(model, learning_rate=1e-5)
    return model


def main() -> None:
    result = _collect_bootstrap_dataset()
    if result is None:
        print("No bootstrap images found in backend/uploads.")
        print("Place sample X-rays there or run scripts/train_model.py with the full dataset.")
        sys.exit(1)

    dataset_dir, n_normal, n_pneumonia = result

    # Compute class weights to counter imbalance (NORMAL gets higher weight if fewer)
    total = n_normal + n_pneumonia
    class_weight = {
        0: total / (2.0 * n_normal),      # index 0 = NORMAL (alphabetical)
        1: total / (2.0 * n_pneumonia),   # index 1 = PNEUMONIA
    }
    print(f"Class weights: NORMAL={class_weight[0]:.2f}, PNEUMONIA={class_weight[1]:.2f}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=2,
        class_mode="binary",
        shuffle=True,
    )

    model = _build_model()
    model.fit(
        train_data,
        epochs=15,
        class_weight=class_weight,
        verbose=1,
    )
    model.save(str(MODEL_OUTPUT))

    # Export lightweight TFLite model for production
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(BACKEND_DIR / "scripts" / "export_tflite.py")],
            check=True,
        )
    except Exception as exc:
        print(f"Warning: TFLite export failed ({exc}). Run scripts/export_tflite.py manually.")

    shutil.rmtree(dataset_dir, ignore_errors=True)
    print(f"Bootstrap model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
