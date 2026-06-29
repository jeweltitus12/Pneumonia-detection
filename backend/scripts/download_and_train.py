"""
Download the Kaggle Chest X-Ray dataset and train a production-quality model.

Prerequisites:
    pip install kaggle tensorflow scipy
    Place your kaggle.json API credentials at ~/.kaggle/kaggle.json
    (Get from: https://www.kaggle.com/settings -> API -> Create New Token)

Usage:
    python scripts/download_and_train.py
    python scripts/download_and_train.py --epochs 10
    python scripts/download_and_train.py --skip-download --dataset path/to/chest_xray
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"
MODEL_OUTPUT = WEIGHTS_DIR / "pneumonia_model.h5"
DEFAULT_DATASET_DIR = BACKEND_DIR / "chest_xray"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def download_dataset(output_dir: Path) -> Path:
    """Download the chest X-ray dataset from Kaggle."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        sys.exit(1)

    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        print("ERROR: Kaggle API credentials not found.")
        print(f"Expected: {kaggle_creds}")
        print("Get your credentials from: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)

    import kaggle  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "chest-xray-pneumonia.zip"

    print("Downloading chest X-ray dataset from Kaggle (~1.2 GB)...")
    kaggle.api.dataset_download_files(
        "paultimothymooney/chest-xray-pneumonia",
        path=str(output_dir),
        unzip=False,
    )

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    zip_path.unlink(missing_ok=True)

    extracted = output_dir / "chest_xray"
    if extracted.exists():
        return extracted

    return output_dir


def build_model() -> "tf.keras.Model":
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Phase 1: freeze base, train only head
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train(dataset_dir: Path, epochs: int) -> None:
    import tensorflow as tf

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Count classes to compute class weights
    n_normal = len(list((train_dir / "NORMAL").glob("*")))
    n_pneumonia = len(list((train_dir / "PNEUMONIA").glob("*")))
    total = n_normal + n_pneumonia
    class_weight = {
        0: total / (2.0 * n_normal),
        1: total / (2.0 * n_pneumonia),
    }
    print(f"Training set: {n_normal} NORMAL, {n_pneumonia} PNEUMONIA")
    print(f"Class weights: NORMAL={class_weight[0]:.2f}, PNEUMONIA={class_weight[1]:.2f}")

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
    )
    eval_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True,
    )
    val_data_dir = val_dir if val_dir.exists() else test_dir
    val_data = eval_gen.flow_from_directory(
        val_data_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=3, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n=== Phase 1: Training head (frozen base) for {epochs} epochs ===")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("\n=== Phase 2: Fine-tuning top layers ===")
    base_model = model.layers[2]  # MobileNetV2 layer
    base_model.trainable = True
    # Unfreeze only the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(str(MODEL_OUTPUT))

    if test_dir.exists():
        test_data = eval_gen.flow_from_directory(
            test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            class_mode="binary", shuffle=False,
        )
        results = model.evaluate(test_data)
        names = model.metrics_names
        for name, val in zip(names, results):
            print(f"Test {name}: {val:.4f}")

    print(f"\nModel saved to {MODEL_OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset and train pneumonia model")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to chest_xray dataset root (used with --skip-download)",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for phase 1")
    args = parser.parse_args()

    if args.skip_download:
        dataset_dir = args.dataset
        if not dataset_dir.exists():
            print(f"ERROR: Dataset not found at {dataset_dir}")
            sys.exit(1)
    else:
        dataset_dir = download_dataset(DEFAULT_DATASET_DIR)

    train(dataset_dir, args.epochs)

    print("\nExporting to TFLite...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "export_tflite.py")],
        check=False,
    )
    if result.returncode != 0:
        print("Warning: TFLite export failed. Run scripts/export_tflite.py manually.")
    else:
        print("Done! Commit backend/weights/pneumonia_model.tflite and push to GitHub.")
        print("The GitHub Action will sync it to Hugging Face automatically.")


if __name__ == "__main__":
    main()
