"""
Download the Kaggle Chest X-Ray dataset and train a production-quality model.

Prerequisites:
    pip install kaggle tensorflow scipy
    Place your kaggle.json API credentials at ~/.kaggle/kaggle.json
    (Get from: https://www.kaggle.com/settings -> API -> Create New Token)

Usage:
    python scripts/download_and_train.py
    python scripts/download_and_train.py --epochs 10
    python scripts/download_and_train.py --model ResNet50
    python scripts/download_and_train.py --skip-download --dataset path/to/chest_xray
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_builder import (
    WEIGHTS_DIR,
    build_transfer_model,
    compile_classifier,
    unfreeze_top_layers,
)
from services.model_registry import get_image_size, get_weights_path, normalize_model_name

DEFAULT_DATASET_DIR = BACKEND_DIR / "chest_xray"
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


def build_model(model_name: str = "MobileNetV2") -> "tf.keras.Model":
    import tensorflow as tf

    return build_transfer_model(model_name, trainable_base=False, learning_rate=1e-3)


def train(dataset_dir: Path, epochs: int, model_name: str = "MobileNetV2") -> None:
    import tensorflow as tf

    model_name = normalize_model_name(model_name)
    image_size = get_image_size(model_name)
    model_output = get_weights_path(model_name)

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
        train_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    val_data_dir = val_dir if val_dir.exists() else test_dir
    val_data = eval_gen.flow_from_directory(
        val_data_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=3, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n=== Phase 1: Training {model_name} head (frozen base) for {epochs} epochs ===")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print(f"\n=== Phase 2: Fine-tuning top layers of {model_name} ===")
    unfreeze_top_layers(model, num_layers=30)
    compile_classifier(model, learning_rate=1e-5)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(str(model_output))

    if test_dir.exists():
        test_data = eval_gen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )
        results = model.evaluate(test_data)
        names = model.metrics_names
        for name, val in zip(names, results):
            print(f"Test {name}: {val:.4f}")

    print(f"\n{model_name} saved to {model_output}")


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
    parser.add_argument(
        "--model",
        default="MobileNetV2",
        help="Architecture to train (default: MobileNetV2)",
    )
    args = parser.parse_args()

    if args.skip_download:
        dataset_dir = args.dataset
        if not dataset_dir.exists():
            print(f"ERROR: Dataset not found at {dataset_dir}")
            sys.exit(1)
    else:
        dataset_dir = download_dataset(DEFAULT_DATASET_DIR)

    train(dataset_dir, args.epochs, args.model)

    if normalize_model_name(args.model) == "MobileNetV2":
        print("\nExporting MobileNetV2 to TFLite...")
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
