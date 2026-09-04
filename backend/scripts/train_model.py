"""
Train a pneumonia detection CNN on the Kaggle Chest X-Ray dataset.

Expected dataset layout:
    chest_xray/
      train/
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/

Usage:
    python scripts/train_model.py --dataset path/to/chest_xray
    python scripts/train_model.py --dataset path/to/chest_xray --model DenseNet121
    python scripts/train_model.py --dataset path/to/chest_xray --model all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_builder import AVAILABLE_MODELS, WEIGHTS_DIR, build_transfer_model, normalize_model_name
from services.model_registry import get_image_size, get_weights_path

BATCH_SIZE = 32
EPOCHS = 8


def build_model(model_name: str = "MobileNetV2") -> tf.keras.Model:
    return build_transfer_model(model_name, trainable_base=False, learning_rate=1e-4)


def create_generators(dataset_dir: Path, image_size: tuple[int, int] = (224, 224)):
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Download the Kaggle Chest X-Ray dataset and pass --dataset."
        )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_data = None
    if test_dir.exists():
        test_data = test_gen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )

    return train_data, val_data, test_data


def _class_weights(dataset_dir: Path) -> dict[int, float] | None:
    train_dir = dataset_dir / "train"
    n_normal = len(list((train_dir / "NORMAL").glob("*"))) if (train_dir / "NORMAL").exists() else 0
    n_pneumonia = (
        len(list((train_dir / "PNEUMONIA").glob("*"))) if (train_dir / "PNEUMONIA").exists() else 0
    )
    if n_normal == 0 or n_pneumonia == 0:
        return None
    total = n_normal + n_pneumonia
    class_weight = {0: total / (2.0 * n_normal), 1: total / (2.0 * n_pneumonia)}
    print(f"Training set: {n_normal} NORMAL, {n_pneumonia} PNEUMONIA")
    print(f"Class weights: NORMAL={class_weight[0]:.2f}, PNEUMONIA={class_weight[1]:.2f}")
    return class_weight


def train_one(model_name: str, dataset_dir: Path, epochs: int, class_weight) -> None:
    image_size = get_image_size(model_name)
    output_path = get_weights_path(model_name)
    print(f"\n=== Training {model_name} ({image_size[0]}x{image_size[1]}) ===")
    print(f"Weights will be saved to {output_path}")

    train_data, val_data, test_data = create_generators(dataset_dir, image_size=image_size)
    model = build_model(model_name)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    if test_data is not None:
        results = model.evaluate(test_data)
        for name, val in zip(model.metrics_names, results):
            print(f"{model_name} test {name}: {val:.4f}")

    model.save(str(output_path))
    print(f"{model_name} saved to {output_path}")


FIVE_MODELS = (
    "DenseNet121",
    "ResNet50",
    "EfficientNetB0",
    "VGG16",
    "InceptionV3",
)


def _resolve_models(model_arg: str) -> list[str]:
    key = model_arg.strip().lower()
    if key == "all":
        return list(AVAILABLE_MODELS)
    if key in {"five", "5", "extra"}:
        return list(FIVE_MODELS)
    return [normalize_model_name(model_arg)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to chest_xray dataset root",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--model",
        default="MobileNetV2",
        help=(
            "Architecture to train: "
            + ", ".join(AVAILABLE_MODELS)
            + ", 'five' (all except MobileNetV2), or 'all'. Default: MobileNetV2."
        ),
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    class_weight = _class_weights(args.dataset)
    print(f"Using dataset: {args.dataset}")

    for model_name in _resolve_models(args.model):
        train_one(model_name, args.dataset, args.epochs, class_weight)


if __name__ == "__main__":
    main()
