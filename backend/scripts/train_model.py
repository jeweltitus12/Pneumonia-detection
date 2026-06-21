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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

WEIGHTS_DIR = BACKEND_DIR / "weights"
MODEL_OUTPUT = WEIGHTS_DIR / "pneumonia_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8


def build_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_generators(dataset_dir: Path):
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
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
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
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )

    return train_data, val_data, test_data


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
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using dataset: {args.dataset}")
    train_data, val_data, test_data = create_generators(args.dataset)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if test_data is not None:
        loss, accuracy = model.evaluate(test_data)
        print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    model.save(str(MODEL_OUTPUT))
    print(f"Model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
