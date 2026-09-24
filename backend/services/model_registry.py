"""Model names, input sizes, and weight filenames (no TensorFlow import)."""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"

# Existing MobileNetV2 files — other architectures must not overwrite these.
MOBILENET_H5 = "pneumonia_model.h5"
MOBILENET_TFLITE = "pneumonia_model.tflite"

MODEL_REGISTRY: dict[str, dict] = {
    "MobileNetV2": {
        "image_size": (224, 224),
        "weights_filename": MOBILENET_H5,
        "tflite_filename": MOBILENET_TFLITE,
        "preprocess": "mobilenet_v2",
        "grad_cam_layer": "block_16_project",
    },
    "DenseNet121": {
        "image_size": (224, 224),
        "weights_filename": "densenet121.h5",
        "tflite_filename": None,
        "preprocess": "densenet",
        "grad_cam_layer": "conv5_block32_2_conv",
    },
    "ResNet50": {
        "image_size": (224, 224),
        "weights_filename": "resnet50.h5",
        "tflite_filename": None,
        "preprocess": "resnet",
        "grad_cam_layer": "conv5_block3_3_conv",
    },
    "EfficientNetB0": {
        "image_size": (224, 224),
        "weights_filename": "efficientnetb0.h5",
        "tflite_filename": None,
        "preprocess": "efficientnet",
        "grad_cam_layer": "top_conv",
    },
    "VGG16": {
        "image_size": (224, 224),
        "weights_filename": "vgg16.h5",
        "tflite_filename": None,
        "preprocess": "vgg16",
        "grad_cam_layer": "block5_conv3",
    },
    "InceptionV3": {
        "image_size": (299, 299),
        "weights_filename": "inceptionv3.h5",
        "tflite_filename": None,
        "preprocess": "inception_v3",
        "grad_cam_layer": "mixed10",
    },
}

AVAILABLE_MODELS = tuple(MODEL_REGISTRY.keys())


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        raise ValueError("model_name is required")

    if model_name in MODEL_REGISTRY:
        return model_name

    lookup = {name.lower(): name for name in MODEL_REGISTRY}
    if model_name.lower() in lookup:
        return lookup[model_name.lower()]

    compact = {
        name.lower().replace("_", "").replace("-", "").replace(" ", ""): name
        for name in MODEL_REGISTRY
    }
    key = model_name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in compact:
        return compact[key]

    raise ValueError(
        f"Unknown model '{model_name}'. Supported: {', '.join(AVAILABLE_MODELS)}"
    )


def get_registry_entry(model_name: str) -> dict:
    return MODEL_REGISTRY[normalize_model_name(model_name)]


def get_weights_path(model_name: str) -> Path:
    entry = get_registry_entry(model_name)
    return WEIGHTS_DIR / entry["weights_filename"]


def get_image_size(model_name: str) -> tuple[int, int]:
    return tuple(get_registry_entry(model_name)["image_size"])


def get_grad_cam_layer(model_name: str) -> str:
    return get_registry_entry(model_name)["grad_cam_layer"]


def get_decision_threshold(model_name: str | None = None) -> float:
    """Return the validation-chosen threshold for a model, else the env default.

    Thresholds are written by scripts/train_improved.py after a validation-only
    search. They are not tuned on the test set.
    """
    import json
    import os

    default = float(os.environ.get("PNEUMONIA_THRESHOLD", "0.5"))
    if not model_name:
        return default

    path = WEIGHTS_DIR / "thresholds.json"
    if not path.exists():
        return default

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = normalize_model_name(model_name)
    except (OSError, ValueError, json.JSONDecodeError):
        return default

    entry = data.get(name)
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict) and isinstance(entry.get("threshold"), (int, float)):
        return float(entry["threshold"])
    return default
