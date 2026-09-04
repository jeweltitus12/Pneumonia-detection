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
        "preprocess": "rescale",
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
        "grad_cam_layer": "conv5_block3_3_conv3",
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
