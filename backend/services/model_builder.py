"""
Reusable transfer-learning builders for pneumonia binary classification.

MobileNetV2 keeps the original Rescaling(1/255) head so existing
weights/pneumonia_model.h5 and pneumonia_model.tflite stay compatible.
Other architectures use Keras ImageNet preprocess_input inside the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras.applications import densenet, efficientnet, inception_v3, resnet, vgg16

from services.model_registry import (
    AVAILABLE_MODELS,
    MODEL_REGISTRY,
    WEIGHTS_DIR,
    get_image_size,
    get_registry_entry,
    get_weights_path,
    normalize_model_name,
)

__all__ = [
    "AVAILABLE_MODELS",
    "ImageNetPreprocess",
    "KERAS_CUSTOM_OBJECTS",
    "WEIGHTS_DIR",
    "build_transfer_model",
    "compile_classifier",
    "get_backbone",
    "get_image_size",
    "get_model_spec",
    "get_weights_path",
    "normalize_model_name",
    "unfreeze_top_layers",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    constructor: Callable[..., tf.keras.Model]
    image_size: tuple[int, int]
    weights_filename: str
    preprocess: str


_CONSTRUCTORS = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "DenseNet121": tf.keras.applications.DenseNet121,
    "ResNet50": tf.keras.applications.ResNet50,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "VGG16": tf.keras.applications.VGG16,
    "InceptionV3": tf.keras.applications.InceptionV3,
}

_PREPROCESS_FNS = {
    "densenet": densenet.preprocess_input,
    "resnet": resnet.preprocess_input,
    "efficientnet": efficientnet.preprocess_input,
    "vgg16": vgg16.preprocess_input,
    "inception_v3": inception_v3.preprocess_input,
}


class ImageNetPreprocess(tf.keras.layers.Layer):
    """Architecture-specific ImageNet preprocessing, serialized with the model."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        return _PREPROCESS_FNS[self.mode](x)

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


KERAS_CUSTOM_OBJECTS = {"ImageNetPreprocess": ImageNetPreprocess}


def get_model_spec(model_name: str) -> ModelSpec:
    name = normalize_model_name(model_name)
    entry = MODEL_REGISTRY[name]
    return ModelSpec(
        name=name,
        constructor=_CONSTRUCTORS[name],
        image_size=tuple(entry["image_size"]),
        weights_filename=entry["weights_filename"],
        preprocess=entry["preprocess"],
    )


def _apply_preprocess(inputs: tf.Tensor, spec: ModelSpec) -> tf.Tensor:
    if spec.preprocess == "rescale":
        return tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    return ImageNetPreprocess(spec.preprocess, name="preprocess")(inputs)


def _build_backbone(spec: ModelSpec, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    kwargs = {
        "input_shape": input_shape,
        "include_top": False,
        "weights": "imagenet",
    }
    try:
        return spec.constructor(**kwargs, include_preprocessing=False)
    except TypeError:
        return spec.constructor(**kwargs)


def build_transfer_model(
    model_name: str,
    *,
    trainable_base: bool = False,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build a binary Normal/Pneumonia classifier with ImageNet weights."""
    spec = get_model_spec(model_name)
    input_shape = (*spec.image_size, 3)

    base_model = _build_backbone(spec, input_shape)
    base_model.trainable = trainable_base
    base_model._name = "backbone"

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _apply_preprocess(inputs, spec)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{spec.name}_pneumonia")
    compile_classifier(model, learning_rate=learning_rate)
    return model


def compile_classifier(model: tf.keras.Model, learning_rate: float = 1e-4) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    return model.get_layer("backbone")


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> None:
    backbone = get_backbone(model)
    backbone.trainable = True
    freeze_until = max(len(backbone.layers) - num_layers, 0)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
