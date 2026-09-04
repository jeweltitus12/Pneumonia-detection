"""
Grad-CAM explainability for pneumonia detection models.

Each architecture uses its own final convolutional layer inside the `backbone`
submodel. Grad-CAM requires a Keras .h5 model and TensorFlow — TFLite weights
cannot be used for gradient-based visualization.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from services.model_registry import get_grad_cam_layer, get_image_size, normalize_model_name


class GradCamError(Exception):
    """Raised when Grad-CAM cannot be produced for the requested model."""


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise GradCamError(
            "TensorFlow is required for Grad-CAM explainability. "
            "Install tensorflow locally or use requirements-dev.txt."
        ) from exc
    return tf


def resolve_grad_cam_layer(model, model_name: str):
    """Resolve the architecture-specific final conv layer inside `backbone`."""
    name = normalize_model_name(model_name)
    configured_layer = get_grad_cam_layer(name)

    try:
        backbone = model.get_layer("backbone")
    except ValueError as exc:
        raise GradCamError(
            f"Model {name} is missing the expected 'backbone' layer for Grad-CAM."
        ) from exc

    try:
        return backbone.get_layer(configured_layer)
    except ValueError as exc:
        raise GradCamError(
            f"Grad-CAM layer '{configured_layer}' was not found in {name} backbone. "
            "The saved model may be incompatible with this architecture."
        ) from exc


def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Map normalized grayscale heatmap values to an RGB jet-like colormap."""
    gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _array_to_data_url(image_array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _make_gradcam_heatmap(img_batch: np.ndarray, model, conv_layer) -> np.ndarray:
    tf = _require_tensorflow()

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise GradCamError(
            "Unable to compute gradients for Grad-CAM. "
            f"The selected layer '{conv_layer.name}' may not be connected to the model output."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) <= 0:
        raise GradCamError(
            "Grad-CAM produced an empty heatmap. The model may not have usable activations "
            f"for the selected layer '{conv_layer.name}'."
        )

    heatmap = heatmap / max_value
    return heatmap.numpy()


def _render_visualizations(
    image_path: str,
    heatmap: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, str]:
    original = Image.open(image_path).convert("RGB")
    display_original = original.resize(image_size, Image.Resampling.LANCZOS)
    original_array = np.asarray(display_original, dtype=np.uint8)

    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
        image_size,
        Image.Resampling.BILINEAR,
    )
    heatmap_gray = np.asarray(heatmap_resized, dtype=np.float32) / 255.0
    heatmap_rgb = np.uint8(255 * _jet_colormap(heatmap_gray))

    overlay = np.uint8(0.55 * original_array + 0.45 * heatmap_rgb)

    return {
        "heatmap": _array_to_data_url(heatmap_rgb),
        "overlay": _array_to_data_url(overlay),
        "original": _array_to_data_url(original_array),
    }


def generate_grad_cam(
    image_path: str,
    model,
    model_name: str,
    img_batch: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Build a Grad-CAM heatmap and overlay for the given Keras model and image.

    Returns a dict with base64 PNG data URLs and metadata about the conv layer used.
    """
    name = normalize_model_name(model_name)
    image_size = get_image_size(name)
    conv_layer = resolve_grad_cam_layer(model, name)

    if img_batch is None:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img_batch = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)

    heatmap = _make_gradcam_heatmap(img_batch, model, conv_layer)
    visuals = _render_visualizations(image_path, heatmap, image_size)

    return {
        **visuals,
        "layer": conv_layer.name,
        "backbone_layer": f"backbone/{conv_layer.name}",
    }
