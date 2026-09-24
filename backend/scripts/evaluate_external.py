"""
External validation of trained pneumonia models (no retraining).

Evaluates existing Kaggle-trained weights on an external folder dataset such as
the VIT Bhopal / Bhopal Mendeley chest X-ray set (NORMAL/ + PNEUMONIA/).

Usage:
    python scripts/evaluate_external.py --dataset datasets/bhopal_chest_xray
    python scripts/evaluate_external.py --dataset datasets/bhopal_chest_xray --model all
    python scripts/evaluate_external.py --dataset datasets/bhopal_chest_xray --model MobileNetV2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load backend/.env so PNEUMONIA_THRESHOLD matches the running app when set.
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from services.model_registry import (  # noqa: E402
    AVAILABLE_MODELS,
    get_image_size,
    get_weights_path,
    normalize_model_name,
)
from services.ai_model import (  # noqa: E402
    IMAGE_SIZE,
    _is_tflite_handle,
    _score_from_raw,
    _tflite_is_usable,
    get_model,
    is_model_available,
    load_model,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _threshold() -> float:
    # Prefer app .env (often 0.65); fall back to 0.65 to match .env.example / plan.
    return float(os.environ.get("PNEUMONIA_THRESHOLD", "0.65"))


def _collect_images(dataset_dir: Path) -> list[tuple[Path, int]]:
    """Return (path, label) with label 0=NORMAL, 1=PNEUMONIA."""
    samples: list[tuple[Path, int]] = []
    for label_name, label_id in (("NORMAL", 0), ("PNEUMONIA", 1)):
        folder = dataset_dir / label_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected class folder missing: {folder}\n"
                "Layout must be dataset/NORMAL/ and dataset/PNEUMONIA/."
            )
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label_id))
    if not samples:
        raise FileNotFoundError(f"No images found under {dataset_dir}")
    return samples


def _preprocess(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def _predict_score_tflite(interpreter, image_path: Path) -> float:
    batch = _preprocess(image_path, IMAGE_SIZE)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details[0]["index"])
    return _score_from_raw(raw)


def _predict_scores_keras(model, paths: list[Path], image_size: tuple[int, int]) -> list[float]:
    """Batched Keras inference for faster external validation."""
    scores: list[float] = []
    chunk = 32
    for start in range(0, len(paths), chunk):
        part = paths[start : start + chunk]
        batch = np.concatenate([_preprocess(p, image_size) for p in part], axis=0)
        raw = model.predict(batch, verbose=0)
        raw = np.asarray(raw)
        if raw.ndim == 2 and raw.shape[1] > 1:
            part_scores = raw[:, 1]
        else:
            part_scores = np.squeeze(raw, axis=-1) if raw.ndim > 1 else raw
        scores.extend(float(np.clip(s, 0.0, 1.0)) for s in np.asarray(part_scores).tolist())
        print(f"  {min(start + chunk, len(paths))}/{len(paths)} ...", flush=True)
    return scores


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        # Fallback AUC via Mann–Whitney if sklearn is unavailable.
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) and len(neg):
            ranks = 0.0
            for p in pos:
                ranks += float(np.sum(neg < p) + 0.5 * np.sum(neg == p))
            auc = float(ranks / (len(pos) * len(neg)))

    return {
        "n_images": int(len(y_true)),
        "n_normal": int(np.sum(y_true == 0)),
        "n_pneumonia": int(np.sum(y_true == 1)),
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": None if auc is None else round(auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def evaluate_model(model_name: str, samples: list[tuple[Path, int]], threshold: float) -> dict:
    if not is_model_available(model_name):
        return {
            "model": model_name,
            "available": False,
            "weights_path": str(get_weights_path(model_name)),
            "error": "Weights not found",
        }

    print(f"\n=== Evaluating {model_name} ({len(samples)} images) ===", flush=True)
    model = get_model(model_name)
    use_tflite = model_name == "MobileNetV2" and (
        _is_tflite_handle(model) or _tflite_is_usable()
    )
    weights_path = (
        str(BACKEND_DIR / "weights" / "pneumonia_model.tflite")
        if use_tflite
        else str(get_weights_path(model_name))
    )

    y_true = [label for _, label in samples]
    paths = [path for path, _ in samples]

    if use_tflite:
        interpreter = model if _is_tflite_handle(model) else load_model()
        y_score = []
        for i, path in enumerate(paths, start=1):
            y_score.append(_predict_score_tflite(interpreter, path))
            if i % 50 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} ...", flush=True)
    else:
        y_score = _predict_scores_keras(model, paths, get_image_size(model_name))

    result = _metrics(np.asarray(y_true), np.asarray(y_score, dtype=np.float64), threshold)
    result.update(
        {
            "model": model_name,
            "available": True,
            "weights_path": weights_path,
            "error": None,
        }
    )
    cm = result["confusion_matrix"]
    print(
        f"  accuracy={result['accuracy']:.4f}  precision={result['precision']:.4f}  "
        f"recall={result['recall']:.4f}  f1={result['f1']:.4f}  auc={result['auc']}",
        flush=True,
    )
    print(f"  confusion: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}", flush=True)
    return result


def _resolve_models(model_arg: str) -> list[str]:
    key = model_arg.strip().lower()
    if key == "all":
        return list(AVAILABLE_MODELS)
    return [normalize_model_name(model_arg)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External validation of trained pneumonia models (no retraining)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to external dataset root with NORMAL/ and PNEUMONIA/",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model name or 'all' (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR / "reports" / "bhopal_external_validation.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pneumonia decision threshold (default: PNEUMONIA_THRESHOLD env or 0.65)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset if args.dataset.is_absolute() else (BACKEND_DIR / args.dataset)
    dataset_dir = dataset_dir.resolve()
    threshold = args.threshold if args.threshold is not None else _threshold()
    samples = _collect_images(dataset_dir)
    models = _resolve_models(args.model)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(
        f"Images: {len(samples)} "
        f"(NORMAL={sum(1 for _, y in samples if y == 0)}, "
        f"PNEUMONIA={sum(1 for _, y in samples if y == 1)})",
        flush=True,
    )
    print(f"Threshold: {threshold}", flush=True)
    print(f"Models: {', '.join(models)}", flush=True)

    results = [evaluate_model(name, samples, threshold) for name in models]

    report = {
        "dataset": str(dataset_dir),
        "dataset_source": (
            "Mendeley Data 10.17632/kpg5yz77gj.1 — "
            "Detection of Pneumonia Disease using Chest Radiograph Image Dataset "
            "(VIT Bhopal University / Bhopal pathology labs)"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "n_images": len(samples),
        "models": results,
    }

    output_path = args.output if args.output.is_absolute() else (BACKEND_DIR / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
