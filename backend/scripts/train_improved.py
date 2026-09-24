"""
Retrain all pneumonia CNNs with a locked, stratified 70/15/15 split.

The test split is never used for training, early stopping, learning-rate
selection, dropout selection, or threshold selection.

Usage (from backend/):
    python scripts/train_improved.py --dataset "C:\\Users\\jewel\\OneDrive\\Scans\\chest_xray"
    python scripts/train_improved.py --dataset path\\to\\chest_xray --model MobileNetV2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_builder import (  # noqa: E402
    KERAS_CUSTOM_OBJECTS,
    WEIGHTS_DIR,
    build_transfer_model,
    compile_classifier,
    unfreeze_top_layers,
)
from services.model_registry import (  # noqa: E402
    AVAILABLE_MODELS,
    get_image_size,
    get_weights_path,
    normalize_model_name,
)

REPORT_DIR = BACKEND_DIR / "reports" / "improved_training"
SPLIT_PATH = REPORT_DIR / "split.json"
RESULTS_PATH = REPORT_DIR / "test_results.json"
THRESHOLDS_PATH = WEIGHTS_DIR / "thresholds.json"
BACKUP_DIR = WEIGHTS_DIR / "pre_improved_backup"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THRESHOLD_GRID = (0.40, 0.45, 0.50, 0.55, 0.60)
PROBE_LR = (1e-3, 1e-4, 1e-5)
PROBE_DROPOUT = (0.2, 0.3, 0.5)
SEED = 42

# How many trailing backbone layers to unfreeze in stage 2.
FINE_TUNE_LAYERS = {
    "MobileNetV2": 40,
    "DenseNet121": 80,
    "ResNet50": 33,
    "EfficientNetB0": 40,
    "VGG16": 8,
    "InceptionV3": 50,
}


def _log(message: str) -> None:
    print(message, flush=True)


def collect_images(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pool the Kaggle train/val/test folders. The official val set is only 16 images."""
    dataset_dir = Path(dataset_dir)
    paths: list[str] = []
    labels: list[int] = []
    for split in ("train", "val", "test"):
        for name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            folder = dataset_dir / split / name
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(path.resolve()))
                    labels.append(label)
    if len(paths) < 100:
        raise FileNotFoundError(
            f"Expected train/val/test with NORMAL and PNEUMONIA under {dataset_dir}. "
            f"Found {len(paths)} images."
        )
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def stratified_split(paths: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, list]]:
    """70% train, 15% validation, 15% test. Test is held out from this point on."""
    rest_paths, test_paths, rest_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=SEED,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        rest_paths,
        rest_labels,
        test_size=0.15 / 0.85,
        stratify=rest_labels,
        random_state=SEED,
    )
    return {
        "train": {"paths": train_paths.tolist(), "labels": train_labels.astype(int).tolist()},
        "val": {"paths": val_paths.tolist(), "labels": val_labels.astype(int).tolist()},
        "test": {"paths": test_paths.tolist(), "labels": test_labels.astype(int).tolist()},
    }


def _counts(labels: list[int]) -> dict[str, int]:
    arr = np.asarray(labels)
    return {"normal": int(np.sum(arr == 0)), "pneumonia": int(np.sum(arr == 1)), "total": int(len(arr))}


def class_weights(labels: list[int]) -> dict[int, float]:
    counts = _counts(labels)
    total = counts["total"]
    return {
        0: total / (2.0 * counts["normal"]),
        1: total / (2.0 * counts["pneumonia"]),
    }


def make_augmenter():
    import tensorflow as tf

    # Images stay in 0–255 until the model applies architecture-specific preprocess_input.
    # Brightness factor 0.1 is ±10% of the 0–255 range.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(10 / 360, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="cxr_augment",
    )


class XraySequence:
    """Loads chest X-rays on the fly. Augmentation is applied only when training."""

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        image_size: tuple[int, int],
        batch_size: int,
        augment: bool,
        shuffle: bool,
        images: np.ndarray | None = None,
    ):
        self.paths = np.asarray(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.images = None if images is None else np.asarray(images)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmenter = make_augmenter() if augment else None
        if shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.paths) / self.batch_size))

    def on_epoch_end(self) -> None:
        if not self.shuffle:
            return
        order = np.random.permutation(len(self.paths))
        self.paths = self.paths[order]
        self.labels = self.labels[order]
        if self.images is not None:
            self.images = self.images[order]

    def __getitem__(self, index: int):
        import tensorflow as tf

        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.paths))
        if self.images is not None:
            batch = self.images[start:end].astype(np.float32)
        else:
            images = []
            for path in self.paths[start:end]:
                img = tf.keras.utils.load_img(path, target_size=self.image_size, color_mode="rgb")
                images.append(tf.keras.utils.img_to_array(img))
            batch = np.stack(images).astype(np.float32)
        if self.augmenter is not None:
            batch = self.augmenter(batch, training=True).numpy()
            batch = np.clip(batch, 0.0, 255.0)
        return batch, self.labels[start:end]


def build_sequence(paths, labels, image_size, batch_size, augment, shuffle, images=None):
    import tensorflow as tf

    class _Seq(XraySequence, tf.keras.utils.Sequence):
        pass

    return _Seq(paths, labels, image_size, batch_size, augment, shuffle, images=images)


def preload_images(paths: list[str], image_size: tuple[int, int]) -> np.ndarray:
    """Decode each X-ray once so later epochs do not reread OneDrive."""
    from PIL import Image

    height, width = image_size
    cache = np.empty((len(paths), height, width, 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        with Image.open(path) as img:
            img = img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            cache[index] = np.asarray(img, dtype=np.uint8)
        if (index + 1) % 1000 == 0 or index + 1 == len(paths):
            _log(f"  cached {index + 1}/{len(paths)}")
    return cache


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "specificity": round(float(specificity), 4),
        "auc": None if auc is None else round(auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, list[dict]]:
    """Pick a threshold on validation only. Tie-break toward balanced precision/recall."""
    rows = [binary_metrics(y_true, scores, threshold) for threshold in THRESHOLD_GRID]
    best = max(rows, key=lambda row: (row["f1"], -abs(row["precision"] - row["recall"])))
    return float(best["threshold"]), rows


def predict_scores(model, paths, labels, image_size, batch_size, images=None) -> np.ndarray:
    sequence = build_sequence(
        paths, labels, image_size, batch_size, augment=False, shuffle=False, images=images
    )
    raw = model.predict(sequence, verbose=0)
    return np.clip(np.asarray(raw).reshape(-1), 0.0, 1.0)


def _callbacks(checkpoint_path: Path, patience: int = 3):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=patience,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]


def _subset(paths: list[str], labels: list[int], n: int, seed: int) -> tuple[list[str], list[int]]:
    paths_arr = np.asarray(paths)
    labels_arr = np.asarray(labels)
    if len(paths_arr) <= n:
        return paths, labels
    chosen, _, chosen_y, _ = train_test_split(
        paths_arr,
        labels_arr,
        train_size=n,
        stratify=labels_arr,
        random_state=seed,
    )
    return chosen.tolist(), chosen_y.astype(int).tolist()


def probe_hyperparameters(
    model_name: str,
    train_paths: list[str],
    train_labels: list[int],
    val_paths: list[str],
    val_labels: list[int],
    image_size: tuple[int, int],
    weights: dict[int, float],
) -> dict:
    """1-epoch probes on a training subset, scored only on a validation subset."""
    import tensorflow as tf

    probe_train_x, probe_train_y = _subset(train_paths, train_labels, 320, SEED)
    probe_val_x, probe_val_y = _subset(val_paths, val_labels, 160, SEED + 1)
    trials = []

    def _run(dropout: float, learning_rate: float) -> float:
        tf.keras.backend.clear_session()
        model = build_transfer_model(
            model_name,
            trainable_base=False,
            learning_rate=learning_rate,
            dropout_rate=dropout,
        )
        history = model.fit(
            build_sequence(probe_train_x, probe_train_y, image_size, 32, True, True),
            validation_data=build_sequence(probe_val_x, probe_val_y, image_size, 32, False, False),
            epochs=1,
            class_weight=weights,
            verbose=2,
        )
        auc = float(history.history["val_auc"][-1])
        del model
        gc.collect()
        return auc

    dropout_scores = []
    for dropout in PROBE_DROPOUT:
        auc = _run(dropout, 1e-3)
        dropout_scores.append({"dropout": dropout, "learning_rate": 1e-3, "val_auc": round(auc, 4)})
        _log(f"  probe dropout={dropout} lr=1e-3 val_auc={auc:.4f}")
    best_dropout = max(dropout_scores, key=lambda row: row["val_auc"])["dropout"]

    lr_scores = [row for row in dropout_scores if row["dropout"] == best_dropout]
    for learning_rate in (1e-4, 1e-5):
        auc = _run(best_dropout, learning_rate)
        row = {"dropout": best_dropout, "learning_rate": learning_rate, "val_auc": round(auc, 4)}
        lr_scores.append(row)
        _log(f"  probe dropout={best_dropout} lr={learning_rate} val_auc={auc:.4f}")
    best_lr = max(lr_scores, key=lambda row: row["val_auc"])["learning_rate"]
    trials.extend(dropout_scores)
    trials.extend(row for row in lr_scores if row["learning_rate"] != 1e-3)
    return {
        "dropout": best_dropout,
        "stage1_lr": best_lr,
        "stage2_lr": float(np.clip(best_lr / 100.0, 1e-6, 1e-4)),
        "trials": trials,
        "probe_train_images": len(probe_train_x),
        "probe_val_images": len(probe_val_x),
    }


def train_one(
    model_name: str,
    split: dict,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    fast: bool = False,
) -> dict:
    import tensorflow as tf

    image_size = get_image_size(model_name)
    train_paths = split["train"]["paths"]
    train_labels = split["train"]["labels"]
    val_paths = split["val"]["paths"]
    val_labels = split["val"]["labels"]
    test_paths = split["test"]["paths"]
    test_labels = split["test"]["labels"]
    weights = class_weights(train_labels)
    patience = 2 if fast else 3

    _log(f"\n===== {model_name} =====")
    _log(f"Input {image_size[0]}x{image_size[1]} | class weights {weights}")
    train_images = val_images = test_images = None
    if fast:
        _log("Caching resized images in memory")
        cached = preload_images(train_paths + val_paths + test_paths, image_size)
        n_train = len(train_paths)
        n_val = len(val_paths)
        train_images = cached[:n_train]
        val_images = cached[n_train : n_train + n_val]
        test_images = cached[n_train + n_val :]

    hparams_path = REPORT_DIR / f"{model_name}_hparams.json"
    stage1_weights = REPORT_DIR / f"{model_name}_stage1.keras"
    checkpoint = REPORT_DIR / f"{model_name}_best.weights.h5"
    train_seq = build_sequence(
        train_paths, train_labels, image_size, batch_size, True, True, images=train_images
    )
    val_seq = build_sequence(
        val_paths, val_labels, image_size, batch_size, False, False, images=val_images
    )

    if hparams_path.exists() and stage1_weights.exists() and not fast:
        chosen = json.loads(hparams_path.read_text(encoding="utf-8"))
        _log("Resuming from saved stage-1 model (probe and stage 1 already finished).")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(stage1_weights, custom_objects=KERAS_CUSTOM_OBJECTS)
        val_scores_stage1 = predict_scores(
            model, val_paths, val_labels, image_size, batch_size, images=val_images
        )
        stage1_auc = float(roc_auc_score(val_labels, val_scores_stage1))
    else:
        if fast:
            # Match the learning rate that worked for the other four models.
            # VGG16 uses a bit more dropout because the backbone has many parameters.
            chosen = {
                "dropout": 0.4 if model_name == "VGG16" else 0.3,
                "stage1_lr": 1e-3,
                "stage2_lr": 1e-5,
                "trials": [],
                "fast": True,
            }
            _log("Fast mode: skipped the learning-rate/dropout search")
        else:
            chosen = probe_hyperparameters(
                model_name,
                train_paths,
                train_labels,
                val_paths,
                val_labels,
                image_size,
                weights,
            )
        hparams_path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
        _log(
            f"Selected dropout={chosen['dropout']} "
            f"stage1_lr={chosen['stage1_lr']} stage2_lr={chosen['stage2_lr']}"
        )

        tf.keras.backend.clear_session()
        model = build_transfer_model(
            model_name,
            trainable_base=False,
            learning_rate=chosen["stage1_lr"],
            dropout_rate=chosen["dropout"],
        )
        _log(f"Stage 1: frozen backbone, up to {stage1_epochs} epochs")
        model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=stage1_epochs,
            class_weight=weights,
            callbacks=_callbacks(checkpoint, patience=patience),
            verbose=2,
        )

        val_scores_stage1 = predict_scores(
            model, val_paths, val_labels, image_size, batch_size, images=val_images
        )
        stage1_auc = float(roc_auc_score(val_labels, val_scores_stage1))
        model.save(stage1_weights)
    _log(f"Stage 1 validation AUC={stage1_auc:.4f}")

    _log(
        f"Stage 2: unfreeze last {FINE_TUNE_LAYERS[model_name]} layers, "
        f"up to {stage2_epochs} epochs"
    )
    unfreeze_top_layers(model, num_layers=FINE_TUNE_LAYERS[model_name])
    compile_classifier(model, learning_rate=chosen["stage2_lr"])
    stage2_checkpoint = REPORT_DIR / f"{model_name}_stage2.weights.h5"
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=stage2_epochs,
        class_weight=weights,
        callbacks=_callbacks(stage2_checkpoint, patience=patience),
        verbose=2,
    )

    val_scores = predict_scores(
        model, val_paths, val_labels, image_size, batch_size, images=val_images
    )
    stage2_auc = float(roc_auc_score(val_labels, val_scores))
    _log(f"Stage 2 validation AUC={stage2_auc:.4f}")
    if stage2_auc + 1e-4 < stage1_auc:
        _log("Stage 2 did not improve validation AUC. Restoring stage 1 weights.")
        model = tf.keras.models.load_model(stage1_weights, custom_objects=KERAS_CUSTOM_OBJECTS)
        val_scores = val_scores_stage1
        kept_stage = 1
    else:
        kept_stage = 2

    threshold, val_threshold_table = choose_threshold(np.asarray(val_labels), val_scores)
    val_metrics = binary_metrics(np.asarray(val_labels), val_scores, threshold)
    _log(f"Validation-chosen threshold={threshold:.2f} | val F1={val_metrics['f1']:.4f}")

    # Test set is touched only here, after every choice is locked.
    test_scores = predict_scores(
        model, test_paths, test_labels, image_size, batch_size, images=test_images
    )
    test_metrics = binary_metrics(np.asarray(test_labels), test_scores, threshold)
    _log(
        "TEST "
        f"acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"auc={test_metrics['auc']:.4f}"
    )

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = get_weights_path(model_name)
    _backup_existing(output_path)
    model.save(output_path)
    _log(f"Saved {output_path}")

    if model_name == "MobileNetV2":
        _export_tflite()

    result = {
        "model": model_name,
        "image_size": list(image_size),
        "hyperparameters": chosen,
        "fine_tune_layers": FINE_TUNE_LAYERS[model_name],
        "kept_stage": kept_stage,
        "stage1_val_auc": round(stage1_auc, 4),
        "stage2_val_auc": round(stage2_auc, 4),
        "threshold": threshold,
        "validation_threshold_search": val_threshold_table,
        "validation": val_metrics,
        "test": test_metrics,
        "roc": {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr]},
        "weights_path": str(output_path),
    }
    _write_threshold(model_name, threshold)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / path.name
    if not dest.exists():
        shutil.copy2(path, dest)
        _log(f"Backed up previous weights to {dest}")


def _export_tflite() -> None:
    try:
        sys.path.insert(0, str(BACKEND_DIR / "scripts"))
        from export_tflite import export_model

        export_model("MobileNetV2")
    except Exception as exc:
        _log(f"TFLite export failed: {exc}")


def _write_threshold(model_name: str, threshold: float) -> None:
    data = {}
    if THRESHOLDS_PATH.exists():
        try:
            data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[model_name] = threshold
    THRESHOLDS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_or_create_split(dataset_dir: Path) -> dict:
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        _log(f"Reusing locked split {SPLIT_PATH}")
        return split
    paths, labels = collect_images(dataset_dir)
    split = stratified_split(paths, labels)
    split["dataset"] = str(dataset_dir)
    split["seed"] = SEED
    split["fractions"] = {"train": 0.70, "val": 0.15, "test": 0.15}
    split["counts"] = {name: _counts(split[name]["labels"]) for name in ("train", "val", "test")}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({k: split[k] for k in split if k != "unused"}, indent=2), encoding="utf-8")
    # The file above includes full path lists. That is intentional so the split stays fixed.
    return split


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"models": []}


def _save_results(payload: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_plots(models: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finished = [row for row in models if row.get("test")]
    if not finished:
        return

    names = [row["model"] for row in finished]
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        values = [row["test"][metric] for row in finished]
        ax.bar(x + (i - 2) * width, values, width, label=metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "comparison_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for row in finished:
        ax.plot(row["roc"]["fpr"], row["roc"]["tpr"], label=f"{row['model']} AUC={row['test']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    for row in finished:
        cm = row["test"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(row["model"])
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"confusion_{row['model']}.png", dpi=150)
        plt.close(fig)

    _log(f"Plots written to {REPORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved stratified training for all CNNs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Retrain even if a test result exists")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Skip the hyperparameter search and cache images")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        write_plots(_load_results().get("models", []))
        return

    split = load_or_create_split(args.dataset.resolve())
    for name in ("train", "val", "test"):
        counts = split.get("counts", {}).get(name) or _counts(split[name]["labels"])
        _log(f"{name}: {counts['total']} images ({counts['normal']} NORMAL, {counts['pneumonia']} PNEUMONIA)")

    if args.model.strip().lower() == "all":
        models = list(AVAILABLE_MODELS)
    else:
        models = [normalize_model_name(args.model)]

    payload = _load_results()
    payload["dataset"] = split.get("dataset", str(args.dataset))
    payload["split"] = split.get("counts") or {name: _counts(split[name]["labels"]) for name in ("train", "val", "test")}
    payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
    done = {row["model"] for row in payload["models"] if row.get("test")}

    for model_name in models:
        if model_name in done and not args.force:
            _log(f"Skipping {model_name}; test result already saved. Use --force to retrain.")
            continue
        result = train_one(
            model_name,
            split,
            batch_size=16 if model_name == "InceptionV3" else args.batch_size,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            fast=args.fast,
        )
        payload["models"] = [row for row in payload["models"] if row.get("model") != model_name]
        payload["models"].append(result)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_results(payload)
        write_plots(payload["models"])

    _log("\nFinal held-out test results")
    _log("Model | Accuracy | Precision | Recall | F1 | AUC")
    ordered = []
    by_name = {row["model"]: row for row in payload["models"]}
    for name in AVAILABLE_MODELS:
        if name in by_name and by_name[name].get("test"):
            ordered.append(by_name[name])
    for row in ordered:
        test = row["test"]
        _log(
            f"{row['model']} | {test['accuracy']:.4f} | {test['precision']:.4f} | "
            f"{test['recall']:.4f} | {test['f1']:.4f} | {test['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
