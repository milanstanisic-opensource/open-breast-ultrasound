"""Load a saved experiment and score images with the training preprocessor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import prepare_image, resolve_image_path
from .utils import CLASS_NAMES, ProjectError, read_json


@dataclass
class Runtime:
    model: object
    class_names: tuple[str, ...]
    image_size: int
    temperature: float
    threshold: float | None
    threshold_locked: bool
    model_sha256: str | None
    experiment_dir: Path


def load_model_only(experiment_dir: Path):
    metadata = read_json(Path(experiment_dir) / "experiment.json")
    import keras

    from .model import get_custom_objects

    model_path = Path(experiment_dir) / "model.keras"
    if not model_path.is_file():
        raise ProjectError(f"Saved model not found: {model_path}")
    model = keras.models.load_model(model_path, custom_objects=get_custom_objects(), compile=False)
    class_names = tuple(metadata["class_names"])
    if class_names != tuple(CLASS_NAMES):
        raise ProjectError(f"Saved class order {class_names} does not match {CLASS_NAMES}.")
    return model, list(class_names), int(metadata["image_size"])


def load_runtime(experiment_dir: Path) -> Runtime:
    experiment_dir = Path(experiment_dir)
    metadata = read_json(experiment_dir / "experiment.json")
    model, class_names, image_size = load_model_only(experiment_dir)
    temperature = 1.0
    temperature_path = experiment_dir / "temperature.json"
    if temperature_path.is_file():
        temperature = float(read_json(temperature_path)["temperature"])
    elif metadata.get("temperature") not in (None, ""):
        temperature = float(metadata["temperature"])
    threshold = metadata.get("decision_threshold")
    return Runtime(
        model=model,
        class_names=tuple(class_names),
        image_size=image_size,
        temperature=temperature,
        threshold=None if threshold is None else float(threshold),
        threshold_locked=bool(metadata.get("threshold_locked", False)),
        model_sha256=metadata.get("model_sha256"),
        experiment_dir=experiment_dir,
    )


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    """Inference preprocessing. Augmentation is not applied."""
    return prepare_image(path, image_size, training=False, augment_fn=None)


def predict_arrays(runtime: Runtime, arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ProjectError("No images were provided to score.")
    batch = np.stack([np.asarray(array, dtype=np.float32) for array in arrays], axis=0)
    expected = (len(arrays), runtime.image_size, runtime.image_size, 3)
    if batch.shape != expected:
        raise ProjectError(f"Expected batch shape {expected}, got {batch.shape}.")
    raw = runtime.model.predict(batch, verbose=0)
    probs = np.asarray(raw, dtype=np.float64)
    if probs.shape != (len(arrays), len(runtime.class_names)):
        raise ProjectError(f"Model returned shape {probs.shape}.")
    if abs(float(runtime.temperature) - 1.0) > 1e-8:
        from .calibration import apply_temperature

        probs = apply_temperature(probs, float(runtime.temperature))
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ProjectError("Model probabilities could not be normalized.")
    return probs / totals


def predict_records(runtime: Runtime, records: Sequence[dict], data_root: Path) -> np.ndarray:
    arrays = []
    for record in records:
        path = resolve_image_path(data_root, record["relative_path"])
        if not path.is_file():
            raise ProjectError(f"Image listed in the manifest is missing: {record['relative_path']}")
        arrays.append(preprocess_image(path, runtime.image_size))
    return predict_arrays(runtime, arrays)


def predict_image(runtime: Runtime, path: Path) -> dict[str, float]:
    array = preprocess_image(path, runtime.image_size)
    probs = predict_arrays(runtime, [array])[0]
    return {name: float(probs[index]) for index, name in enumerate(runtime.class_names)}


def format_scores(scores: dict[str, float], runtime: Runtime) -> str:
    lines = [
        "Model scores (not a diagnosis)",
        f"Temperature applied: {runtime.temperature:.4f}",
    ]
    if runtime.threshold is not None:
        lines.append(
            f"Research binary threshold for the malignant score: {runtime.threshold:.4f}. "
            "This threshold is not clinically approved."
        )
    for name in runtime.class_names:
        lines.append(f"{name}: {scores[name]:.6f}")
    malignant = scores["malignant"]
    if runtime.threshold is not None:
        decision = "malignant" if malignant >= runtime.threshold else "not_malignant"
        lines.append(f"Binary research decision at the locked threshold: {decision}")
    lines.append(f"Multiclass argmax label: {max(scores, key=scores.get)}")
    lines.append("Argmax and the binary threshold are different rules.")
    lines.append("This is a whole-image classification score, not a localization or a segmentation.")
    return "\n".join(lines)
