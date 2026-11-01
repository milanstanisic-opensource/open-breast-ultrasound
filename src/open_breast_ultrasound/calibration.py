"""Validation-only temperature scaling and optional threshold selection.

Fitting uses validation predictions only. Nothing here reads a test label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import ProjectError, read_json, write_json

TEMPERATURE_DISCLAIMER = (
    "Temperature scaling was fit on validation predictions only. "
    "A lower validation log loss does not show clinical reliability, "
    "calibration in another hospital, or a safe decision threshold."
)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Scale probabilities by ``temperature`` via log-probabilities.

    ``softmax(log(p) / T)`` equals ``p`` when ``T`` is 1 and rows already sum to 1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ProjectError("Probabilities must be a two-dimensional array.")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ProjectError("Temperature must be a positive finite number.")
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    logits = logits - np.max(logits, axis=1, keepdims=True)
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def multiclass_nll(probs: np.ndarray, y_index: Sequence[int]) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y_index = np.asarray(list(y_index), dtype=int)
    if len(probs) == 0:
        raise ProjectError("Cannot compute NLL without rows.")
    chosen = np.clip(probs[np.arange(len(y_index)), y_index], 1e-15, 1.0)
    return float(np.mean(-np.log(chosen)))


def fit_temperature(probs: np.ndarray, y_index: Sequence[int], *, t_min: float = 0.05, t_max: float = 10.0) -> dict[str, Any]:
    """Minimize validation multiclass NLL over a positive temperature."""
    if t_min <= 0 or t_max <= t_min:
        raise ProjectError("Temperature search bounds are invalid.")
    y_index = list(y_index)
    if len(y_index) == 0:
        raise ProjectError("Cannot fit a temperature without validation rows.")

    def nll(temperature: float) -> float:
        return multiclass_nll(apply_temperature(probs, temperature), y_index)

    grid = np.linspace(t_min, t_max, 80)
    grid_values = [nll(float(temperature)) for temperature in grid]
    best_index = int(np.argmin(grid_values))
    span = (t_max - t_min) / 79.0
    left = max(t_min, float(grid[best_index]) - span)
    right = min(t_max, float(grid[best_index]) + span)
    phi = (1.0 + 5.0**0.5) / 2.0
    for _ in range(50):
        left_probe = right - (right - left) / phi
        right_probe = left + (right - left) / phi
        if nll(left_probe) < nll(right_probe):
            right = right_probe
        else:
            left = left_probe
    temperature = float((left + right) / 2.0)
    return {
        "temperature": temperature,
        "validation_nll": nll(temperature),
        "validation_nll_at_temperature_1": nll(1.0),
        "n_validation_images": len(y_index),
        "fit_on": "validation",
        "test_used": False,
        "disclaimer": TEMPERATURE_DISCLAIMER,
        "clinically_validated": False,
    }


def run_calibration(
    experiment_dir: Path,
    data_root: Path,
    inventory_path: Path,
    split_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit temperature on validation predictions. Optionally lock a validation threshold.

    Does not train, does not read test labels for fitting, and does not change weights.
    """
    from .audit import assert_files_unchanged, load_inventory
    from .evaluate import select_threshold_on_validation
    from .inference import load_model_only, predict_records
    from .utils import CLASS_NAMES

    experiment_dir = Path(experiment_dir)
    metadata_path = experiment_dir / "experiment.json"
    metadata = read_json(metadata_path)
    temperature_path = experiment_dir / "temperature.json"
    if temperature_path.exists() and not overwrite:
        raise ProjectError(
            f"Refusing to overwrite {temperature_path}. "
            "Calibration is not re-fit silently. Pass --overwrite only if you intend to replace it."
        )
    inventory = load_inventory(inventory_path)
    from .splits import assert_groups_fit_split, load_split_manifest

    manifest = load_split_manifest(split_path)
    assert_files_unchanged(inventory, data_root)
    assert_groups_fit_split(inventory, manifest)
    validation_rows = [row for row in manifest if row["split"] == "validation"]
    test_rows = [row for row in manifest if row["split"] == "test"]
    if not validation_rows:
        raise ProjectError("The split manifest has no validation rows.")
    model, class_names, image_size = load_model_only(experiment_dir)
    if tuple(class_names) != tuple(CLASS_NAMES):
        raise ProjectError("Experiment class order does not match this release.")
    probs = predict_records_with_model(
        model,
        validation_rows,
        data_root,
        class_names,
        image_size=image_size,
        temperature=1.0,
    )
    y_index = [class_names.index(row["label"]) for row in validation_rows]
    fitted = fit_temperature(probs, y_index)
    malignant_scores = apply_temperature(probs, fitted["temperature"])[:, class_names.index("malignant")]
    policy = metadata.get("threshold_policy", "predefined")
    if policy == "predefined":
        threshold_info = {
            "threshold": metadata.get("decision_threshold"),
            "criterion": "predefined in the training config",
            "fit_on": None,
            "clinically_approved": False,
            "changed": False,
        }
    elif policy == "validation_max_f1":
        labels = [row["label"] for row in validation_rows]
        threshold_info = select_threshold_on_validation(labels, malignant_scores)
        threshold_info["changed"] = True
    else:
        raise ProjectError(f"Unknown threshold policy {policy!r}.")
    if threshold_info["threshold"] is None:
        raise ProjectError("Calibration finished without a threshold. Nothing was written as zero.")
    # Guard against an accidental test-label argument. test_rows is counted only.
    fitted["n_test_images_not_used"] = len(test_rows)
    fitted["threshold"] = threshold_info
    write_json(temperature_path, fitted)
    metadata["temperature"] = fitted["temperature"]
    metadata["temperature_fit_on"] = "validation"
    metadata["decision_threshold"] = threshold_info["threshold"]
    metadata["threshold_locked"] = True
    metadata["threshold_clinically_approved"] = False
    metadata["threshold_selection"] = threshold_info
    write_json(metadata_path, metadata)
    return {
        "temperature": fitted["temperature"],
        "decision_threshold": threshold_info["threshold"],
        "n_validation_images": len(validation_rows),
        "n_test_images_not_used": len(test_rows),
    }


def predict_records_with_model(model, rows, data_root, class_names, *, image_size: int, temperature: float):
    from .inference import Runtime, predict_records

    runtime = Runtime(
        model=model,
        class_names=tuple(class_names),
        image_size=int(image_size),
        temperature=float(temperature),
        threshold=None,
        threshold_locked=False,
        model_sha256=None,
        experiment_dir=Path("."),
    )
    return predict_records(runtime, rows, data_root)
