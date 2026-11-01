"""Metrics and locked-model evaluation.

Missing classes produce null metrics and a reason. They never produce a
placeholder zero that could be mistaken for a measured result.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .utils import CLASS_NAMES, ProjectError, write_csv, write_json

REPORT_DISCLAIMER = (
    "These are research metrics for a whole-image classifier. "
    "They are not a clinical validation, not regulatory evidence, and not proof "
    "that the model improves diagnosis or early detection. "
    "Temperature scaling and a validation-chosen threshold do not make scores clinically reliable. "
    "A held-out split from the same source is not an external multi-site validation."
)
BINARY_DEFINITION = (
    "Positive class is malignant. Every other label is negative. "
    "The score is the model probability of malignant after temperature scaling "
    "(temperature 1 means no scaling). "
    "The binary decision is score >= threshold. "
    "This is separate from the multiclass argmax decision. "
    "The threshold is not clinically approved."
)
BRIER_DEFINITION = (
    "Multiclass Brier score is the mean over images of the sum over classes of "
    "(probability_k - one_hot_k)^2, using the model's full class list."
)
BOOTSTRAP_ASSUMPTIONS = (
    "Percentile intervals resample groups with replacement, not images as if they "
    "were independent patients. Groups are duplicate or supplied-id components, "
    "or single images when no link is known. They are not confirmed patients. "
    "Small numbers of groups make the interval unstable. "
    "Correlated images are not treated as independent patients."
)


def _as_labels(labels: Sequence[str]) -> list[str]:
    return [str(label) for label in labels]


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        if true not in index or pred not in index:
            raise ProjectError(f"Label outside the class list: true={true!r} pred={pred!r}")
        matrix[index[true]][index[pred]] += 1
    return matrix


def per_class_scores(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> dict[str, dict[str, float | int | None]]:
    scores: dict[str, dict[str, float | int | None]] = {}
    for position, label in enumerate(labels):
        support = int(sum(matrix[position][column] for column in range(len(labels))))
        predicted = int(sum(matrix[row][position] for row in range(len(labels))))
        true_positive = int(matrix[position][position])
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        reason = None
        if support == 0:
            reason = f"No true examples of {label}; recall and F1 are undefined."
        elif predicted == 0:
            reason = f"The argmax decision never predicted {label}; precision and F1 are undefined."
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "reason": reason,
        }
    return scores


def binary_counts(y_true: Sequence[str], scores: Sequence[float], threshold: float, positive: str = "malignant") -> dict[str, int]:
    if len(y_true) != len(scores):
        raise ProjectError("Labels and scores have different lengths.")
    true_positive = false_positive = true_negative = false_negative = 0
    for label, score in zip(y_true, scores):
        actual = 1 if label == positive else 0
        predicted = 1 if float(score) >= threshold else 0
        if actual == 1 and predicted == 1:
            true_positive += 1
        elif actual == 0 and predicted == 1:
            false_positive += 1
        elif actual == 0 and predicted == 0:
            true_negative += 1
        else:
            false_negative += 1
    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def auroc(y_true_binary: Sequence[int], scores: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties. None if either class is absent."""
    y = np.asarray(list(y_true_binary), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ProjectError("AUROC inputs must be one-dimensional and the same length.")
    n_pos = int(np.sum(y == 1))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(y.size, dtype=float)
    start = 0
    while start < y.size:
        end = start
        while end + 1 < y.size and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + (end + 1))
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    sum_positive_ranks = float(ranks[y == 1].sum())
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true_binary: Sequence[int], scores: Sequence[float]) -> float | None:
    """Non-interpolated average precision.

    Operating points are sorted by descending score, with ties broken by a stable
    sort. Average precision is sum (recall_n - recall_(n-1)) * precision_n,
    starting from recall 0. None unless both classes are present.
    """
    y = np.asarray(list(y_true_binary), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    total = 0.0
    for label in y_sorted:
        if label == 1:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / n_pos
        if recall != previous_recall:
            total += (recall - previous_recall) * precision
            previous_recall = recall
    return float(total)


def multiclass_brier(probs: np.ndarray, y_index: Sequence[int]) -> float | None:
    probs = np.asarray(probs, dtype=float)
    y_index = np.asarray(list(y_index), dtype=int)
    if len(probs) == 0:
        return None
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_index)), y_index] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def binary_brier(scores: Sequence[float], y_true_binary: Sequence[int]) -> float | None:
    if len(scores) == 0:
        return None
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(y_true_binary), dtype=float)
    return float(np.mean((s - y) ** 2))


def negative_log_likelihood(probs: np.ndarray, y_index: Sequence[int]) -> float | None:
    probs = np.asarray(probs, dtype=float)
    y_index = np.asarray(list(y_index), dtype=int)
    if len(probs) == 0:
        return None
    chosen = np.clip(probs[np.arange(len(y_index)), y_index], 1e-15, 1.0)
    return float(np.mean(-np.log(chosen)))


def malignant_f1(y_true: Sequence[str], scores: Sequence[float], threshold: float) -> float | None:
    counts = binary_counts(y_true, scores, threshold)
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def select_threshold_on_validation(y_true: Sequence[str], scores: Sequence[float]) -> dict[str, Any]:
    """Pick a malignant threshold by maximum F1. Caller must pass validation rows only."""
    if len(y_true) == 0:
        raise ProjectError("Cannot select a threshold without validation rows.")
    candidates = sorted(set(np.round(np.asarray(list(scores), dtype=float), 6).tolist()) | {0.5})
    best_threshold = None
    best_f1 = None
    for threshold in candidates:
        f1 = malignant_f1(y_true, scores, float(threshold))
        if f1 is None:
            continue
        if best_f1 is None or f1 > best_f1 or (f1 == best_f1 and float(threshold) < float(best_threshold)):
            best_f1 = f1
            best_threshold = float(threshold)
    if best_threshold is None:
        raise ProjectError(
            "Malignant F1 was undefined at every validation threshold candidate. "
            "The threshold was not selected on test data, and no fallback zero was stored."
        )
    return {
        "threshold": best_threshold,
        "validation_malignant_f1": best_f1,
        "criterion": "maximum F1 of the malignant-versus-rest decision on validation predictions",
        "clinically_approved": False,
        "fit_on": "validation",
    }


def reliability_table(y_true: Sequence[str], scores: Sequence[float], *, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width bins of the malignant score. Empty bins have null rates, not zero."""
    if n_bins < 2:
        raise ProjectError("n_bins must be at least 2.")
    y = np.asarray([1 if label == "malignant" else 0 for label in y_true], dtype=int)
    s = np.asarray(list(scores), dtype=float)
    rows = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            mask = (s >= low) & (s <= high)
        else:
            mask = (s >= low) & (s < high)
        count = int(np.sum(mask))
        if count == 0:
            mean_score = None
            fraction = None
        else:
            mean_score = float(np.mean(s[mask]))
            fraction = float(np.mean(y[mask]))
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "mean_predicted": mean_score,
                "fraction_malignant": fraction,
            }
        )
    return rows


def _unavailable(metric: str, reason: str) -> dict[str, str]:
    return {"metric": metric, "reason": reason}


def group_bootstrap(
    records: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if n_bootstrap < 1:
        raise ProjectError("n_bootstrap must be positive.")
    groups = sorted({str(record["group_id"]) for record in records})
    point = metric_fn(records)
    result: dict[str, Any] = {
        "estimate": point,
        "ci_low": None,
        "ci_high": None,
        "n_groups": len(groups),
        "n_bootstrap": n_bootstrap,
        "assumptions": BOOTSTRAP_ASSUMPTIONS,
    }
    if len(groups) < 2:
        result["reason"] = "Fewer than 2 groups, so a group bootstrap interval was not estimated."
        return result
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group_id"])].append(record)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    skipped = 0
    for _ in range(n_bootstrap):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        resampled: list[Mapping[str, Any]] = []
        for group_id in drawn:
            resampled.extend(by_group[str(group_id)])
        value = metric_fn(resampled)
        if value is None:
            skipped += 1
            continue
        samples.append(float(value))
    result["skipped_replicates"] = skipped
    result["n_replicates"] = len(samples)
    if len(samples) < max(10, int(0.5 * n_bootstrap)):
        result["reason"] = "Too many bootstrap replicates were undefined, so the interval was not reported."
        return result
    low, high = np.percentile(np.asarray(samples, dtype=float), [2.5, 97.5])
    result["ci_low"] = float(low)
    result["ci_high"] = float(high)
    return result


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    probs: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    threshold: float,
    temperature: float,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
    split_name: str,
    source: str,
) -> dict[str, Any]:
    """Score an already-run model. ``records`` align with ``probs`` rows."""
    class_names = tuple(class_names)
    if list(class_names) != list(CLASS_NAMES):
        raise ProjectError(f"This release expects class order {list(CLASS_NAMES)}.")
    probs = np.asarray(probs, dtype=float)
    if len(records) != len(probs):
        raise ProjectError("Prediction rows and probability rows differ in length.")
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ProjectError(f"Expected probabilities with {len(class_names)} columns.")
    labels = _as_labels([record["label"] for record in records])
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        raise ProjectError(f"Labels are outside the model classes: {unknown}")
    index_of = {name: position for position, name in enumerate(class_names)}
    y_index = [index_of[label] for label in labels]
    argmax = [class_names[int(position)] for position in np.argmax(probs, axis=1)]
    malignant_index = index_of["malignant"]
    malignant_scores = probs[:, malignant_index]
    matrix = confusion_matrix(labels, argmax, class_names)
    per_class = per_class_scores(matrix, class_names)
    counts = binary_counts(labels, malignant_scores, threshold)
    y_binary = [1 if label == "malignant" else 0 for label in labels]
    n_positive = int(sum(y_binary))
    n_negative = len(y_binary) - n_positive
    unavailable: list[dict[str, str]] = []

    sensitivity = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    specificity = safe_divide(counts["tn"], counts["tn"] + counts["fp"])
    if sensitivity is None:
        unavailable.append(_unavailable("malignant_sensitivity", "No malignant examples; the denominator is zero."))
    if specificity is None:
        unavailable.append(_unavailable("specificity", "No non-malignant examples; the denominator is zero."))
    roc = auroc(y_binary, malignant_scores)
    if roc is None:
        unavailable.append(
            _unavailable(
                "malignant_one_vs_rest_auroc",
                "AUROC is undefined unless both malignant and non-malignant examples are present.",
            )
        )
    ap = average_precision(y_binary, malignant_scores)
    if ap is None:
        unavailable.append(
            _unavailable(
                "average_precision",
                "Average precision is undefined unless both malignant and non-malignant examples are present.",
            )
        )
    correct = sum(true == pred for true, pred in zip(labels, argmax))
    accuracy = safe_divide(correct, len(labels))
    present = sorted({label for label in labels})
    classes_absent = [name for name in class_names if name not in present]
    for name in classes_absent:
        if per_class[name]["reason"]:
            unavailable.append(_unavailable(f"{name}_recall", str(per_class[name]["reason"])))
    brier = multiclass_brier(probs, y_index)
    mal_brier = binary_brier(malignant_scores, y_binary)
    nll = negative_log_likelihood(probs, y_index)
    group_ids = [str(record["group_id"]) for record in records]
    normal_support = int(per_class["normal"]["support"])
    report: dict[str, Any] = {
        "disclaimer": REPORT_DISCLAIMER,
        "source": source,
        "split": split_name,
        "same_source_held_out_split": source == "internal_held_out_split",
        "external_validation": source == "external_manifest",
        "patient_level_independence": False,
        "n_images": len(records),
        "n_groups": len(set(group_ids)),
        "class_counts": {name: int(per_class[name]["support"]) for name in class_names},
        "classes_present": present,
        "classes_absent": classes_absent,
        "normal_examples_invented": False,
        "temperature": temperature,
        "decision_threshold": threshold,
        "threshold_clinically_approved": False,
        "binary_definition": BINARY_DEFINITION,
        "multiclass_argmax": {
            "confusion_matrix": {"labels": list(class_names), "matrix": matrix, "rows": "true", "columns": "predicted"},
            "per_class": per_class,
            "accuracy": accuracy,
            "note": "Accuracy and the confusion matrix use argmax. They are not the malignant threshold decision.",
        },
        "malignant_binary": {
            "definition": BINARY_DEFINITION,
            "counts": counts,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auroc": roc,
            "auroc_name": "malignant one-vs-rest AUROC",
            "average_precision": ap,
            "average_precision_name": "average_precision",
            "binary_brier": mal_brier,
            "binary_brier_definition": "Mean squared error between P(malignant) and 1 if the label is malignant, else 0.",
        },
        "multiclass_brier": {"value": brier, "definition": BRIER_DEFINITION},
        "negative_log_likelihood": nll,
        "normal_class_note": (
            None
            if normal_support > 0
            else "No normal examples were present and none were invented. "
            "Normal recall is null. Multiclass Brier still scores the model's normal probability against a one-hot target of 0."
        ),
        "unavailable": unavailable,
        "reliability_malignant": reliability_table(labels, malignant_scores),
    }
    if n_bootstrap:
        annotated = []
        for record, label, score in zip(records, labels, malignant_scores):
            annotated.append(
                {
                    **dict(record),
                    "label": label,
                    "malignant_score": float(score),
                    "threshold": float(threshold),
                }
            )
        report["bootstrap"] = {
            "malignant_sensitivity": group_bootstrap(
                annotated,
                lambda items: _metric_from_counts(items, "sensitivity"),
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            ),
            "specificity": group_bootstrap(
                annotated,
                lambda items: _metric_from_counts(items, "specificity"),
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + 1,
            ),
            "malignant_one_vs_rest_auroc": group_bootstrap(
                annotated, _auroc_of, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 2
            ),
            "average_precision": group_bootstrap(
                annotated, _ap_of, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 3
            ),
        }
    else:
        report["bootstrap"] = None
    return report


def _metric_from_counts(items: Sequence[Mapping[str, Any]], kind: str) -> float | None:
    labels = [str(item["label"]) for item in items]
    scores = [float(item["malignant_score"]) for item in items]
    if not items or "threshold" not in items[0]:
        raise ProjectError("Bootstrap records are missing the locked threshold.")
    threshold = float(items[0]["threshold"])
    counts = binary_counts(labels, scores, threshold)
    if kind == "sensitivity":
        return safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    if kind == "specificity":
        return safe_divide(counts["tn"], counts["tn"] + counts["fp"])
    raise ProjectError(f"Unknown bootstrap metric {kind}.")


def _auroc_of(items: Sequence[Mapping[str, Any]]) -> float | None:
    y = [1 if item["label"] == "malignant" else 0 for item in items]
    return auroc(y, [float(item["malignant_score"]) for item in items])


def _ap_of(items: Sequence[Mapping[str, Any]]) -> float | None:
    y = [1 if item["label"] == "malignant" else 0 for item in items]
    return average_precision(y, [float(item["malignant_score"]) for item in items])


def attach_threshold(records: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [{**dict(record), "threshold": threshold} for record in records]


def write_evaluation_files(output_dir: Path, report: dict[str, Any], prediction_rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    matrix = report["multiclass_argmax"]["confusion_matrix"]
    confusion_rows = []
    for label, row in zip(matrix["labels"], matrix["matrix"]):
        confusion_rows.append({"true_label": label, **{f"pred_{name}": value for name, value in zip(matrix["labels"], row)}})
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["true_label", *[f"pred_{name}" for name in matrix["labels"]]],
    )
    write_csv(
        output_dir / "reliability_malignant.csv",
        report["reliability_malignant"],
        ["bin_low", "bin_high", "count", "mean_predicted", "fraction_malignant"],
    )
    write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        [
            "image_id",
            "relative_path",
            "group_id",
            "label",
            "split",
            "p_benign",
            "p_malignant",
            "p_normal",
            "argmax_label",
            "malignant_score",
            "binary_threshold",
            "binary_prediction",
        ],
    )
    if report.get("bootstrap"):
        write_json(output_dir / "bootstrap.json", report["bootstrap"])


def prediction_table(records: Sequence[Mapping[str, Any]], probs: np.ndarray, *, threshold: float, split_name: str) -> list[dict[str, Any]]:
    probs = np.asarray(probs, dtype=float)
    rows = []
    for record, prob in zip(records, probs):
        argmax = CLASS_NAMES[int(np.argmax(prob))]
        score = float(prob[CLASS_NAMES.index("malignant")])
        rows.append(
            {
                "image_id": record["image_id"],
                "relative_path": record["relative_path"],
                "group_id": record["group_id"],
                "label": record["label"],
                "split": record.get("split", split_name),
                "p_benign": float(prob[0]),
                "p_malignant": float(prob[1]),
                "p_normal": float(prob[2]),
                "argmax_label": argmax,
                "malignant_score": score,
                "binary_threshold": threshold,
                "binary_prediction": "malignant" if score >= threshold else "not_malignant",
            }
        )
    return rows


def run_evaluation(
    experiment_dir: Path,
    data_root: Path,
    inventory_path: Path,
    split_path: Path,
    *,
    output_dir: Path | None = None,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate the locked model on the manifest's test split. Does not fit anything."""
    from .audit import assert_files_unchanged, load_inventory
    from .inference import load_runtime, predict_records
    from .splits import assert_groups_fit_split, load_split_manifest
    from .utils import ensure_fresh_directory

    runtime = load_runtime(experiment_dir)
    if not runtime.threshold_locked or runtime.threshold is None:
        raise ProjectError(
            "The decision threshold is not locked. "
            "If the policy is validation_max_f1, run calibration/threshold selection on validation first. "
            "Refusing to choose the threshold on the test split."
        )
    inventory = load_inventory(inventory_path)
    manifest = load_split_manifest(split_path)
    assert_files_unchanged(inventory, data_root)
    assert_groups_fit_split(inventory, manifest)
    test_rows = [row for row in manifest if row["split"] == "test"]
    if not test_rows:
        raise ProjectError("The split manifest has no test rows.")
    probs = predict_records(runtime, test_rows, data_root)
    annotated = attach_threshold(test_rows, float(runtime.threshold))
    report = evaluate_predictions(
        annotated,
        probs,
        class_names=runtime.class_names,
        threshold=float(runtime.threshold),
        temperature=float(runtime.temperature),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        split_name="test",
        source="internal_held_out_split",
    )
    report["experiment_dir"] = str(experiment_dir)
    report["model_sha256"] = runtime.model_sha256
    table = prediction_table(test_rows, probs, threshold=float(runtime.threshold), split_name="test")
    destination = Path(output_dir) if output_dir else Path(experiment_dir) / "evaluation"
    ensure_fresh_directory(destination, overwrite=overwrite)
    write_evaluation_files(destination, report, table)
    return {"output_dir": str(destination), "n_images": report["n_images"], "unavailable": report["unavailable"]}


def run_external_evaluation(
    experiment_dir: Path,
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score a separate manifest with the locked model, temperature, and threshold."""
    from .inference import load_runtime, predict_records
    from .utils import ensure_fresh_directory, read_csv

    runtime = load_runtime(experiment_dir)
    if not runtime.threshold_locked or runtime.threshold is None:
        raise ProjectError("The locked experiment has no decision threshold. Refusing to choose one on external data.")
    rows = read_csv(manifest_path)
    if not rows:
        raise ProjectError("External manifest is empty.")
    for column in ("relative_path", "label"):
        if column not in rows[0]:
            raise ProjectError("External manifest needs relative_path and label columns.")
    records = []
    for index, row in enumerate(rows):
        label = row["label"].strip()
        if label not in CLASS_NAMES:
            raise ProjectError(f"External label {label!r} is not one of {list(CLASS_NAMES)}.")
        group_id = (row.get("group_id") or "").strip() or f"solo:{index}"
        records.append(
            {
                "image_id": (row.get("image_id") or "").strip() or f"ext_{index:05d}",
                "relative_path": row["relative_path"].strip(),
                "label": label,
                "group_id": group_id,
                "split": "external",
            }
        )
    probs = predict_records(runtime, records, data_root)
    annotated = attach_threshold(records, float(runtime.threshold))
    report = evaluate_predictions(
        annotated,
        probs,
        class_names=runtime.class_names,
        threshold=float(runtime.threshold),
        temperature=float(runtime.temperature),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        split_name="external",
        source="external_manifest",
    )
    report["retrained"] = False
    report["temperature_refit"] = False
    report["threshold_changed"] = False
    report["partial_label_set"] = "normal" in report["classes_absent"]
    if report["partial_label_set"]:
        report["partial_label_note"] = (
            "This manifest does not contain every model class. "
            "Missing classes were left absent. No normal, benign, or malignant examples were invented."
        )
    table = prediction_table(records, probs, threshold=float(runtime.threshold), split_name="external")
    destination = ensure_fresh_directory(Path(output_dir), overwrite=overwrite)
    write_evaluation_files(destination, report, table)
    return {"output_dir": str(destination), "n_images": len(records), "classes_absent": report["classes_absent"]}
