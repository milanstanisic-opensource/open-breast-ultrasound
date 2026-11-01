"""Two-phase training. The test split is never used for fitting or model selection."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import (
    REVIEW_NOT_MEDICAL,
    assert_files_unchanged,
    assert_reviewed_for_training,
    included_rows,
    load_inventory,
)
from .model import (
    balanced_class_weight_list,
    build_model,
    compile_model,
    configure_backbone,
    training_objective,
)
from .splits import assert_groups_fit_split, load_split_manifest
from .utils import (
    CLASS_NAMES,
    ProjectError,
    device_info,
    ensure_fresh_directory,
    git_commit,
    library_versions,
    read_yaml,
    set_global_seed,
    sha256_file,
    sha256_json,
    write_json,
    write_yaml,
)


def load_training_config(source: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        config = dict(source)
    else:
        config = read_yaml(Path(source))
    if tuple(config.get("classes", ())) != tuple(CLASS_NAMES):
        raise ProjectError(
            f"Config classes must be exactly {list(CLASS_NAMES)} in that order. "
            "The mapping is fixed so folders cannot silently change the output order."
        )
    for key in ("seed", "image_size", "batch_size", "model", "training", "augmentation", "decision"):
        if key not in config:
            raise ProjectError(f"Config is missing '{key}'.")
    if config["model"].get("backbone") != "EfficientNetB0":
        raise ProjectError("The implemented baseline backbone is EfficientNetB0.")
    policy = config["decision"].get("threshold_policy", "predefined")
    if policy not in {"predefined", "validation_max_f1"}:
        raise ProjectError("decision.threshold_policy must be 'predefined' or 'validation_max_f1'.")
    if policy == "predefined" and config["decision"].get("malignant_threshold") is None:
        raise ProjectError("A predefined threshold needs decision.malignant_threshold.")
    loss_name = config["training"].get("loss", "sparse_categorical_crossentropy")
    if loss_name not in {"sparse_categorical_crossentropy", "focal"}:
        raise ProjectError(f"Unsupported loss {loss_name!r}.")
    weights = config["model"].get("weights", "imagenet")
    if isinstance(weights, str) and weights.lower() in {"none", "null"}:
        config["model"]["weights"] = None
    return config


def rows_for_training(split_rows: Sequence[Mapping[str, str]]) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]], int]:
    train_rows = [row for row in split_rows if row["split"] == "train"]
    validation_rows = [row for row in split_rows if row["split"] == "validation"]
    n_test = sum(1 for row in split_rows if row["split"] == "test")
    if not train_rows or not validation_rows:
        raise ProjectError(
            "Training needs both a train split and a validation split. "
            "The test split is not used for fitting, early stopping, or model selection."
        )
    return train_rows, validation_rows, n_test


def assert_inventory_matches_split(inventory_rows: Sequence[Mapping[str, str]], split_rows: Sequence[Mapping[str, str]]) -> None:
    included = {row["image_id"]: row for row in included_rows(inventory_rows)}
    manifest = {row["image_id"]: row for row in split_rows}
    if set(included) != set(manifest):
        raise ProjectError(
            "The included inventory rows and the split manifest do not list the same image_id values. "
            "Re-run the split after any change to include, paths, or labels."
        )
    for image_id, row in included.items():
        other = manifest[image_id]
        if other["label"] != row["label"] or other["sha256"] != row["sha256"] or other["relative_path"] != row["relative_path"]:
            raise ProjectError(
                f"Inventory and split disagree for {row['relative_path']}. "
                "Re-run the split. Training will not ignore a changed label or checksum."
            )
    assert_groups_fit_split(inventory_rows, split_rows)


def build_callbacks(checkpoint_path: Path, patience: int):
    import keras

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(patience),
        restore_best_weights=True,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    return [early, checkpoint]


def make_sequence(
    rows: Sequence[Mapping[str, str]],
    data_root: Path,
    image_size: int,
    batch_size: int,
    *,
    training: bool,
    augment_config: Mapping[str, Any],
    seed: int,
    class_weights: Sequence[float] | None,
):
    """A new sequence. Training rows are augmented; validation rows are not."""
    import keras

    from .data import augment_training_image, prepare_image, resolve_image_path

    label_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    class ExampleSequence(keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.rows = [dict(row) for row in rows]
            self.epoch = 0
            if training:
                random.Random(seed).shuffle(self.rows)

        def __len__(self) -> int:
            return max(1, math.ceil(len(self.rows) / batch_size))

        def on_epoch_end(self) -> None:
            self.epoch += 1
            if training:
                random.Random(seed + self.epoch).shuffle(self.rows)

        def __getitem__(self, item: int):
            batch = self.rows[item * batch_size : (item + 1) * batch_size]
            if not batch:
                raise ProjectError("An empty training batch was requested.")
            images = []
            labels = []
            for offset, row in enumerate(batch):
                path = resolve_image_path(data_root, row["relative_path"])
                if training:
                    rng = np.random.default_rng(seed + self.epoch * 100003 + item * 1009 + offset)

                    def augment(image, rng=rng, config=augment_config):
                        return augment_training_image(image, dict(config), rng)

                    array = prepare_image(path, image_size, training=True, augment_fn=augment)
                else:
                    array = prepare_image(path, image_size, training=False, augment_fn=None)
                images.append(array)
                labels.append(label_index[row["label"]])
            features = np.stack(images).astype(np.float32)
            targets = np.asarray(labels, dtype=np.int32)
            if training and class_weights is not None:
                sample_weight = np.asarray([class_weights[int(label)] for label in targets], dtype=np.float32)
                return features, targets, sample_weight
            return features, targets

    return ExampleSequence()


def _history_payload(history) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def _best_val_loss(history) -> float:
    values = history.history.get("val_loss") or []
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        raise ProjectError(
            "No finite val_loss was recorded. Early stopping and model selection need validation loss. "
            "Test data was not used as a replacement."
        )
    return min(finite)


def run_training(
    *,
    config: Path | Mapping[str, Any],
    data_root: Path,
    inventory_path: Path,
    split_path: Path,
    output_dir: Path,
    confirm_review: bool,
    overwrite: bool = False,
    verbose: int = 1,
) -> dict[str, Any]:
    resolved = load_training_config(config)
    inventory = load_inventory(inventory_path)
    assert_reviewed_for_training(inventory, confirm_review=confirm_review)
    assert_files_unchanged(inventory, data_root)
    manifest = load_split_manifest(split_path)
    assert_inventory_matches_split(inventory, manifest)
    train_rows, validation_rows, n_test = rows_for_training(manifest)
    output = ensure_fresh_directory(Path(output_dir), overwrite=overwrite)
    seed = int(resolved["seed"])
    set_global_seed(seed)
    image_size = int(resolved["image_size"])
    batch_size = max(1, min(int(resolved["batch_size"]), len(train_rows)))
    class_weight_list = balanced_class_weight_list([row["label"] for row in train_rows])
    objective = training_objective(
        resolved["training"].get("loss", "sparse_categorical_crossentropy"),
        bool(resolved["training"].get("use_class_weight", True)),
        class_weight_list,
    )
    if objective["sequence_class_weights"] is not None and objective["focal_class_weights"] is not None:
        raise ProjectError("Class weights would be applied twice. Refusing to train.")
    weights = resolved["model"].get("weights", "imagenet")
    model, backbone = build_model(
        image_size=image_size,
        dropout=float(resolved["model"].get("dropout", 0.2)),
        weights=weights,
        fine_tune_from=str(resolved["model"].get("fine_tune_from", "block6")),
    )
    augment_config = dict(resolved["augmentation"])
    train_sequence = make_sequence(
        train_rows,
        data_root,
        image_size,
        batch_size,
        training=True,
        augment_config=augment_config,
        seed=seed,
        class_weights=objective["sequence_class_weights"],
    )
    validation_sequence = make_sequence(
        validation_rows,
        data_root,
        image_size,
        batch_size,
        training=False,
        augment_config=augment_config,
        seed=seed,
        class_weights=None,
    )
    patience = int(resolved["training"].get("early_stopping_patience", 3))
    phase1_path = output / "phase1_best.keras"
    phase2_path = output / "phase2_best.keras"
    phase1_callbacks = build_callbacks(phase1_path, patience)
    compile_model(
        model,
        learning_rate=float(resolved["training"].get("learning_rate_phase1", 1e-3)),
        loss_name=objective["loss_name"],
        focal_gamma=float(resolved["training"].get("focal_gamma", 2.0)),
        focal_class_weights=objective["focal_class_weights"],
    )
    history1 = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=int(resolved["training"].get("phase1_epochs", 10)),
        callbacks=phase1_callbacks,
        verbose=verbose,
    )
    model.save(phase1_path)
    configure_backbone(backbone, phase="finetune", fine_tune_from=str(resolved["model"].get("fine_tune_from", "block6")))
    compile_model(
        model,
        learning_rate=float(resolved["training"].get("learning_rate_phase2", 1e-5)),
        loss_name=objective["loss_name"],
        focal_gamma=float(resolved["training"].get("focal_gamma", 2.0)),
        focal_class_weights=objective["focal_class_weights"],
    )
    # New sequence and callback objects. Keras callback state must not carry over.
    train_sequence_phase2 = make_sequence(
        train_rows,
        data_root,
        image_size,
        batch_size,
        training=True,
        augment_config=augment_config,
        seed=seed + 1,
        class_weights=objective["sequence_class_weights"],
    )
    validation_sequence_phase2 = make_sequence(
        validation_rows,
        data_root,
        image_size,
        batch_size,
        training=False,
        augment_config=augment_config,
        seed=seed,
        class_weights=None,
    )
    phase2_callbacks = build_callbacks(phase2_path, patience)
    if any(left is right for left, right in zip(phase1_callbacks, phase2_callbacks)):
        raise ProjectError("Internal error: callback objects were reused across phases.")
    history2 = model.fit(
        train_sequence_phase2,
        validation_data=validation_sequence_phase2,
        epochs=int(resolved["training"].get("phase2_epochs", 10)),
        callbacks=phase2_callbacks,
        verbose=verbose,
    )
    model.save(phase2_path)
    best1 = _best_val_loss(history1)
    best2 = _best_val_loss(history2)
    # Tie goes to the frozen-base phase. Selection never looks at the test split.
    selected = "phase1" if best1 <= best2 else "phase2"
    selected_val = best1 if selected == "phase1" else best2
    import keras

    from .model import get_custom_objects

    selected_source = phase1_path if selected == "phase1" else phase2_path
    selected_model = keras.models.load_model(selected_source, custom_objects=get_custom_objects(), compile=False)
    model_path = output / "model.keras"
    selected_model.save(model_path)
    policy = resolved["decision"].get("threshold_policy", "predefined")
    if policy == "predefined":
        threshold = float(resolved["decision"]["malignant_threshold"])
        locked = True
    else:
        threshold = None
        locked = False
    history1_payload = _history_payload(history1)
    history2_payload = _history_payload(history2)
    write_json(output / "phase1_history.json", history1_payload)
    write_json(output / "phase2_history.json", history2_payload)
    write_yaml(output / "config.yaml", resolved)
    dataset_fingerprint = sha256_json(
        [
            {"relative_path": row["relative_path"], "sha256": row["sha256"], "label": row["label"]}
            for row in sorted(included_rows(inventory), key=lambda row: row["relative_path"])
        ]
    )
    split_fingerprint = sha256_json(
        [
            {"image_id": row["image_id"], "split": row["split"], "group_id": row["group_id"], "sha256": row["sha256"]}
            for row in sorted(manifest, key=lambda row: row["image_id"])
        ]
    )
    metadata = {
        "project": "open-breast-ultrasound",
        "seed": seed,
        "image_size": image_size,
        "class_names": list(CLASS_NAMES),
        "malignant_index": CLASS_NAMES.index("malignant"),
        "config": resolved,
        "library_versions": library_versions(),
        "device": device_info(),
        "git_commit": git_commit(Path.cwd()),
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "phases": {
            "phase1": {
                "best_val_loss": best1,
                "epochs_ran": len(history1_payload.get("loss", [])),
                "trainable": "classification head only; backbone frozen; BatchNormalization not trainable",
            },
            "phase2": {
                "best_val_loss": best2,
                "epochs_ran": len(history2_payload.get("loss", [])),
                "trainable": f"backbone from {resolved['model'].get('fine_tune_from', 'block6')}; BatchNormalization not trainable",
            },
        },
        "selected_phase": selected,
        "selected_val_loss": selected_val,
        "selection_metric": "minimum validation loss across phases; equal loss keeps phase1; test data not used",
        "model_sha256": sha256_file(model_path),
        "loss_name": objective["loss_name"],
        "class_weights": {name: class_weight_list[index] for index, name in enumerate(CLASS_NAMES)},
        "class_weight_applied_in": objective["where"],
        "decision_threshold": threshold,
        "threshold_policy": policy,
        "threshold_locked": locked,
        "threshold_clinically_approved": False,
        "temperature": None,
        "temperature_fit_on": None,
        "n_train_images": len(train_rows),
        "n_validation_images": len(validation_rows),
        "n_test_images_held_out": n_test,
        "preprocessing": {
            "resize": "bilinear",
            "color": "RGB",
            "exif_transpose": True,
            "pixel_range": [0, 255],
            "divided_by_255_before_model": False,
            "efficientnet_preprocess_input_called": False,
            "augmentation": "training batches only",
        },
        "batch_normalization": "frozen and non-trainable in both phases",
        "review_disclaimer": REVIEW_NOT_MEDICAL,
        "patient_level_independence": False,
        "clinical_use": False,
    }
    write_json(output / "experiment.json", metadata)
    return {
        "output_dir": str(output),
        "selected_phase": selected,
        "selected_val_loss": selected_val,
        "n_test_images_held_out": n_test,
        "model_sha256": metadata["model_sha256"],
    }
