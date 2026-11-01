"""Small TensorFlow tests. They do not download ImageNet weights or BUSI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _reviewed_inventory(root: Path, audit_dir: Path) -> None:
    from PIL import Image

    from open_breast_ultrasound.audit import load_inventory, run_audit
    from open_breast_ultrasound.utils import CLASS_NAMES, write_csv
    from open_breast_ultrasound.audit import INVENTORY_FIELDS

    for label_index, label in enumerate(CLASS_NAMES):
        for index in range(3):
            path = root / label / f"{label}_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.zeros((32, 32, 3), dtype=np.uint8)
            array[:, :, label_index] = 40 + index * 30
            array[index * 4 : index * 4 + 6, :, :] = 255
            Image.fromarray(array).save(path)
    run_audit(root, audit_dir)
    rows = load_inventory(audit_dir / "inventory.csv")
    for row in rows:
        if row["include"] == "true":
            row["reviewed"] = "true"
            row["review_note"] = "synthetic software fixture, not a medical review"
    write_csv(audit_dir / "inventory.csv", rows, INVENTORY_FIELDS)


def test_backbone_contract_batch_norm_and_minimal_train(tmp_path: Path) -> None:
    import keras

    from open_breast_ultrasound.model import (
        assert_efficientnet_input_contract,
        build_focal_loss,
        build_model,
        compile_model,
        configure_backbone,
        focal_per_example_numpy,
        get_custom_objects,
    )
    from open_breast_ultrasound.splits import run_split
    from open_breast_ultrasound.train import run_training

    assert_efficientnet_input_contract()
    model, backbone = build_model(image_size=32, weights=None, dropout=0.1, fine_tune_from="block6")
    assert any(isinstance(layer, keras.layers.Rescaling) for layer in backbone.layers)
    batch_norms = [layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)]
    before = [layer.moving_mean.numpy().copy() for layer in batch_norms]
    configure_backbone(backbone, phase="finetune", fine_tune_from="block6")
    assert all(not layer.trainable for layer in batch_norms)
    assert backbone.get_layer("stem_conv").trainable is False
    assert backbone.get_layer("block6a_expand_conv").trainable is True
    compile_model(
        model,
        learning_rate=1e-3,
        loss_name="sparse_categorical_crossentropy",
        focal_gamma=2.0,
        focal_class_weights=None,
    )
    features = np.random.default_rng(0).uniform(0, 255, size=(4, 32, 32, 3)).astype(np.float32)
    targets = np.array([0, 1, 2, 1], dtype=np.int32)
    model.fit(features, targets, epochs=1, batch_size=2, verbose=0, validation_data=(features[:2], targets[:2]))
    after = [layer.moving_mean.numpy().copy() for layer in batch_norms]
    assert all(np.allclose(left, right) for left, right in zip(before, after))

    loss_cls = build_focal_loss(2.0, [1.0, 2.0, 1.0])
    loss = loss_cls(reduction="none")
    y_true = np.array([0, 1, 2], dtype=np.int32)
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]], dtype=np.float32)
    per_example = np.asarray(loss(y_true, y_pred))
    assert per_example.shape == (3,)
    expected = focal_per_example_numpy(y_true, y_pred, 2.0, [1.0, 2.0, 1.0])
    assert np.allclose(per_example, expected, atol=1e-5)
    config = loss.get_config()
    json.dumps(config)
    restored = loss_cls.from_config(config)
    assert restored.get_config()["gamma"] == 2.0

    data = tmp_path / "data"
    audit_dir = tmp_path / "audit"
    _reviewed_inventory(data, audit_dir)
    run_split(audit_dir / "inventory.csv", tmp_path / "split", seed=4)
    training_config = {
        "seed": 4,
        "image_size": 32,
        "batch_size": 2,
        "classes": ["benign", "malignant", "normal"],
        "model": {"backbone": "EfficientNetB0", "weights": None, "dropout": 0.0, "fine_tune_from": "block6"},
        "training": {
            "phase1_epochs": 1,
            "phase2_epochs": 1,
            "early_stopping_patience": 1,
            "learning_rate_phase1": 1e-3,
            "learning_rate_phase2": 1e-4,
            "loss": "sparse_categorical_crossentropy",
            "focal_gamma": 2.0,
            "use_class_weight": True,
        },
        "augmentation": {"horizontal_flip": True, "rotation_degrees": 5, "contrast": 0.05},
        "decision": {"malignant_threshold": 0.5, "threshold_policy": "predefined"},
    }
    result = run_training(
        config=training_config,
        data_root=data,
        inventory_path=audit_dir / "inventory.csv",
        split_path=tmp_path / "split" / "split_manifest.csv",
        output_dir=tmp_path / "experiment",
        confirm_review=True,
        verbose=0,
    )
    assert result["n_test_images_held_out"] > 0
    metadata = json.loads((tmp_path / "experiment" / "experiment.json").read_text(encoding="utf-8"))
    assert metadata["preprocessing"]["divided_by_255_before_model"] is False
    assert metadata["threshold_clinically_approved"] is False
    assert metadata["patient_level_independence"] is False
    assert metadata["n_test_images_held_out"] == result["n_test_images_held_out"]
    loaded = keras.models.load_model(tmp_path / "experiment" / "model.keras", compile=False)
    sample = np.zeros((1, 32, 32, 3), dtype=np.float32)
    probabilities = np.asarray(loaded.predict(sample, verbose=0))
    assert probabilities.shape == (1, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    from open_breast_ultrasound.calibration import run_calibration
    from open_breast_ultrasound.evaluate import run_evaluation
    from open_breast_ultrasound.inference import load_runtime, predict_image

    calibration = run_calibration(
        tmp_path / "experiment",
        data,
        audit_dir / "inventory.csv",
        tmp_path / "split" / "split_manifest.csv",
    )
    assert calibration["n_test_images_not_used"] == result["n_test_images_held_out"]
    evaluation = run_evaluation(
        tmp_path / "experiment",
        data,
        audit_dir / "inventory.csv",
        tmp_path / "split" / "split_manifest.csv",
        n_bootstrap=20,
        bootstrap_seed=1,
    )
    report = json.loads((Path(evaluation["output_dir"]) / "report.json").read_text(encoding="utf-8"))
    assert report["source"] == "internal_held_out_split"
    assert report["external_validation"] is False
    assert report["normal_examples_invented"] is False
    image = next((data / "benign").glob("*.png"))
    runtime = load_runtime(tmp_path / "experiment")
    scores = predict_image(runtime, image)
    assert set(scores) == {"benign", "malignant", "normal"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
    # Custom-object registry can rebuild the focal class used by a saved loss.
    assert "KerasFocalLoss" in get_custom_objects()
