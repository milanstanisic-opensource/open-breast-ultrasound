"""Software tests that do not need TensorFlow, a GPU, or the BUSI files."""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFile

from open_breast_ultrasound.audit import (
    assert_files_unchanged,
    assert_reviewed_for_training,
    load_inventory,
    run_audit,
)
from open_breast_ultrasound.calibration import apply_temperature, fit_temperature
from open_breast_ultrasound.cli import build_parser
from open_breast_ultrasound.data import image_to_model_array, load_rgb, prepare_image
from open_breast_ultrasound.demo import DISCLAIMER, build_interface, launch_demo
from open_breast_ultrasound.evaluate import (
    attach_threshold,
    auroc,
    average_precision,
    binary_brier,
    evaluate_predictions,
    multiclass_brier,
    negative_log_likelihood,
    prediction_table,
    select_threshold_on_validation,
)
from open_breast_ultrasound.inference import preprocess_image
from open_breast_ultrasound.model import focal_per_example_numpy, training_objective
from open_breast_ultrasound.splits import assign_groups, build_groups, load_split_manifest, run_split
from open_breast_ultrasound.train import assert_inventory_matches_split, rows_for_training
from open_breast_ultrasound.utils import CLASS_NAMES, ProjectError


def _solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _color(label: str, index: int) -> tuple[int, int, int]:
    base = {"benign": 30, "malignant": 100, "normal": 170}[label]
    return (base, 20 + index * 25, 240 - index * 20)


def _dataset(root: Path) -> None:
    for label in CLASS_NAMES:
        for index in range(5):
            _solid(root / label / f"{label} ({index}).png", _color(label, index))
    original = root / "benign" / "benign (0).png"
    (root / "benign" / "benign (0)_copy.png").write_bytes(original.read_bytes())
    Image.new("RGB", (32, 32), _color("benign", 0)).save(root / "benign" / "same_pixels.bmp")
    _solid(root / "benign" / "benign (0)_mask.png", (255, 255, 255))
    _solid(root / "benign" / "benign (0)_mask_1.png", (0, 0, 0))
    (root / "normal" / "broken.png").write_bytes(b"this is not an image")
    truncated = root / "malignant" / "truncated.png"
    _solid(truncated, (1, 2, 3))
    truncated.write_bytes(truncated.read_bytes()[:24])
    (root / "benign" / "notes.txt").write_text("ignore me\n", encoding="utf-8")


def _audit(tmp_path: Path, **kwargs):
    data = tmp_path / "Dataset_BUSI_with_GT"
    _dataset(data)
    output = tmp_path / "audit"
    summary = run_audit(data, output, **kwargs)
    return data, output, summary


def test_masks_unreadable_and_filename_numbers(tmp_path: Path) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data, output, _summary = _audit(tmp_path)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    rows = load_inventory(output / "inventory.csv")
    paths = [row["relative_path"] for row in rows]
    assert not any("mask" in path.lower() for path in paths)
    summary = __import__("json").loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["n_masks_skipped"] == 2
    assert summary["filename_numbers_used_as_patient_id"] is False
    included = [row for row in rows if row["include"] == "true"]
    assert all(row["patient_id"] == "" and row["case_id"] == "" for row in included)
    benign_named = next(row for row in included if row["relative_path"] == "benign/benign (0).png")
    assert benign_named["patient_id"] == ""
    excluded = [row for row in rows if row["include"] == "false"]
    assert {Path(row["relative_path"]).name for row in excluded} == {"broken.png", "truncated.png"}
    assert all(row["exclusion_reason"] for row in excluded)
    by_path = {row["relative_path"]: row for row in included}
    assert by_path["benign/benign (0).png"]["duplicate_group"]
    assert by_path["benign/benign (0).png"]["duplicate_group"] == by_path["benign/benign (0)_copy.png"]["duplicate_group"]
    assert by_path["benign/benign (0).png"]["duplicate_group"] == by_path["benign/same_pixels.bmp"]["duplicate_group"]
    assert by_path["benign/benign (0).png"]["sha256"] == by_path["benign/benign (0)_copy.png"]["sha256"]
    assert by_path["benign/benign (0).png"]["sha256"] != by_path["benign/same_pixels.bmp"]["sha256"]
    assert by_path["benign/benign (0).png"]["pixel_sha256"] == by_path["benign/same_pixels.bmp"]["pixel_sha256"]
    with (output / "duplicate_candidates.csv").open(encoding="utf-8") as handle:
        perceptual = [row for row in csv.DictReader(handle) if row["method"] == "perceptual_dhash"]
    assert perceptual
    assert all(row["auto_grouped"] == "false" for row in perceptual)
    assert summary["patient_id_confirmed_by_this_software"] is False
    assert (data / "benign" / "benign (0).png").is_file()
    assert (data / "benign" / "benign (0)_mask.png").is_file()


def test_id_map_is_not_inferred_from_names(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _dataset(data)
    id_map = tmp_path / "ids.csv"
    id_map.write_text(
        "relative_path,patient_id,case_id\nnormal/normal (0).png,P-1,C-1\nnormal/normal (1).png,P-1,C-1\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    run_audit(data, output, id_map=id_map)
    rows = {row["relative_path"]: row for row in load_inventory(output / "inventory.csv")}
    assert rows["normal/normal (0).png"]["patient_id"] == "P-1"
    assert rows["normal/normal (1).png"]["case_id"] == "C-1"
    assert rows["benign/benign (4).png"]["patient_id"] == ""


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (30, 10), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(path, format="JPEG", exif=exif.tobytes(), quality=95)
    decoded = load_rgb(path)
    assert decoded.size == (10, 30)
    array = image_to_model_array(path, 32)
    assert array.shape == (32, 32, 3)
    assert array.dtype == np.float32
    assert float(array.max()) <= 255.0
    assert float(array.max()) > 1.0


def test_preprocessing_is_shared_and_not_rescaled(tmp_path: Path) -> None:
    path = tmp_path / "white.png"
    _solid(path, (255, 255, 255), size=(16, 16))

    def fail(_image):
        raise AssertionError("augmentation ran")

    first = prepare_image(path, 32, training=False, augment_fn=fail)
    second = preprocess_image(path, 32)
    direct = image_to_model_array(path, 32)
    assert np.array_equal(first, second)
    assert np.array_equal(second, direct)
    assert float(first.max()) == 255.0
    flipped = prepare_image(path, 32, training=True, augment_fn=lambda image: np.zeros_like(image))
    assert float(flipped.max()) == 0.0


def test_split_groups_links_and_is_stable(tmp_path: Path) -> None:
    data, _output, _summary = _audit(tmp_path)
    id_map = tmp_path / "ids.csv"
    id_map.write_text(
        "relative_path,patient_id,case_id\nnormal/normal (0).png,P-9,\nnormal/normal (1).png,P-9,\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit-ids"
    run_audit(data, output, id_map=id_map)
    inventory = load_inventory(output / "inventory.csv")
    groups = build_groups(inventory)
    linked = False
    for group in groups:
        member_paths = [member["relative_path"] for member in group["members"]]
        if "normal/normal (0).png" in member_paths:
            assert "normal/normal (1).png" in member_paths
            linked = True
    assert linked
    first = run_split(output / "inventory.csv", tmp_path / "split-a", seed=7)
    run_split(output / "inventory.csv", tmp_path / "split-b", seed=7)
    assert (tmp_path / "split-a" / "split_manifest.csv").read_text(encoding="utf-8") == (
        tmp_path / "split-b" / "split_manifest.csv"
    ).read_text(encoding="utf-8")
    with (tmp_path / "split-a" / "split_manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    groups_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    labels_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    split_of = {}
    for row in manifest:
        by_split[row["split"]].add(row["image_id"])
        groups_by_split[row["split"]].add(row["group_id"])
        labels_by_split[row["split"]].add(row["label"])
        split_of[row["relative_path"]] = row["split"]
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["validation"])
    for split, labels in labels_by_split.items():
        assert labels == set(CLASS_NAMES), split
    assert split_of["normal/normal (0).png"] == split_of["normal/normal (1).png"]
    assert split_of["benign/benign (0).png"] == split_of["benign/benign (0)_copy.png"]
    assert split_of["benign/benign (0).png"] == split_of["benign/same_pixels.bmp"]
    assert first["summary"]["patient_level_independence"] is False
    assert first["summary"]["seed"] == 7
    assert "mask" not in (tmp_path / "split-a" / "split_manifest.csv").read_text(encoding="utf-8")


def test_split_refuses_too_few_groups() -> None:
    groups = []
    for label in CLASS_NAMES:
        groups.append(
            {
                "group_id": f"only-{label}",
                "member_ids": [label],
                "members": [{"image_id": label, "label": label}],
                "labels": [label],
                "stratum": label,
                "mixed": False,
                "n": 1,
            }
        )
    with pytest.raises(ProjectError, match="At least 3 groups"):
        assign_groups(groups, ratios={"train": 0.6, "validation": 0.2, "test": 0.2}, seed=1)


def test_review_gate_and_changed_file(tmp_path: Path) -> None:
    data, output, _summary = _audit(tmp_path)
    rows = load_inventory(output / "inventory.csv")
    with pytest.raises(ProjectError, match="confirm-review"):
        assert_reviewed_for_training(rows, confirm_review=False)
    with pytest.raises(ProjectError, match="reviewed=false"):
        assert_reviewed_for_training(rows, confirm_review=True)
    target = data / "benign" / "benign (1).png"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(ProjectError, match="changed after the inventory"):
        assert_files_unchanged(rows, data)


def test_split_disagrees_after_group_edit(tmp_path: Path) -> None:
    _data, output, _summary = _audit(tmp_path)
    run_split(output / "inventory.csv", tmp_path / "split", seed=3)
    inventory = load_inventory(output / "inventory.csv")
    manifest = load_split_manifest(tmp_path / "split" / "split_manifest.csv")
    included = [row for row in inventory if row["include"] == "true"]
    for row in included:
        row["reviewed"] = "true"
    train_paths = [row["relative_path"] for row in manifest if row["split"] == "train"]
    test_paths = [row["relative_path"] for row in manifest if row["split"] == "test"]
    by_path = {row["relative_path"]: row for row in included}
    by_path[train_paths[0]]["duplicate_group"] = "manual-link"
    by_path[test_paths[0]]["duplicate_group"] = "manual-link"
    with pytest.raises(ProjectError, match="spans more than one split"):
        assert_inventory_matches_split(included, manifest)


def test_rows_for_training_hold_out_test() -> None:
    rows = [
        {"split": "train", "image_id": "a"},
        {"split": "validation", "image_id": "b"},
        {"split": "test", "image_id": "c"},
    ]
    train_rows, validation_rows, n_test = rows_for_training(rows)
    assert [row["image_id"] for row in train_rows] == ["a"]
    assert [row["image_id"] for row in validation_rows] == ["b"]
    assert n_test == 1


def test_metrics_null_when_class_missing_and_known_values() -> None:
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert auroc([1, 0], [0.1, 0.9]) == pytest.approx(0.0)
    assert auroc([1, 1], [0.4, 0.6]) is None
    assert average_precision([1, 1, 0], [0.9, 0.8, 0.1]) == pytest.approx(1.0)
    assert average_precision([0, 0], [0.2, 0.3]) is None
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert multiclass_brier(probs, [0, 1]) == pytest.approx(0.0)
    assert binary_brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert negative_log_likelihood(probs, [0, 1]) == pytest.approx(0.0)
    records = [
        {"image_id": "a", "relative_path": "a.png", "group_id": "g1", "label": "benign", "split": "test"},
        {"image_id": "b", "relative_path": "b.png", "group_id": "g2", "label": "normal", "split": "test"},
    ]
    report = evaluate_predictions(
        attach_threshold(records, 0.5),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]),
        threshold=0.5,
        temperature=1.0,
        n_bootstrap=20,
        bootstrap_seed=1,
        split_name="test",
        source="internal_held_out_split",
    )
    binary = report["malignant_binary"]
    assert binary["sensitivity"] is None
    assert binary["auroc"] is None
    assert binary["average_precision"] is None
    assert binary["specificity"] == pytest.approx(1.0)
    assert 0 not in {binary["sensitivity"], binary["auroc"], binary["average_precision"]}
    names = {item["metric"] for item in report["unavailable"]}
    assert "malignant_sensitivity" in names
    assert "malignant_one_vs_rest_auroc" in names
    assert report["normal_examples_invented"] is False
    assert report["bootstrap"]["malignant_sensitivity"]["ci_low"] is None
    external_records = [
        {"image_id": "e1", "relative_path": "b.png", "group_id": "p1", "label": "benign", "split": "external"},
        {"image_id": "e2", "relative_path": "m.png", "group_id": "p2", "label": "malignant", "split": "external"},
    ]
    external = evaluate_predictions(
        external_records,
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        threshold=0.5,
        temperature=1.0,
        n_bootstrap=0,
        split_name="external",
        source="external_manifest",
    )
    assert external["classes_absent"] == ["normal"]
    assert external["normal_examples_invented"] is False
    assert external["n_images"] == 2
    assert external["class_counts"]["normal"] == 0
    assert external["multiclass_argmax"]["per_class"]["normal"]["recall"] is None
    table = prediction_table(
        external_records,
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        threshold=0.5,
        split_name="external",
    )
    assert [row["binary_prediction"] for row in table] == ["not_malignant", "malignant"]
    assert [row["argmax_label"] for row in table] == ["benign", "malignant"]


def test_temperature_and_threshold_are_validation_only() -> None:
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    same = apply_temperature(probs, 1.0)
    assert np.allclose(same, probs)
    softened = apply_temperature(probs, 2.0)
    assert softened[0, 0] < probs[0, 0]
    assert np.allclose(softened.sum(axis=1), 1.0)
    fitted = fit_temperature(probs, [0, 1, 2])
    assert fitted["test_used"] is False
    assert fitted["fit_on"] == "validation"
    assert math.isfinite(fitted["temperature"])
    assert fitted["temperature"] > 0
    chosen = select_threshold_on_validation(["malignant", "benign", "benign"], [0.9, 0.2, 0.4])
    assert chosen["fit_on"] == "validation"
    assert chosen["clinically_approved"] is False
    assert 0.0 <= chosen["threshold"] <= 1.0
    assert "test" not in inspect.signature(select_threshold_on_validation).parameters


def test_focal_weights_are_not_applied_twice() -> None:
    labels = np.array([0, 1])
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    loss = focal_per_example_numpy(labels, probs, gamma=2.0, class_weights=[1.0, 2.0, 1.0])
    assert loss.shape == (2,)
    manual = -((1 - 0.7) ** 2) * math.log(0.7)
    assert loss[0] == pytest.approx(manual)
    assert loss[1] == pytest.approx(2 * (-((1 - 0.6) ** 2) * math.log(0.6)))
    cross_entropy = training_objective("sparse_categorical_crossentropy", True, [1.0, 1.0, 1.0])
    focal = training_objective("focal", True, [1.0, 2.0, 1.0])
    assert cross_entropy["sequence_class_weights"] == [1.0, 1.0, 1.0]
    assert cross_entropy["focal_class_weights"] is None
    assert focal["sequence_class_weights"] is None
    assert focal["focal_class_weights"] == [1.0, 2.0, 1.0]
    assert CLASS_NAMES.index("malignant") == 1


def test_demo_copy_and_cli_commands() -> None:
    lowered = DISCLAIMER.lower()
    assert "not a diagnosis" in lowered
    assert "model scores" in lowered
    assert "99%" not in DISCLAIMER
    assert "cancer detected" not in lowered
    assert "does not promise zero storage" in lowered
    assert "early detection" in lowered
    assert inspect.signature(launch_demo).parameters["allow_public_share"].default is False
    parser = build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    expected = {
        "audit",
        "check-review",
        "split",
        "train",
        "calibrate",
        "evaluate",
        "evaluate-external",
        "predict",
        "demo",
    }
    assert expected == commands


def test_demo_interface_builds() -> None:
    from types import SimpleNamespace

    runtime = SimpleNamespace(
        class_names=("benign", "malignant", "normal"),
        temperature=1.0,
        threshold=0.5,
        image_size=32,
    )
    interface = build_interface(runtime)
    assert interface.title == "Open Breast Ultrasound — Research Prototype"


def test_project_source_does_not_accept_truncated_images() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "LOAD_TRUNCATED_IMAGES = True" not in text
    assert "LOAD_TRUNCATED_IMAGES=True" not in text
