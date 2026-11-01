"""Group-aware train/validation/test splits.

Groups are the leakage boundary this software can actually see: a supplied
patient_id, a supplied case_id, and exact file or pixel duplicates. Without
confirmed patient IDs this is not patient-level separation.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import REVIEW_NOT_MEDICAL, included_rows, load_inventory
from .utils import CLASS_NAMES, ProjectError, parse_bool, sha256_json, write_csv, write_json

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FIELDS = (
    "image_id",
    "relative_path",
    "label",
    "group_id",
    "split",
    "stratum",
    "mixed_label_group",
    "sha256",
)

PATIENT_LEVEL_WARNING = (
    "This split does not establish patient-level independence. "
    "Public BUSI filenames were not treated as patient IDs. "
    "Images stay together only when they share a supplied patient_id, "
    "a supplied case_id, or an exact duplicate group."
)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_groups(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Connected components of included rows."""
    included = list(included_rows(rows))
    if not included:
        raise ProjectError("Cannot build groups because no rows are included.")
    finder = _UnionFind()
    for row in included:
        finder.add(row["image_id"])
    for field in ("patient_id", "case_id", "duplicate_group"):
        buckets: dict[str, list[str]] = defaultdict(list)
        for row in included:
            value = (row.get(field) or "").strip()
            if value:
                buckets[value].append(row["image_id"])
        for members in buckets.values():
            for other in members[1:]:
                finder.union(members[0], other)
    components: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_id = {row["image_id"]: row for row in included}
    for row in included:
        components[finder.find(row["image_id"])].append(row)
    groups: list[dict[str, Any]] = []
    for members in components.values():
        ordered = sorted(members, key=lambda row: row["image_id"])
        labels = [row["label"] for row in ordered]
        counts = Counter(labels)
        top = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == top)
        member_ids = [row["image_id"] for row in ordered]
        import hashlib

        digest = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:12]
        groups.append(
            {
                "group_id": f"grp_{digest}",
                "member_ids": member_ids,
                "members": ordered,
                "labels": labels,
                "stratum": winners[0],
                "mixed": len(set(labels)) > 1,
                "n": len(ordered),
            }
        )
    del by_id
    return sorted(groups, key=lambda group: group["group_id"])


def assign_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to train/validation/test.

    One group of each stratum is placed in validation and one in test first,
    so a class is not dropped to make the ratio look better. Remaining groups
    fill the largest per-class deficit. The seed is not chosen by model score.
    """
    _validate_ratios(ratios)
    by_class: dict[str, list[Mapping[str, Any]]] = {name: [] for name in CLASS_NAMES}
    for group in groups:
        stratum = str(group["stratum"])
        if stratum not in by_class:
            raise ProjectError(f"Unknown stratum {stratum!r}.")
        by_class[stratum].append(group)
    for name, items in by_class.items():
        if len(items) < 3:
            raise ProjectError(
                f"Class '{name}' has {len(items)} group(s) after linking patient_id, "
                "case_id, and exact duplicates. At least 3 groups per class are required "
                "so train, validation, and test can each contain that class without "
                "breaking a group. Refusing to split a group or to drop a class."
            )
    rng = random.Random(seed)
    for name in CLASS_NAMES:
        rng.shuffle(by_class[name])

    class_totals = {
        name: sum(int(group["n"]) for group in items) for name, items in by_class.items()
    }
    class_targets = {
        split: {name: float(ratios[split]) * class_totals[name] for name in CLASS_NAMES}
        for split in SPLIT_NAMES
    }
    class_counts = {split: {name: 0 for name in CLASS_NAMES} for split in SPLIT_NAMES}
    global_counts = {split: 0 for split in SPLIT_NAMES}
    total_images = sum(class_totals.values())
    global_targets = {split: float(ratios[split]) * total_images for split in SPLIT_NAMES}
    assignment: dict[str, str] = {}

    remaining: list[Mapping[str, Any]] = []
    for name in CLASS_NAMES:
        items = by_class[name]
        for group, split in ((items[0], "validation"), (items[1], "test")):
            _place(assignment, class_counts, global_counts, group, split)
        remaining.extend(items[2:])

    rng.shuffle(remaining)
    remaining.sort(key=lambda group: (-int(group["n"]), str(group["group_id"])))
    for group in remaining:
        stratum = str(group["stratum"])

        def sort_key(split: str, stratum: str = stratum) -> tuple[float, float, int]:
            class_deficit = class_targets[split][stratum] - class_counts[split][stratum]
            global_deficit = global_targets[split] - global_counts[split]
            tie = {"train": 2, "validation": 1, "test": 0}[split]
            return (class_deficit, global_deficit, tie)

        split = max(SPLIT_NAMES, key=sort_key)
        _place(assignment, class_counts, global_counts, group, split)

    _assert_assignment(groups, assignment)
    return assignment


def _place(
    assignment: dict[str, str],
    class_counts: dict[str, dict[str, int]],
    global_counts: dict[str, int],
    group: Mapping[str, Any],
    split: str,
) -> None:
    assignment[str(group["group_id"])] = split
    class_counts[split][str(group["stratum"])] += int(group["n"])
    global_counts[split] += int(group["n"])


def _assert_assignment(groups: Sequence[Mapping[str, Any]], assignment: dict[str, str]) -> None:
    seen = set(assignment)
    expected = {str(group["group_id"]) for group in groups}
    if seen != expected:
        raise ProjectError("Internal split error: not every group was assigned exactly once.")
    split_of_group = assignment
    images_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    labels_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    for group in groups:
        split = split_of_group[str(group["group_id"])]
        for member in group["members"]:
            image_id = member["image_id"]
            if any(image_id in bucket for bucket in images_by_split.values()):
                raise ProjectError(f"Image {image_id} was assigned more than once.")
            images_by_split[split].add(image_id)
            labels_by_split[split].add(member["label"])
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            shared = images_by_split[left] & images_by_split[right]
            if shared:
                raise ProjectError(f"Images overlap between {left} and {right}.")
    for split in SPLIT_NAMES:
        if not images_by_split[split]:
            raise ProjectError(f"Split '{split}' is empty.")
        missing = [name for name in CLASS_NAMES if name not in labels_by_split[split]]
        if missing:
            raise ProjectError(
                f"Split '{split}' is missing class(es): {', '.join(missing)}. "
                "The seed was not changed to chase a better division. "
                "Review mixed-label duplicate groups or choose a different documented seed."
            )


def assert_groups_fit_split(
    inventory_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
) -> None:
    """Fail if current links cross the stored split."""
    groups = build_groups(inventory_rows)
    split_of = {row["image_id"]: row["split"] for row in split_rows}
    for group in groups:
        splits = {split_of.get(image_id) for image_id in group["member_ids"]}
        if None in splits or len(splits) != 1:
            raise ProjectError(
                "A linked group now spans more than one split or is missing from the manifest. "
                f"Group {group['group_id']} maps to {sorted(str(item) for item in splits)}. "
                "Re-run the split after editing patient_id, case_id, or duplicate_group. "
                "Do not keep a split that leaks known links."
            )


def run_split(
    inventory_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    train: float = 0.6,
    validation: float = 0.2,
    test: float = 0.2,
    overwrite: bool = False,
) -> dict[str, Any]:
    from .utils import ensure_fresh_directory

    rows = load_inventory(inventory_path)
    included = included_rows(rows)
    ratios = {"train": train, "validation": validation, "test": test}
    _validate_ratios(ratios)
    groups = build_groups(included)
    assignment = assign_groups(groups, ratios=ratios, seed=seed)
    manifest: list[dict[str, Any]] = []
    mixed_groups = []
    for group in groups:
        split = assignment[group["group_id"]]
        if group["mixed"]:
            mixed_groups.append(
                {
                    "group_id": group["group_id"],
                    "labels": sorted(set(group["labels"])),
                    "split": split,
                    "note": "Labels were not changed. The whole group stayed in one split.",
                }
            )
        for member in group["members"]:
            manifest.append(
                {
                    "image_id": member["image_id"],
                    "relative_path": member["relative_path"],
                    "label": member["label"],
                    "group_id": group["group_id"],
                    "split": split,
                    "stratum": group["stratum"],
                    "mixed_label_group": group["mixed"],
                    "sha256": member["sha256"],
                }
            )
    manifest.sort(key=lambda row: (row["split"], row["relative_path"]))
    counts = Counter(row["split"] for row in manifest)
    class_counts = {
        split: dict(Counter(row["label"] for row in manifest if row["split"] == split))
        for split in SPLIT_NAMES
    }
    total = len(manifest)
    achieved = {split: counts[split] / total for split in SPLIT_NAMES}
    warnings = [PATIENT_LEVEL_WARNING, REVIEW_NOT_MEDICAL]
    supplied_patients = any((row.get("patient_id") or "").strip() for row in included)
    if supplied_patients:
        warnings.append(
            "Some rows have a supplied patient_id and those IDs were kept together. "
            "This software did not verify that the IDs are real or complete."
        )
    else:
        warnings.append("patient_id is empty for the included rows. Each non-duplicate image is its own group.")
    ratio_warnings = []
    for split in SPLIT_NAMES:
        if abs(achieved[split] - ratios[split]) > 0.10:
            ratio_warnings.append(
                f"{split} contains {achieved[split]:.1%} of images; the target was {ratios[split]:.1%}. "
                "Groups were not broken to force the target ratio."
            )
    warnings.extend(ratio_warnings)
    if mixed_groups:
        warnings.append(
            "At least one linked group contains more than one label. "
            "Labels were not edited. Inspect mixed_label_groups before trusting the split."
        )
    unreviewed = [row["relative_path"] for row in included if not parse_bool(row["reviewed"])]
    if unreviewed:
        warnings.append(
            f"{len(unreviewed)} included row(s) are not reviewed. "
            "A split can be drafted first, but training will refuse these rows."
        )
    summary = {
        "seed": seed,
        "ratios_requested": ratios,
        "ratios_achieved": achieved,
        "n_images": total,
        "n_groups": len(groups),
        "n_per_split": {split: counts[split] for split in SPLIT_NAMES},
        "class_counts": class_counts,
        "n_mixed_label_groups": len(mixed_groups),
        "mixed_label_groups": mixed_groups,
        "patient_level_independence": False,
        "warnings": warnings,
        "ratio_warnings": ratio_warnings,
        "manifest_fingerprint": sha256_json(
            [
                {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "group_id": row["group_id"],
                    "label": row["label"],
                    "sha256": row["sha256"],
                }
                for row in manifest
            ]
        ),
        "selection_note": "The seed is an input. It is not searched against model metrics.",
    }
    output_dir = ensure_fresh_directory(Path(output_dir), overwrite=overwrite)
    write_csv(output_dir / "split_manifest.csv", manifest, SPLIT_FIELDS)
    write_json(output_dir / "split_summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary}


def load_split_manifest(path: Path) -> list[dict[str, str]]:
    from .utils import read_csv

    rows = read_csv(path)
    missing = [column for column in ("image_id", "relative_path", "label", "group_id", "split", "sha256") if rows and column not in rows[0]]
    if not rows:
        raise ProjectError(f"Split manifest is empty: {path}")
    if missing:
        raise ProjectError(f"Split manifest is missing columns: {', '.join(missing)}")
    unknown = sorted({row["split"] for row in rows if row["split"] not in SPLIT_NAMES})
    if unknown:
        raise ProjectError(f"Split manifest has unknown split names: {unknown}")
    return rows


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    for name in SPLIT_NAMES:
        if name not in ratios:
            raise ProjectError(f"Missing split ratio '{name}'.")
        if float(ratios[name]) <= 0:
            raise ProjectError(f"Split ratio '{name}' must be positive.")
    total = sum(float(ratios[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise ProjectError(f"Split ratios must sum to 1. They sum to {total}.")
