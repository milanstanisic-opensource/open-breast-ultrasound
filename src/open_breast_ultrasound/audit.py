"""Non-destructive inventory, duplicate candidates, and review gates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import (
    dhash,
    dhash_hex,
    discover_files,
    hamming_distance,
    is_mask_filename,
    label_for_path,
    load_rgb,
    pixel_sha256,
    relative_posix,
    resolve_data_root,
    resolve_image_path,
)
from .utils import (
    CLASS_NAMES,
    ProjectError,
    parse_bool,
    read_csv,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)

INVENTORY_FIELDS = (
    "image_id",
    "relative_path",
    "label",
    "sha256",
    "width",
    "height",
    "patient_id",
    "case_id",
    "duplicate_group",
    "include",
    "reviewed",
    "review_note",
    "exclusion_reason",
    "pixel_sha256",
    "perceptual_hash",
    "source_mode",
)

CANDIDATE_FIELDS = (
    "image_id_a",
    "image_id_b",
    "relative_path_a",
    "relative_path_b",
    "method",
    "distance",
    "similarity",
    "auto_grouped",
    "note",
)

REVIEW_NOT_MEDICAL = (
    "reviewed=true records that a person looked at this inventory row. "
    "It is not a medical validation, a radiology review, or a corrected label."
)
PATIENT_ID_LIMITATION = (
    "No confirmed patient_id is present in the public BUSI filenames. "
    "Numbers in names such as 'benign (12)' were not used as patient IDs. "
    "Patient-level independence is not established."
)
PERCEPTUAL_NOTE = (
    "dHash distance is a review aid only. It does not prove that files are "
    "identical or that they come from the same patient, and it misses duplicates "
    "that do not share this hash."
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
        # Keep the sort order stable by pointing the larger root at the smaller one.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def image_id_for(relative_path: str) -> str:
    import hashlib

    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"img_{digest}"


def load_id_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_csv(path)
    required = {"relative_path"}
    if rows and not required.issubset(rows[0]):
        raise ProjectError("ID map must include a relative_path column.")
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("relative_path", "").strip()
        if not key:
            continue
        mapping[key] = {
            "patient_id": row.get("patient_id", "").strip(),
            "case_id": row.get("case_id", "").strip(),
        }
    return mapping


def run_audit(
    data_root: Path,
    output_dir: Path,
    *,
    id_map: Path | None = None,
    perceptual_max_distance: int = 6,
    group_perceptual: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write ``inventory.csv`` without deleting or modifying source images."""
    from .utils import ensure_fresh_directory

    if perceptual_max_distance < 0 or perceptual_max_distance > 64:
        raise ProjectError("perceptual max distance must be between 0 and 64.")
    output_dir = ensure_fresh_directory(Path(output_dir), overwrite=overwrite)
    discovered = discover_files(data_root)
    root: Path = discovered["root"]  # type: ignore[assignment]
    id_lookup = load_id_map(id_map)

    decoded: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for path in discovered["images"]:  # type: ignore[union-attr]
        relative = relative_posix(path, root)
        label = label_for_path(path, root)
        file_hash = sha256_file(path)
        row: dict[str, Any] = {
            "image_id": image_id_for(relative),
            "relative_path": relative,
            "label": label,
            "sha256": file_hash,
            "width": "",
            "height": "",
            "patient_id": id_lookup.get(relative, {}).get("patient_id", ""),
            "case_id": id_lookup.get(relative, {}).get("case_id", ""),
            "duplicate_group": "",
            "include": False,
            "reviewed": False,
            "review_note": "",
            "exclusion_reason": "",
            "pixel_sha256": "",
            "perceptual_hash": "",
            "source_mode": "",
        }
        try:
            image = load_rgb(path)
        except ProjectError as exc:
            row["include"] = False
            row["exclusion_reason"] = str(exc)
            unreadable.append(row)
            continue
        row["width"] = image.width
        row["height"] = image.height
        row["source_mode"] = "RGB"
        row["pixel_sha256"] = pixel_sha256(image)
        row["perceptual_hash"] = dhash_hex(dhash(image))
        row["include"] = True
        decoded.append(row)

    _assign_duplicate_groups(decoded, group_perceptual=group_perceptual, perceptual_max_distance=perceptual_max_distance)
    candidates = _candidate_rows(
        decoded,
        perceptual_max_distance=perceptual_max_distance,
    )
    rows = sorted(decoded + unreadable, key=lambda row: row["relative_path"])
    unused_map = sorted(set(id_lookup) - {row["relative_path"] for row in rows})
    summary = {
        "resolved_data_root_name": root.name,
        "class_names": list(CLASS_NAMES),
        "n_images_discovered": len(discovered["images"]),  # type: ignore[arg-type]
        "n_included": sum(1 for row in rows if row["include"]),
        "n_excluded": sum(1 for row in rows if not row["include"]),
        "n_masks_skipped": len(discovered["masks"]),  # type: ignore[arg-type]
        "n_non_image_files_ignored": len(discovered["ignored"]),  # type: ignore[arg-type]
        "n_duplicate_groups": len({row["duplicate_group"] for row in decoded if row["duplicate_group"]}),
        "n_perceptual_candidate_pairs": sum(1 for row in candidates if row["method"] == "perceptual_dhash"),
        "perceptual_hash": "dhash64",
        "perceptual_candidate_max_distance": perceptual_max_distance,
        "group_perceptual_candidates": group_perceptual,
        "perceptual_hash_limitation": PERCEPTUAL_NOTE,
        "patient_id_source": "user_supplied_id_map" if id_map else "not_available",
        "patient_id_confirmed_by_this_software": False,
        "patient_id_limitation": PATIENT_ID_LIMITATION,
        "filename_numbers_used_as_patient_id": False,
        "labels_automatically_changed": False,
        "reviewed_default": False,
        "review_meaning": REVIEW_NOT_MEDICAL,
        "id_map_paths_not_found": unused_map,
        "inventory_fingerprint": sha256_json(
            [
                {
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "include": row["include"],
                    "label": row["label"],
                }
                for row in rows
            ]
        ),
        "known_busi_concerns": (
            "Pawłowska, Karwat and Żołek, Data in Brief 2023;48:109247 "
            "(DOI 10.1016/j.dib.2023.109247) report copies, non-breast images, "
            "and questionable labels in BUSI. This audit does not apply that "
            "list automatically and does not change labels."
        ),
    }
    write_csv(output_dir / "inventory.csv", rows, INVENTORY_FIELDS)
    write_csv(output_dir / "duplicate_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_json(output_dir / "audit_summary.json", summary)
    return {"output_dir": str(output_dir), "summary": summary, "n_rows": len(rows)}


def _assign_duplicate_groups(
    rows: Sequence[dict[str, Any]],
    *,
    group_perceptual: bool,
    perceptual_max_distance: int,
) -> None:
    finder = _UnionFind()
    for row in rows:
        finder.add(row["relative_path"])
    by_file: dict[str, list[str]] = defaultdict(list)
    by_pixel: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_file[row["sha256"]].append(row["relative_path"])
        by_pixel[row["pixel_sha256"]].append(row["relative_path"])
    for bucket in list(by_file.values()) + list(by_pixel.values()):
        for other in bucket[1:]:
            finder.union(bucket[0], other)
    if group_perceptual:
        hashes = [(row["relative_path"], int(row["perceptual_hash"], 16)) for row in rows]
        for index, (left_path, left_hash) in enumerate(hashes):
            for right_path, right_hash in hashes[index + 1 :]:
                if hamming_distance(left_hash, right_hash) <= perceptual_max_distance:
                    finder.union(left_path, right_path)
    components: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        components[finder.find(row["relative_path"])].append(row["relative_path"])
    group_of: dict[str, str] = {}
    for members in components.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        import hashlib

        digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:12]
        group = f"dup_{digest}"
        for member in ordered:
            group_of[member] = group
    for row in rows:
        row["duplicate_group"] = group_of.get(row["relative_path"], "")


def _candidate_rows(rows: Sequence[dict[str, Any]], *, perceptual_max_distance: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: row["relative_path"])
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left["sha256"] == right["sha256"]:
                method, distance, similarity, grouped = "identical_file", 0, 1.0, True
            elif left["pixel_sha256"] and left["pixel_sha256"] == right["pixel_sha256"]:
                method, distance, similarity, grouped = "identical_pixels", 0, 1.0, True
            else:
                distance = hamming_distance(int(left["perceptual_hash"], 16), int(right["perceptual_hash"], 16))
                if distance > perceptual_max_distance:
                    continue
                method = "perceptual_dhash"
                similarity = 1.0 - (distance / 64.0)
                grouped = left["duplicate_group"] != "" and left["duplicate_group"] == right["duplicate_group"]
            candidates.append(
                {
                    "image_id_a": left["image_id"],
                    "image_id_b": right["image_id"],
                    "relative_path_a": left["relative_path"],
                    "relative_path_b": right["relative_path"],
                    "method": method,
                    "distance": distance,
                    "similarity": f"{similarity:.6f}",
                    "auto_grouped": grouped,
                    "note": PERCEPTUAL_NOTE if method == "perceptual_dhash" else "Exact match on file bytes or decoded RGB pixels.",
                }
            )
    return candidates


def load_inventory(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    missing = [column for column in INVENTORY_FIELDS if rows and column not in rows[0]]
    if missing:
        raise ProjectError(f"Inventory is missing columns: {', '.join(missing)}")
    for row in rows:
        if row.get("label") not in CLASS_NAMES and parse_bool(row.get("include", "false")):
            raise ProjectError(f"Included row has an unknown label: {row.get('label')!r}")
        if is_mask_filename(row.get("relative_path", "")):
            raise ProjectError(f"Inventory contains a mask path as an image: {row.get('relative_path')}")
    return rows


def included_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if parse_bool(row["include"])]


def assert_reviewed_for_training(rows: Sequence[Mapping[str, str]], *, confirm_review: bool) -> None:
    if not confirm_review:
        raise ProjectError(
            "Training requires --confirm-review after a person has reviewed inventory.csv. "
            + REVIEW_NOT_MEDICAL
        )
    included = included_rows(rows)
    if not included:
        raise ProjectError("The inventory has no included images to train on.")
    pending = [row["relative_path"] for row in included if not parse_bool(row["reviewed"])]
    if pending:
        raise ProjectError(
            f"{len(pending)} included image(s) still have reviewed=false. "
            "Set reviewed=true yourself only after checking the row. "
            + REVIEW_NOT_MEDICAL
            + f" First pending file: {pending[0]}"
        )


def assert_files_unchanged(rows: Sequence[Mapping[str, str]], data_root: Path) -> None:
    root = resolve_data_root(data_root)
    checked = 0
    for row in included_rows(rows):
        path = resolve_image_path(root, row["relative_path"])
        if not path.is_file():
            raise ProjectError(f"Included file is missing: {row['relative_path']}")
        current = sha256_file(path)
        if current != row["sha256"]:
            raise ProjectError(
                "File content changed after the inventory was written: "
                f"{row['relative_path']}. Inventory sha256={row['sha256']} "
                f"current sha256={current}. Re-run the audit and review the new inventory. "
                "Training and evaluation will not continue on a swapped file."
            )
        checked += 1
    if checked == 0:
        raise ProjectError("No included files were available to hash-check.")
