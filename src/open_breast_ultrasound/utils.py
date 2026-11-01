"""Shared helpers that do not import TensorFlow."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CLASS_NAMES = ("benign", "malignant", "normal")
PROJECT_NAME = "open-breast-ultrasound"


class ProjectError(RuntimeError):
    """A user-facing error. The CLI prints it without a stack trace."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ProjectError(f"Expected a boolean value, got {value!r}.")


def csv_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise ProjectError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProjectError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ProjectError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ProjectError(f"YAML config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProjectError(f"YAML config must be a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def ensure_fresh_directory(path: Path, *, overwrite: bool) -> Path:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ProjectError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise ProjectError(
                    f"Refusing to overwrite non-empty directory: {path}. "
                    "Choose a new output directory, or pass --overwrite to replace it explicitly."
                )
            import shutil

            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(distribution)
    except Exception:
        return None


def library_versions() -> dict[str, str | None]:
    names = ("tensorflow", "keras", "numpy", "pillow", "pyyaml", "gradio")
    return {name: package_version(name) for name in names}


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpus": [],
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:  # pragma: no cover - exercised only without TF
        info["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    return info


def git_commit(start: Path | None = None) -> str | None:
    """Return HEAD if ``start`` is inside a git checkout, else None."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            try:
                completed = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=candidate,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception:
                return None
            if completed.returncode == 0:
                commit = completed.stdout.strip()
                return commit or None
            return None
    return None


def set_global_seed(seed: int) -> None:
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def require_columns(rows: Sequence[Mapping[str, Any]], columns: Iterable[str], *, what: str) -> None:
    if not rows:
        raise ProjectError(f"{what} is empty.")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ProjectError(f"{what} is missing columns: {', '.join(missing)}")


def as_float(value: Any, *, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{what} must be a number, got {value!r}.") from exc


def as_int(value: Any, *, what: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{what} must be an integer, got {value!r}.") from exc
    if not number.is_integer():
        raise ProjectError(f"{what} must be an integer, got {value!r}.")
    return int(number)
