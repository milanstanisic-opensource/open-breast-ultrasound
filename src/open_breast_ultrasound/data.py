"""Image discovery, decoding, and the shared train/inference preprocessing path."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

from .utils import CLASS_NAMES, ProjectError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Keras EfficientNet includes Rescaling(1/255) and expects this range.
# Do not divide by 255 before the model, and do not call
# keras.applications.efficientnet.preprocess_input (it is a no-op in current Keras).
MODEL_INPUT_MIN = 0.0
MODEL_INPUT_MAX = 255.0


def is_mask_filename(name: str) -> bool:
    """BUSI segmentation masks contain ``mask`` in the filename.

    This is a filename rule, not a clinical detector. A non-mask file that
    happens to contain that substring is also excluded from classification.
    """
    return "mask" in Path(name).name.lower()


def has_class_directories(root: Path) -> bool:
    names = {child.name.lower() for child in root.iterdir() if child.is_dir()} if root.is_dir() else set()
    return set(CLASS_NAMES).issubset(names)


def resolve_data_root(data_root: Path) -> Path:
    """Return the directory that directly contains benign/malignant/normal."""
    root = Path(data_root).expanduser()
    if not root.exists():
        raise ProjectError(f"Data root does not exist: {root}")
    if not root.is_dir():
        raise ProjectError(f"Data root is not a directory: {root}")
    root = root.resolve()
    if has_class_directories(root):
        return root
    children = sorted(path for path in root.iterdir() if path.is_dir() and has_class_directories(path))
    if len(children) == 1:
        return children[0]
    expected = ", ".join(CLASS_NAMES)
    raise ProjectError(
        f"Could not find class directories ({expected}) in {root} "
        "or in exactly one immediate subdirectory. "
        "Point --data-root at Dataset_BUSI_with_GT or at its parent."
    )


def class_directories(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {name: [] for name in CLASS_NAMES}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in found:
            found[child.name.lower()].append(child)
    ambiguous = {name: paths for name, paths in found.items() if len(paths) != 1}
    if ambiguous:
        raise ProjectError(f"Expected exactly one directory per class, found: {ambiguous}")
    return {name: paths[0] for name, paths in found.items()}


def discover_files(data_root: Path) -> dict[str, list[Path]]:
    """Classify files under the resolved data root.

    Returns keys ``images``, ``masks``, and ``ignored``. Image paths are files
    that will be decoded. Masks are never classification inputs.
    """
    root = resolve_data_root(data_root)
    images: list[Path] = []
    masks: list[Path] = []
    ignored: list[Path] = []
    for directory in class_directories(root).values():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if is_mask_filename(path.name):
                masks.append(path)
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored.append(path)
                continue
            images.append(path)
    return {"root": root, "images": images, "masks": masks, "ignored": ignored}  # type: ignore[dict-item]


def relative_posix(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    if ".." in relative.parts:
        raise ProjectError(f"Path escapes the data root: {path}")
    return relative.as_posix()


def resolve_image_path(data_root: Path, relative_path: str) -> Path:
    if not str(relative_path).strip():
        raise ProjectError("Encountered an empty relative path.")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProjectError(f"Unsafe relative path: {relative_path}")
    root = resolve_data_root(data_root)
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ProjectError(f"Path escapes the data root: {relative_path}")
    return path


def load_rgb(path: Path) -> Image.Image:
    """Fully decode an image to RGB, applying EXIF orientation.

    Truncated files are rejected. ``ImageFile.LOAD_TRUNCATED_IMAGES`` is forced
    off for this call and then restored, so a failure cannot be hidden in order
    to keep training running.
    """
    path = Path(path)
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            if oriented is None:
                raise ProjectError(f"Could not decode image: {path}")
            rgb = oriented.convert("RGB")
            rgb.load()
            # Copy pixels out of the file-backed image before the handle closes.
            return rgb.copy()
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f"Unreadable image ({type(exc).__name__}: {exc}): {path}") from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def image_to_model_array(source: Path | Image.Image, image_size: int) -> np.ndarray:
    """Shared preprocessing for training, evaluation, and the demo.

    Output is float32 ``(image_size, image_size, 3)`` in ``[0, 255]``.
    Values are not divided by 255.
    """
    if image_size < 32:
        raise ProjectError("image_size must be at least 32 for EfficientNet.")
    image = source if isinstance(source, Image.Image) else load_rgb(source)
    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if array.shape != (image_size, image_size, 3):
        raise ProjectError(f"Unexpected array shape {array.shape} for image size {image_size}.")
    if float(array.min()) < MODEL_INPUT_MIN or float(array.max()) > MODEL_INPUT_MAX:
        raise ProjectError("Preprocessed pixels fell outside [0, 255]. Refusing to rescale them implicitly.")
    return array


def prepare_image(
    path: Path,
    image_size: int,
    *,
    training: bool,
    augment_fn=None,
) -> np.ndarray:
    """Decode and resize. Augment only when ``training`` is true."""
    array = image_to_model_array(path, image_size)
    if training:
        if augment_fn is None:
            raise ProjectError("Training preprocessing was asked to augment, but no augment function was provided.")
        array = np.asarray(augment_fn(array), dtype=np.float32)
        if array.shape != (image_size, image_size, 3):
            raise ProjectError(f"Augmentation changed the image shape to {array.shape}.")
        array = np.clip(array, MODEL_INPUT_MIN, MODEL_INPUT_MAX).astype(np.float32)
    return array


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    payload = rgb.size[0].to_bytes(4, "big") + rgb.size[1].to_bytes(4, "big") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (a perceptual hash), returned as an integer.

    Low-detail images can collide. A small distance is only a candidate for
    human review. It does not prove a duplicate file or a shared patient.
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    row_width = hash_size + 1
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * row_width + col]
            right = pixels[row * row_width + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def dhash_hex(value: int, hash_size: int = 8) -> str:
    width = (hash_size * hash_size) // 4
    return f"{value:0{width}x}"


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def augment_training_image(image: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation used only by the training generator.

    Horizontal flip, a small rotation, and a small contrast change.
    Interpolation is done on 8-bit pixels because the source images are 8-bit.
    """
    array = np.asarray(image, dtype=np.float32)
    if config.get("horizontal_flip", True) and float(rng.random()) < 0.5:
        array = np.flip(array, axis=1).copy()
    degrees = float(config.get("rotation_degrees", 0.0) or 0.0)
    if degrees > 0:
        angle = float(rng.uniform(-degrees, degrees))
        pil = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        rotated = pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        array = np.asarray(rotated, dtype=np.float32)
    contrast = float(config.get("contrast", 0.0) or 0.0)
    if contrast > 0:
        factor = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        mean = array.mean(axis=(0, 1), keepdims=True)
        array = (array - mean) * factor + mean
    return np.clip(array, 0.0, 255.0).astype(np.float32)


def label_for_path(path: Path, root: Path) -> str:
    relative = Path(relative_posix(path, root))
    if not relative.parts:
        raise ProjectError(f"Could not assign a label to {path}")
    label = relative.parts[0].lower()
    if label not in CLASS_NAMES:
        raise ProjectError(f"File is not inside a known class directory: {relative.as_posix()}")
    return label
