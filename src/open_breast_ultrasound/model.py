"""EfficientNetB0 baseline.

TensorFlow and Keras are imported inside functions so metric tests can import
the focal-loss formula without initializing TensorFlow.

Input contract: float32 RGB images in ``[0, 255]``. Current Keras EfficientNet
includes ``Rescaling(1/255)`` followed by ImageNet normalization.
``keras.applications.efficientnet.preprocess_input`` is a pass-through.
This project does not call it and does not divide by 255 first.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .utils import CLASS_NAMES, ProjectError


def focal_per_example_numpy(
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    gamma: float,
    class_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Focal loss for each example, before any batch reduction.

    ``class_weights``, when given, is the only weighting. Do not also pass
    Keras ``class_weight`` or per-example sample weights.
    """
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if probs.ndim != 2 or len(probs) != len(labels):
        raise ProjectError("Focal loss expected one probability row per label.")
    probability_true = probs[np.arange(len(labels)), labels]
    loss = -((1.0 - probability_true) ** float(gamma)) * np.log(probability_true)
    if class_weights is not None:
        weights = np.asarray(list(class_weights), dtype=np.float64)
        if np.any(labels < 0) or int(labels.max()) >= len(weights):
            raise ProjectError("Focal class weights do not cover every label index.")
        loss = loss * weights[labels]
    return loss


def assert_efficientnet_input_contract() -> None:
    from keras.applications.efficientnet import preprocess_input

    sample = np.array([0.0, 128.0, 255.0], dtype=np.float32)
    output = np.asarray(preprocess_input(sample), dtype=np.float32).reshape(-1)
    if output.shape != sample.shape or not np.allclose(output, sample):
        raise ProjectError(
            "Refusing to build the model. efficientnet.preprocess_input changes pixel values "
            "in this Keras version. This project feeds [0, 255] floats into the network's own "
            "Rescaling layer and does not apply a second scaling. Upgrade or pin Keras/TensorFlow "
            "to a version where EfficientNet preprocessing is internal and preprocess_input is a pass-through."
        )


def _batch_norm_type():
    import keras

    return keras.layers.BatchNormalization


def configure_backbone(backbone, *, phase: str, fine_tune_from: str) -> dict[str, int]:
    """Freeze the backbone, or unfreeze it from ``fine_tune_from`` onward.

    Batch-normalization layers are always set non-trainable after the other
    flags. In Keras 3, ``training=False`` alone does not stop moving-statistic
    updates when a BatchNormalization layer is still trainable.
    """
    if phase == "frozen":
        backbone.trainable = False
    elif phase == "finetune":
        backbone.trainable = True
        reached = False
        for layer in backbone.layers:
            if layer.name.startswith(fine_tune_from):
                reached = True
            layer.trainable = bool(reached)
        if not reached:
            preview = [layer.name for layer in backbone.layers[:8]]
            raise ProjectError(
                f"No backbone layer name starts with {fine_tune_from!r}. First layers: {preview}."
            )
    else:
        raise ProjectError(f"Unknown backbone phase {phase!r}. Use 'frozen' or 'finetune'.")
    batch_norm = _batch_norm_type()
    frozen_bn = 0
    for layer in backbone.layers:
        if isinstance(layer, batch_norm):
            layer.trainable = False
            frozen_bn += 1
    if frozen_bn == 0:
        raise ProjectError("The backbone has no BatchNormalization layers. Refusing to train.")
    trainable = sum(1 for layer in backbone.layers if layer.trainable)
    return {"batch_norm_frozen": frozen_bn, "trainable_layers": trainable}


def build_model(
    *,
    image_size: int,
    class_names: Sequence[str] = CLASS_NAMES,
    dropout: float = 0.2,
    weights: str | None = "imagenet",
    fine_tune_from: str = "block6",
):
    """Return ``(model, backbone)`` with the backbone frozen and BN non-trainable."""
    if tuple(class_names) != tuple(CLASS_NAMES):
        raise ProjectError(f"class_names must be {list(CLASS_NAMES)} in that order.")
    if image_size < 32:
        raise ProjectError("image_size must be at least 32.")
    if not 0.0 <= float(dropout) < 1.0:
        raise ProjectError("dropout must be in [0, 1).")
    assert_efficientnet_input_contract()
    import keras

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
        name="efficientnetb0",
    )
    if not any(isinstance(layer, keras.layers.Rescaling) for layer in backbone.layers):
        raise ProjectError(
            "This EfficientNetB0 build has no internal Rescaling layer. "
            "Refusing to train on [0, 255] inputs, which would skip the scaling this project relies on."
        )
    configure_backbone(backbone, phase="frozen", fine_tune_from=fine_tune_from)
    # training=False is extra protection. BN trainable=False is what Keras 3 honors.
    features = backbone(inputs, training=False)
    dropped = keras.layers.Dropout(float(dropout), name="head_dropout")(features)
    outputs = keras.layers.Dense(len(class_names), activation="softmax", name="predictions")(dropped)
    model = keras.Model(inputs, outputs, name="open_breast_ultrasound_efficientnetb0")
    return model, backbone


class FocalLoss:
    """Placeholder replaced by the Keras Loss subclass once Keras is imported.

    ``build_focal_loss`` returns the real class. This name exists so tests can
    import the numpy formula without constructing the Keras object.
    """


def build_focal_loss(gamma: float, class_weights: Sequence[float] | None):
    import keras

    class KerasFocalLoss(keras.losses.Loss):
        """Returns a per-example vector. Keras applies ``reduction`` afterwards."""

        def __init__(self, gamma: float = gamma, class_weights: Sequence[float] | None = class_weights, name: str = "focal_loss", reduction: str = "sum_over_batch_size", **kwargs):
            super().__init__(name=name, reduction=reduction, **kwargs)
            self.gamma = float(gamma)
            self.class_weights = None if class_weights is None else [float(value) for value in class_weights]

        def call(self, y_true, y_pred):
            import tensorflow as tf

            labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            probs = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
            probability_true = tf.gather(probs, labels, batch_dims=1)
            loss = -tf.pow(1.0 - probability_true, self.gamma) * tf.math.log(probability_true)
            if self.class_weights is not None:
                weights = tf.constant(self.class_weights, dtype=loss.dtype)
                loss = loss * tf.gather(weights, labels)
            return loss

        def get_config(self):
            config = super().get_config()
            config.update({"gamma": self.gamma, "class_weights": self.class_weights})
            return config

    return KerasFocalLoss


def get_custom_objects() -> dict[str, Any]:
    loss_cls = globals().get("KerasFocalLoss")
    if not isinstance(loss_cls, type):
        loss_cls = build_focal_loss(2.0, None)
        loss_cls.__module__ = __name__
        globals()["KerasFocalLoss"] = loss_cls
    return {"KerasFocalLoss": loss_cls, "focal_loss": loss_cls}


def compile_model(model, *, learning_rate: float, loss_name: str, focal_gamma: float, focal_class_weights: Sequence[float] | None) -> None:
    import keras

    if loss_name == "sparse_categorical_crossentropy":
        loss: Any = "sparse_categorical_crossentropy"
    elif loss_name == "focal":
        loss_cls = build_focal_loss(focal_gamma, focal_class_weights)
        loss_cls.__module__ = __name__
        globals()["KerasFocalLoss"] = loss_cls
        loss = loss_cls()
    else:
        raise ProjectError(
            f"Unknown loss {loss_name!r}. Use sparse_categorical_crossentropy or focal. "
            "Class weights are applied once, either as training sample weights or inside focal loss."
        )
    model.compile(
        optimizer=keras.optimizers.Adam(float(learning_rate)),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def training_objective(loss_name: str, use_class_weight: bool, class_weight_list: Sequence[float] | None) -> dict[str, Any]:
    """Choose a single place to apply class weights."""
    if loss_name not in {"sparse_categorical_crossentropy", "focal"}:
        raise ProjectError(f"Unknown loss {loss_name!r}.")
    weights = list(class_weight_list) if use_class_weight and class_weight_list is not None else None
    if loss_name == "focal":
        return {
            "loss_name": "focal",
            "sequence_class_weights": None,
            "focal_class_weights": weights,
            "where": "inside_focal_loss" if weights is not None else "nowhere",
        }
    return {
        "loss_name": "sparse_categorical_crossentropy",
        "sequence_class_weights": weights,
        "focal_class_weights": None,
        "where": "training_sample_weight" if weights is not None else "nowhere",
    }


def balanced_class_weight_list(labels: Sequence[str], class_names: Sequence[str] = CLASS_NAMES) -> list[float]:
    """``n_samples / (n_classes * n_class)`` from the training split only."""
    total = len(list(labels))
    if total == 0:
        raise ProjectError("Cannot weight classes without training labels.")
    weights = []
    n_classes = len(class_names)
    for name in class_names:
        count = sum(1 for label in labels if label == name)
        if count == 0:
            raise ProjectError(f"Training split has no '{name}' images, so a class weight is undefined.")
        weights.append(total / (n_classes * count))
    return weights
