# Validation report

This file records software checks run while building version 0.1.0. It is not a medical validation, not a BUSI test-set result, and not evidence that a model helps early detection.

No BUSI images were present, and no ImageNet weights were downloaded. The synthetic smoke test does not measure classification performance.

## Environment

| Item | Value |
| --- | --- |
| Operating system | Linux |
| Python | 3.10.22 |
| TensorFlow | 2.17.1 |
| Keras | 3.12.4 |
| NumPy | 1.26.4 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |
| Gradio | 5.50.0 |
| pytest | 9.2.0 |
| GPU | Not available. TensorFlow reported no CUDA driver. |

The integration test builds EfficientNetB0 with `weights=None` and `image_size=32`. The baseline config for a real run remains 224×224 with `weights: imagenet`. That real run has not been executed.

## Commands

From the repository root, after `pip install -e .`:

```bash
python -m compileall -q src tests
python -m pytest -q
```

The notebook file was also parsed with the Python `json` module. Relative documentation links were checked after this report existed. A text search found no API tokens and no `TODO: implement` or “paste code here” placeholders. The notebook’s review flag is initialized to `False`, and the notebook does not assign `reviewed=true` in code.

## What passed

Final command on this tree: `python -m pytest -q`.

Result: **16 passed**, with 2 deprecation warnings emitted by Gradio 5.50 itself (`show_api` will change in Gradio 6). The project tests did not fail.

Covered by those tests:

- Mask filenames are omitted and left on disk. Unreadable and truncated images are excluded with a reason. Truncated-image loading is not enabled.
- Identical files and identical decoded pixels share a `duplicate_group`. dHash candidates are written and are not auto-grouped.
- Filename numbers are not copied into `patient_id`. A supplied id map is.
- EXIF orientation is applied. Training and inference preprocessing both return float32 RGB in `[0, 255]` and do not divide by 255. Inference does not call the augment function.
- Grouped split: same seed is stable, linked ids and exact duplicates stay together, class sets do not disappear, and image sets do not overlap.
- Too few groups is an error. A later edit that links a train image to a test image is an error.
- Training review gate rejects a missing `--confirm-review` and any included row with `reviewed=false`. A changed file hash is an error.
- The training row filter does not return test rows.
- AUROC, average precision, sensitivity, and recall are null when the required class is absent, not zero. A two-class external-style table does not invent `normal` examples.
- Temperature 1 preserves probabilities. A larger temperature softens a peaked row. Threshold selection is a validation-only function.
- Focal loss is per-example, and class weights are not applied both as sample weights and inside focal loss.
- The demo disclaimer avoids diagnostic claims. Public sharing defaults to off. The Gradio interface object builds, with flagging off and analytics off.
- TensorFlow smoke test: EfficientNet `preprocess_input` is a pass-through, the backbone contains `Rescaling`, BatchNormalization moving statistics do not update while those layers are non-trainable, `stem_conv` stays frozen and `block6a_expand_conv` is trainable in the fine-tune phase, focal loss serializes through `get_config`, and a one-epoch-per-phase run on 9 synthetic images saves, reloads, calibrates on validation only, evaluates the held-out synthetic split, and returns three scores that sum to 1.

## What failed and was fixed

1. A unit test rejected the disclaimer because the sentence “does not promise zero storage” contains the words “zero storage”. The sentence is the intended warning. The test was changed to require that sentence. The disclaimer was not weakened.
2. An earlier Keras probe showed that calling the backbone with `training=False` does **not** stop BatchNormalization updates in Keras 3.12 when the layer remains trainable. The training code sets every BatchNormalization layer to `trainable=False` after any unfreeze. The integration test checks that the moving mean is unchanged after a fit.
3. Pillow 11 warned that `Image.fromarray(..., mode="RGB")` is deprecated. The mode argument was removed. The final pytest run no longer reports that warning.

## What was not run

- Training or evaluation on the real BUSI collection.
- Download of ImageNet weights.
- A GPU run.
- GitHub Actions. The workflow file is `.github/workflows/tests.yml`. It was not executed on GitHub from this environment.
- A launched Gradio server, a public share link, or a Colab session.
- An external clinical dataset.
- Any statistical claim about malignant sensitivity, specificity, AUROC, or calibration on medical images.

Those items remain “Not yet evaluated” in the model card.
