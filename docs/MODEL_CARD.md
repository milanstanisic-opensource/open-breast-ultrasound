# Model card

## Model status

Not yet evaluated.

This release contains a training implementation. It does not contain a trained BUSI checkpoint, a test-set score, or a clinical claim. The only model execution recorded in [VALIDATION_REPORT.md](VALIDATION_REPORT.md) is a synthetic smoke test with random initialization (`weights=None`), 32×32 inputs, and one epoch per phase. That smoke test is not a measure of medical performance.

## Intended use and out-of-scope use

Intended use: reproduce a small whole-image research baseline on images a researcher has the right to use, and to report errors honestly.

Out of scope:

- diagnosis, triage, screening, or reassurance of a patient
- claiming that cancer was detected
- localization or segmentation
- deciding a biopsy, a recall, or a treatment
- any statement that the model improves early detection

The software is not a medical device and it has no regulatory authorization.

## Architecture

- Backbone: EfficientNetB0, as implemented by Keras / TensorFlow.
- Initialization for a real run: ImageNet weights, downloaded at training time, not stored here.
- Head: global average pooling (inside the backbone call), dropout (default 0.2), then a dense layer with 3 softmax outputs.
- Class order is fixed: `benign` (index 0), `malignant` (index 1), `normal` (index 2).
- Input: RGB, bilinear resize to 224×224 in the baseline config, float32 values in `[0, 255]`.
- EXIF orientation is applied before resize. The same function is used for training, evaluation, and the demo.
- EfficientNet in current Keras includes `Rescaling(1/255)` and ImageNet normalization. `preprocess_input` is a pass-through. This project does not call it and does not divide by 255 first. If a future Keras build changes that contract, model construction stops instead of double-scaling.

ImageNet pretraining is natural-image pretraining. Ultrasound is a different domain. Initialization is not evidence of clinical performance.

## Training procedure

1. Train the new head with the backbone frozen.
2. Fine-tune from the first layer whose name starts with `block6` (configurable).
3. Batch-normalization layers stay non-trainable in both phases. Keras 3 does not keep their moving statistics frozen merely because the backbone was called with `training=False`; the code sets `trainable=False` on every BatchNormalization layer after any bulk unfreeze, and it recompiles.
4. Augmentation (horizontal flip, small rotation, small contrast) is applied only inside the training sequence.
5. Each phase builds a new EarlyStopping callback and a new ModelCheckpoint callback, both monitored by `val_loss`.
6. The saved model is the phase with the lower best validation loss. A tie keeps phase 1. The test split is not read for this choice.

Loss: sparse categorical cross-entropy by default. Balanced class weights are `n_train / (3 * n_class)`, computed on the training split only, and applied as training sample weights. Validation loss stays unweighted so model selection is not quietly redefined by the weights.

Optional focal loss (`training.loss: focal`) puts those weights inside the loss instead. It is not combined with sample weights. The loss returns one value per example; Keras then applies its reduction. The focal class is registered for `load_model`.

Early stopping, architecture choices, epoch choice, calibration, and the threshold do not use the test split.

## Calibration and decisions

Temperature scaling is optional and fit by minimizing multiclass negative log-likelihood on validation predictions only. The parameter is stored in `temperature.json` and reused by evaluation and the demo. Temperature 1 is used if calibration has not been run.

The default malignant-versus-rest threshold is the predefined value 0.5. The alternative policy `validation_max_f1` selects a threshold by malignant F1 on validation scores after temperature scaling, then locks it. Evaluation refuses to invent a threshold on test or external data. Neither threshold is clinically approved.

Two decisions are reported separately:

- multiclass argmax
- binary decision: malignant if `P(malignant) >= threshold`, otherwise not malignant

Specificity uses that binary setup. The negative class is every non-malignant label present. Sensitivity is malignant recall. Neither is defined when its denominator is zero; the report uses null and a reason, not a fake zero.

Malignant one-vs-rest AUROC is undefined without both malignant and non-malignant examples. Average precision is the non-interpolated area `sum (recall_n - recall_(n-1)) * precision_n` and is named `average_precision` in the report. Multiclass Brier score is the mean over images of the sum over classes of `(p_k - y_k)^2`. Negative log-likelihood is the mean of `-log p_true` with a floor of `1e-15` only to avoid an infinite logarithm.

Group bootstrap percentile intervals resample groups with replacement. Groups are not confirmed patients. Correlated images are not treated as independent patients. Small group counts make the interval unstable, and the report says so.

## Demo

The Gradio app is titled “Open Breast Ultrasound — Research Prototype”. It shows three model scores. It does not say the model is reliable, that it diagnosed anyone, or that it detected cancer. It does not show a heatmap. Public sharing is off unless `--allow-public-share` is set, and that option is only for non-identifying research images. Uploads sit in Gradio's temporary working storage. Zero storage is not promised.

## Factors and limitations

- BUSI has reported duplicates, non-breast images, and questionable labels. See the data card. They are not auto-removed.
- Without patient IDs, images from one person can fall into more than one split.
- A held-out BUSI split is not an external validation.
- The baseline has no test-time augmentation, no segmentation branch, and no reader study.
- Performance numbers below are not available.

## Quantitative results

| Item | Result |
| --- | --- |
| BUSI test-set metrics | Not yet evaluated |
| External dataset | Not yet evaluated |
| Reader study | Not yet evaluated |
| Calibration quality on real images | Not yet evaluated |
| Synthetic smoke test | Software only. See the validation report. Not a medical result. |

## Ethical notes

A wrong malignant score can frighten someone or delay care if it is misused. A wrong non-malignant score can falsely reassure. Do not present scores to patients. Do not upload images you do not have permission to use.
