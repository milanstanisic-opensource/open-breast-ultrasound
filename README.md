# Open Breast Ultrasound Research

Research prototype for whole-image classification of breast ultrasound into `benign`, `malignant`, and `normal`. Author: Srdjan Stanisic. Version 0.1.0.

This repository is software. It is not a medical device, not a diagnosis, and not evidence that a model improves early detection. No BUSI model has been trained or measured for this release. See [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md).

## What it does

- Inventories a local BUSI-style folder without deleting images or masks.
- Skips segmentation files whose names contain `mask`.
- Records SHA-256, decoded size, exact duplicate groups, and perceptual-hash candidates.
- Writes a seeded group split that keeps known links together.
- Trains an EfficientNetB0 baseline only after an explicit inventory review.
- Fits an optional temperature on validation data and evaluates a locked test split.
- Runs a local demo that shows three model scores.

## What it does not do

- It does not read patient IDs from BUSI filename numbers such as `benign (12)`.
- It does not claim patient-level independence.
- It does not change disputed labels or mark rows as medically reviewed.
- It does not tune the threshold or the temperature on the test split.
- It does not ship data, ImageNet weights, or a trained checkpoint.
- It does not provide Grad-CAM. A heatmap is not included, because it would not be proof of localization.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install provides `obu` and `python -m open_breast_ultrasound.cli`.

Run commands from the repository root so `configs/baseline.yaml` resolves.

## Data

Download the BUSI collection yourself and accept the provider's terms. A common mirror is the Kaggle dataset `aryashah2k/breast-ultrasound-images-dataset`. The original paper is Al-Dhabyani et al., Data in Brief 2020;28:104863, DOI `10.1016/j.dib.2019.104863`.

Expected layout:

```text
Dataset_BUSI_with_GT/
  benign/
  malignant/
  normal/
```

Masks stay on disk. They are not classification inputs. Do not commit the dataset.

Known public-data problems are summarized in [docs/DATA_CARD.md](docs/DATA_CARD.md). This code does not apply those exclusions for you.

## Audit, review, split

```bash
python -m open_breast_ultrasound.cli audit \
  --data-root /path/to/Dataset_BUSI_with_GT \
  --output-dir outputs/audit

python -m open_breast_ultrasound.cli check-review \
  --inventory outputs/audit/inventory.csv
```

Open `outputs/audit/inventory.csv`. For every row you intentionally keep, set `reviewed` to `true` after you have looked at it. Set `include` to `false` to drop a row. `reviewed=true` is not a medical validation.

Optional columns `patient_id` and `case_id` are used only if you put them there, or if you pass `--id-map` with columns `relative_path,patient_id,case_id`. Empty IDs stay empty.

`duplicate_candidates.csv` lists exact file matches, exact decoded-pixel matches, and dHash candidates. Exact matches are grouped automatically. Perceptual candidates are not, unless you pass `--group-perceptual` or edit `duplicate_group` yourself. dHash does not prove a duplicate or a shared patient.

Re-running `audit` into the same folder requires `--overwrite`, because an overwrite would discard review flags.

```bash
python -m open_breast_ultrasound.cli split \
  --inventory outputs/audit/inventory.csv \
  --output-dir outputs/splits \
  --seed 42
```

The default ratios are 0.6 / 0.2 / 0.2. Groups are not broken to force the ratio. The seed is an input. It is not searched against model scores. The same seed and inventory reproduce the manifest. A held-out split from BUSI is not an external validation and is not patient-level separation.

## Train

`configs/baseline.yaml` is the baseline: EfficientNetB0, ImageNet initialization, 224×224 RGB, dropout, frozen BatchNormalization, then a short fine-tune from `block6`. Pixels are float values in `[0, 255]`. They are not divided by 255, and `efficientnet.preprocess_input` is not called, because current Keras EfficientNet already rescales internally.

Training refuses to start unless every included row has `reviewed=true` and you pass `--confirm-review`. It also refuses if a file hash changed after the audit, or if known links now cross the split. It will not silently overwrite an experiment directory.

```bash
python -m open_breast_ultrasound.cli train \
  --config configs/baseline.yaml \
  --data-root /path/to/Dataset_BUSI_with_GT \
  --inventory outputs/audit/inventory.csv \
  --split outputs/splits/split_manifest.csv \
  --output-dir outputs/experiments/baseline-seed42 \
  --confirm-review
```

The test split is not used for fitting, early stopping, checkpointing, or choosing the phase. Each phase gets new EarlyStopping and ModelCheckpoint objects. The kept checkpoint is the one with the lower validation loss. A tie keeps phase 1.

Class weights use `n / (3 * n_class)` from the training split only. With cross-entropy they are training sample weights, and validation loss stays unweighted. With optional `training.loss: focal` the same weights go inside the focal loss instead, not in both places. Focal loss returns a per-example value before Keras reduces it.

## Calibrate and evaluate

```bash
python -m open_breast_ultrasound.cli calibrate \
  --data-root /path/to/Dataset_BUSI_with_GT \
  --experiment outputs/experiments/baseline-seed42 \
  --inventory outputs/audit/inventory.csv \
  --split outputs/splits/split_manifest.csv

python -m open_breast_ultrasound.cli evaluate \
  --data-root /path/to/Dataset_BUSI_with_GT \
  --experiment outputs/experiments/baseline-seed42 \
  --inventory outputs/audit/inventory.csv \
  --split outputs/splits/split_manifest.csv \
  --bootstrap 1000
```

Temperature scaling minimizes validation negative log-likelihood. It does not show clinical reliability. The default malignant threshold is the predefined value `0.5`. Set `decision.threshold_policy` to `validation_max_f1` before training if the threshold must be chosen on validation F1; evaluation then refuses to run until calibration locks it. The threshold is never chosen on the test set and is not clinically approved.

Metrics that are undefined, including AUROC when a class is missing, are JSON `null` with a reason. They are not reported as zero. Group bootstrap resamples groups, not images-as-patients. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

External data, without retraining or changing temperature or threshold:

```bash
python -m open_breast_ultrasound.cli evaluate-external \
  --data-root /path/to/external_images \
  --experiment outputs/experiments/baseline-seed42 \
  --manifest /path/to/external_manifest.csv \
  --output-dir outputs/external_eval
```

The manifest needs `relative_path` and `label`. `group_id` is optional. Labels must already be `benign`, `malignant`, or `normal`. A set with no `normal` rows is valid. The code does not invent the missing class.

## One image and the local demo

```bash
python -m open_breast_ultrasound.cli predict \
  --experiment outputs/experiments/baseline-seed42 \
  --image /path/to/image.png

python -m open_breast_ultrasound.cli demo \
  --experiment outputs/experiments/baseline-seed42
```

The demo binds to `127.0.0.1:7860` and does not create a public link. `--allow-public-share` is off unless you set it, and it is only for images that cannot identify a person. Gradio keeps uploads in temporary process storage. This project does not promise zero storage. Flagging is disabled and Gradio analytics are turned off. There is no heatmap.

## Tests

```bash
pytest
```

Logic tests use synthetic images. The integration test builds EfficientNetB0 with `weights=None` at 32×32 for one epoch per phase. It does not download ImageNet weights. A passing test is not a medical validation.

## GitHub

Publish the extracted repository, not the zip as the only file. From the directory that contains `README.md`:

```bash
git init
git add .
git status
git commit -m "Initial research prototype"
git branch -M main
git remote add origin git@github.com:YOUR_ACCOUNT/open-breast-ultrasound.git
git push -u origin main
```

Create the empty GitHub repository `open-breast-ultrasound` first. Do not commit `data/`, experiment outputs, `kaggle.json`, or checkpoints. Those paths are in `.gitignore`, but `git status` is still worth reading.

The code license is Apache-2.0. Dataset terms and ImageNet weights are separate. See `NOTICE`.

## Google Colab

Open [notebooks/01_train_in_colab.ipynb](notebooks/01_train_in_colab.ipynb). A Colab runtime can stop at any time, and files on the Colab disk are not durable. Copy experiment outputs to Google Drive or download them before the session ends. The notebook does not mark the inventory as reviewed.

## VS Code

1. Open the extracted `open-breast-ultrasound` folder.
2. Choose the Python interpreter in `.venv`.
3. Run the commands above in the integrated terminal, from the folder that contains `pyproject.toml`.
4. Tests live in `tests/` and can be run with the pytest extension or `pytest`.

## Reproducing a result

There is no published performance number to reproduce yet. A future run is described by the saved `experiment.json`: config, seed, library versions, device, split fingerprint, dataset fingerprint, and git commit when the tree is a checkout. Do not replace an old experiment directory in order to improve a test score.

## Contributing and funding

[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md), [docs/ROADMAP.md](docs/ROADMAP.md), [docs/FUNDING_PLAN.md](docs/FUNDING_PLAN.md), [docs/SECURITY.md](docs/SECURITY.md).

Citation metadata is in `CITATION.cff`. It intentionally has no DOI, GitHub URL, or release date until a real release exists.

## Funding links

No WhyDonate, GitHub Sponsors, or other payment URL is included, because none exists for this project. Add a real URL only after the account exists, and do not describe the provisional programme figure as a confirmed budget.
