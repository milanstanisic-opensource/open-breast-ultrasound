# Data card

## Dataset this software is written for

Name: Dataset of breast ultrasound images (BUSI), as published by Al-Dhabyani, Gomaa, Khaled, and Fahmy.

Citation: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020;28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

A commonly used download mirror is the Kaggle dataset `aryashah2k/breast-ultrasound-images-dataset`:

<https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>

This repository does not redistribute the images. Check the terms on the page you download from before copying or sharing them. The Apache-2.0 license of this code does not license the images.

## What the original paper reports

These figures are citations of the original paper. This software has not re-counted a fresh download in this release, because the images are not shipped here.

- Images collected in 2018 from women aged 25 to 75 years.
- The paper reports 600 female patients.
- 780 PNG images, with an average size around 500×500 pixels.
- Reported class counts: 437 benign, 210 malignant, 133 normal.
- Ground-truth segmentation masks are distributed with the images.
- The public release used for most papers is the folder tree `benign/`, `malignant/`, and `normal/`.

## What this software does not treat as a patient ID

The public filenames look like `benign (12).png` and `benign (12)_mask.png`. The number is not used as a `patient_id` or a `case_id`. The original paper's count of 600 patients is not a file-to-patient table. If `patient_id` is empty, this software records that patient-level independence is not established.

You may supply `patient_id` and `case_id` yourself with `--id-map` or by editing those columns. The software still does not verify them.

## Masks

Any file whose name contains `mask` is omitted from classification inputs and left on disk. BUSI often has more than one mask for an image (`_mask`, `_mask_1`). This project does not train a segmentation model.

## Known problems reported by others

Pawłowska A, Karwat P, Żołek N. Letter to the Editor. Re: “Dataset of breast ultrasound images by W. Al-Dhabyani, M. Gomaa, H. Khaled & A. Fahmy, Data in Brief, 2020, 28, 104863”. Data in Brief. 2023;48:109247. DOI: [10.1016/j.dib.2023.109247](https://doi.org/10.1016/j.dib.2023.109247).

Their summary states: “In summary, 155 images (similar images are included - see supplementary material and images marked with purple color) are copies of other images in the dataset accounting for 19% of the total collection. The 70 images (more than 8% of the entire collection) show other structures than the breast. The type of the lesion (normal, benign, malignant) of at least 19 images (more than 2%) is questionable.” They also state that removing the listed inaccuracies reduces the collection to at most 529 images.

This repository does not automatically delete those files or change their labels. A published objection list still needs a human decision against the actual files. Some duplicates have been discussed elsewhere as appearing under more than one class label. Treat that as a review item. Do not let the training script “correct” a label.

Supplementary material for the letter is on Mendeley Data, DOI [10.17632/k8t3gnx9h6.1](https://doi.org/10.17632/k8t3gnx9h6.1).

## What the audit writes

`inventory.csv` includes `image_id`, `relative_path`, `label`, `sha256`, `width`, `height`, `patient_id`, `case_id`, `duplicate_group`, `include`, `reviewed`, and `review_note`, plus exclusion and hash columns. Paths are relative to the resolved data root and use forward slashes.

- Unreadable or truncated images are excluded with a reason. Truncated decoding is not enabled to keep training alive.
- Exact file hashes and exact decoded RGB pixels form `duplicate_group`.
- 64-bit difference-hash (dHash) pairs within a distance threshold are written to `duplicate_candidates.csv`. They are not auto-grouped unless `--group-perceptual` is set. dHash misses some duplicates, collides on low-detail images, and does not identify a patient.
- `reviewed` defaults to `false`. Setting it to `true` means a person looked at the row. It is not a radiology review and it is not a corrected ground truth.

## Splits

Included images are grouped by supplied `patient_id`, supplied `case_id`, and exact `duplicate_group`. The default target is 60% / 20% / 20%. Groups are not broken. The seed is recorded and is not chosen by model score.

Three different claims must stay separate:

| Claim | Status in this release |
| --- | --- |
| Held-out test split from the same BUSI folders | Implemented as a software split. Not yet run on the real dataset in this release. |
| External validation on another site or dataset | Supported only as a locked-model command. Not yet performed. |
| Confirmed patient-level separation | Not established. Filename numbers are not patient IDs. |

## Labels and intended use

The three folder labels are the training targets. They are not a BI-RADS category and not a biopsy report. The intended use of the data tooling is reproducible research on a public teaching collection. It is not a clinical data pipeline.

## Personal data

Do not add identifiable clinical images to this repository, to a public Gradio link, or to an issue. The BUSI release is a public research collection; that does not make every later upload safe to share.
