# Contributing

Questions and patches are welcome. Clinical claims are not welcome unless they point at a recorded result.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run commands from the repository root.

## What to send

- A focused change with a test when the change affects audit, splits, metrics, training gates, or preprocessing.
- Documentation that matches the code. If a number was not measured, write “Not yet evaluated”.
- English for code, comments, and the docs in this repository.

## What not to send

- The BUSI images, a checkpoint larger than a small test fixture, or `kaggle.json`.
- A seed or a split that was chosen because it produced the best test score.
- An automatic rewrite of BUSI labels.
- A heatmap presented as proof of localization.
- Names of collaborators, hospitals, or approvals that are not real.

## Scientific changes

If you report a metric, include the experiment directory's seed, the split fingerprint, the library versions, and whether patient IDs were known. A same-source test split, an external set, and a patient-level split are three different statements. Do not collapse them.

## License

Contributions to this repository are under the Apache License 2.0 unless a file says otherwise. Do not contribute code you do not have the right to license that way. Dataset files remain under their own terms and should not be added.
