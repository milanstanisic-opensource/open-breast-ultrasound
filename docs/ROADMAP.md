# Roadmap

This is a sequence of research stages. Later stages are not commitments, and none of them is funded. The current repository is only a start on stage 1.

## 1. Reproducible baseline

- Public code for audit, grouped splits, EfficientNetB0 training, validation-only calibration, and locked evaluation.
- No claimed BUSI score until a run is actually finished and written down with its seed, versions, and split fingerprint.
- Human review of the inventory, using the published BUSI concerns as a checklist, without silent label edits.

## 2. Clinical collaborator and intended use

- A named radiologist or clinical group, with a written role. None is claimed today.
- A one-paragraph intended use that a collaborator agrees to: research only, or a more specific question.
- Agreement on what the tool must not be used for.

## 3. Data and an independent retrospective evaluation

- A second collection with known patient identity, acquired under the permissions that collection requires.
- A locked model from stage 1, evaluated without refitting temperature or the threshold.
- A pre-specified analysis. No retraining after seeing that external result if the result is going to be reported as confirmatory.
- Sample size and missing-data rules written before the run.

## 4. A study with radiologists

- Readers see cases with and without the model scores, under a protocol.
- The endpoint is reader performance or interpretation time, not a claim invented from a heatmap.
- This requires ethics review where the institution requires it. This repository does not have that approval.

## 5. Possible regulatory and clinical development

- Only if the evidence from earlier stages supports a specific use.
- Quality system, clinical evaluation, and regulatory advice are separate work with their own cost.
- Nothing in the current prototype is a technical file, a clinical evaluation report, or an authorization.

## 6. Maintenance and international access

- Keep dependencies installable, tests running, and the documentation honest as libraries change.
- Translations and training materials only after the scientific claim is stable.
- Do not expand into unrelated longevity or bioinformatics products inside this repository's budget or branding.

Stop if a stage shows that the images, the labels, or the model do not support the next claim. Publish that negative or inconclusive result rather than collecting money to continue the same claim.
