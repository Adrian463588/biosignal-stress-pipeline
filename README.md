# End-to-End Biosignal Stress Detection Pipeline

This repository contains the local Python pipeline for ECG/PPG data preparation, ECG-PPG agreement analysis, participant-safe model development, and gated ONNX export. The Android application is maintained separately.

> Research software for decision support. It is not a diagnostic device. This repository snapshot contains code and documentation only; participant recordings, cleaned datasets, identity maps, anchors, run outputs, and model weights are not included.

## Android app downloads

The companion Android application is maintained in the [separate Android repository](https://github.com/Adrian463588/android-stress-detection). Its newest phone build is **0.1.5**, published as an internal debug preview, not a production or Play Store release. The companion Wear APK remains version **0.1.1**.

- [Download phone APK 0.1.5](https://github.com/Adrian463588/android-stress-detection/releases/download/android-v0.1.5/app-production-debug.apk)
- [Download Wear APK 0.1.1](https://github.com/Adrian463588/android-stress-detection/releases/download/android-v0.1.1/wear-production-debug.apk)
- [Release notes and checksums for phone 0.1.5](https://github.com/Adrian463588/android-stress-detection/releases/tag/android-v0.1.5)
- [All app release history](https://github.com/Adrian463588/android-stress-detection/releases)

## Locked research contract

- Use only participants `ID-01` through `ID-32`; participant names and identity mappings must never enter this repository.
- Labels are `relax` (0) and `task` (1). Do not create a `routine` label or participant.
- ECG from Polar H10 is a validation reference. Android model input is one PPG channel at a time.
- Samsung Galaxy Watch PPG channels are `ppg_green`, `ppg_ir`, and `ppg_red`; channel models and readiness remain independent.
- Source phases are 300 seconds. Canonical analysis uses the anchor core `[+30s, +270s)` (240 seconds). Gold windows are 180 seconds with 10-second stride, at most seven per phase.
- Development uses `ID-01` to `ID-28`. `ID-29` to `ID-32` are a reused internal test cohort, not external validation, and must not be used for selection or tuning.
- Minimum research gate: balanced accuracy, F1, and ROC-AUC each at least 0.70. Preferred gate: 0.80, 0.78, and 0.83. Accuracy is report-only.
- Classical ML runs locally. Conditional deep learning is a manual Google Colab T4 handoff for a channel that fails the development gate. Upload only the approved channel Gold data with its manifest and checksum; never upload raw recordings.

Read [PRD.md](PRD.md), [AGENTS.md](AGENTS.md), and the pipeline spec under `_bmad-output/specs/spec-local-biosignal-pipeline/` before changing data contracts or model gates.

## Pipeline stages

| Stage | Canonical implementation and config | Contract |
| --- | --- | --- |
| Raw audit and alignment | PipelinePython/common/ingestion.py, PipelinePython/common/raw_sync.py, PipelinePython/common/alignment.py; PipelinePython/configs/data.yaml, PipelinePython/configs/synchronization.yaml | Audit immutable ECG/PPG inputs and lineage. Correct ECG clock offset or drift against anchors before segmentation. |
| Cleaning and preparation | PipelinePython/common/preprocessing.py, PipelinePython/common/madc.py, PipelinePython/common/sqa.py, PipelinePython/common/workflows/data_preparation.py; PipelinePython/configs/cleaning.yaml, PipelinePython/configs/madc.yaml, PipelinePython/configs/sqa.yaml | Create cleaned Silver and quality-gated Gold without padding long gaps or replacing raw input. |
| Peak and HRV-PRV validation | PipelinePython/common/peak_detection.py, PipelinePython/common/synced_peak_validation.py, PipelinePython/common/hrv_agreement.py; PipelinePython/configs/peak_detection.yaml, PipelinePython/configs/hrv_agreement.yaml | Detect ECG R-peaks and each PPG channel independently. ECG RR intervals provide the HRV reference; PPG pulse-to-pulse intervals provide PRV. Beat matching preserves physiological delay and compares adjacent intervals, not zero-lag peak timestamps. The report contract emits 192 participant, phase, and channel rows, including exclusions. |
| Model development | PipelinePython/common/training_canonical.py, PipelinePython/common/training_protocol.py, PipelinePython/common/workflows/stress_onnx.py; PipelinePython/configs/training.yaml, PipelinePython/configs/features.yaml, PipelinePython/configs/frozen_gates.yaml | Train separate Green, IR, and Red PPG models with participant-safe folds. ECG and PAT are validation evidence, never model predictors. |
| ONNX and parity | PipelinePython/common/export_onnx.py, PipelinePython/common/export/model_validator.py, PipelinePython/common/onnx_v3.py; PipelinePython/configs/onnx.yaml, PipelinePython/configs/android.yaml | Export only a gate-qualified model; check ONNX Runtime and FP32 max_abs_diff below 1e-5, then produce the manifest and vectors for Android parity. |

Read Run the pipeline below for the resumable commands. Do not run publish-internal until model, features, quality cohort, calibration, and threshold choices are frozen; it evaluates the reused internal cohort once.

## Repository layout

```text
PipelinePython/
  01_data_preprocess/   raw audit, cleaning, Silver, Gold
  02_peak_to_peak/      ECG/PPG peak and IBI validation
  03_stress_onnx/       model training, Colab handoff, ONNX export
  common/               shared contracts and implementation
  configs/              versioned pipeline and model configuration
  tests/                unit, pipeline, and parity checks
scripts/                local validation and publication audit
_bmad-output/specs/     pipeline decisions and contracts
```

## Setup

Requires Python 3.12 or later. From the repository root:

```powershell
git clone https://github.com/Adrian463588/biosignal-stress-pipeline.git
cd biosignal-stress-pipeline
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Enable the local pre-push publication check once per clone:

```powershell
git config --local core.hooksPath .githooks
```

The hook audits candidate files and reachable history before Git sends a push. CI repeats the audit on pushes and pull requests.

## Prepare local data

Data is not downloaded by this repository. Use only authorized local sources. The default configuration is `PipelinePython/configs/data.yaml`; set its paths to your local input files before running the pipeline.

- ECG input comes from the complete renamed source; PPG input comes from the trimmed phase source; anchors define the canonical phase boundaries.
- Keep participant directories and any retained lineage identifier in `ID-01` through `ID-32` form. Keep real names and the name-to-ID map outside this checkout.
- Keep anchor files, raw signals, cleaned signals, Gold tables, reports, and run artifacts out of Git. The repository `.gitignore` blocks these paths and data formats by default.
- Do not place raw or cleaned physiological data in notebooks. The publication audit rejects tracked data files and notebook execution outputs.

## Run the pipeline

Create one UTC run ID and reuse it across separate stages:

```powershell
$RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
python -m PipelinePython --help
python -m PipelinePython data --run-id $RunId
python -m PipelinePython peaks --run-id $RunId --resume
python -m PipelinePython stress-onnx --sensor ppg_green --export-mode research --run-id $RunId --resume
```

Repeat the `stress-onnx` command for `ppg_ir` and `ppg_red` as independent channels. For a single command that runs the canonical graph through development training, use:

```powershell
python -m PipelinePython all --export-mode research --run-id $RunId --resume
```

Do not use `publish-internal` until the model, features, quality cohort, calibration, and threshold are frozen. It accesses the reused internal test cohort and is single-use under the project contract.

Runs write generated artifacts beneath `data/artifacts/runs/<run-id>/` and reports beneath `reports/runs/<run-id>/`. These remain local and ignored by Git.

## Development checks

```powershell
python -m ruff check PipelinePython/ --config ruff.toml
python -m ruff format --check PipelinePython/ --config ruff.toml
python -m pytest PipelinePython/tests/pipeline/ PipelinePython/tests/unit/ --ignore=PipelinePython/tests/pipeline/test_anchors.py --ignore=PipelinePython/tests/pipeline/test_v3_artifact_gates.py -q
python scripts/audit_public_repo.py --history
```

CI runs Ruff lint and format checks, the publication audit against candidate files and reachable Git history, and the selected pytest suite. Pytest runs only after lint and publication audit pass. GitHub Actions are pinned to immutable commit SHAs; Dependabot checks Python and Actions dependencies weekly.

## Development workflow (BMAD/SDD and BDD)

For a behavior or data-contract change, read `PRD.md`, `AGENTS.md`, and the relevant spec under `_bmad-output/specs/` first. Agree on observable acceptance behavior, update the relevant spec or decision record, make the smallest implementation change, then run the checks above. Documentation-only or tooling-only changes do not need a new product spec. The publication controls for this repository are specified in [the Public Repository Safety spec](_bmad-output/specs/spec-publication-safety/SPEC.md).

Publication safety acceptance examples:

- **Given** a checkout containing only reviewed source and documentation with participant references in `ID-01`–`ID-32` form, **when** `python scripts/audit_public_repo.py --history` runs, **then** it passes without any physiological records, identity map, or executed notebook output.
- **Given** a candidate file or reachable commit containing a prohibited data artifact, credential pattern, participant-name field, or notebook output, **when** the same audit runs, **then** it fails with the path and rule category and does not print a matched secret value.

The audit is a repository guard, not proof that pseudonymized biosignal data is anonymous. This repository accepts reviewed source and documentation only; keep raw and cleaned physiological data, identity mappings, anchors, model weights, and participant-level reports outside Git. Participant IDs remain sensitive when linked to physiological data.

This lightweight workflow follows the existing BMAD/spec artifacts in this repository and uses BDD-style observable outcomes. Reference implementations: [BMAD Method](https://github.com/bmad-code-org/BMAD-METHOD), [OpenSpec spec-driven workflow](https://github.com/Fission-AI/OpenSpec), and [Cucumber Gherkin reference](https://cucumber.io/docs/gherkin/reference/). No additional workflow runtime is required.

## Data privacy and publication

This snapshot contains no physiological datasets. Do not commit raw recordings, anchor tables, identity mappings, generated participant-level data, screenshots, model artifacts, or notebook outputs. Clean data still needs an explicit release review; participant IDs are pseudonymous and do not make biosignal data non-sensitive.

Before any push, run `python scripts/audit_public_repo.py --history`. If a dataset is separately approved for release, review its provenance, columns, filenames, and participant IDs before adding it; update the repository audit policy at the same time.

When GitHub enables the feature for this repository, turn on secret scanning and push protection in **Settings → Code security and analysis**. Keep the CI audit required for pull requests as well; server-side scanning and the repository audit cover different failure paths.

See [SECURITY.md](SECURITY.md) for reporting a suspected credential or data exposure.

## License

No `LICENSE` file is included. Public visibility alone does not grant reuse rights; add an explicit license before accepting third-party reuse.
