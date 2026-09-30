# End-to-End Biosignal Stress Detection Pipeline

This repository contains the local Python pipeline for ECG/PPG data preparation, ECG-PPG agreement analysis, participant-safe model development, and gated ONNX export. The Android application is maintained separately.

> Research software for decision support. It is not a diagnostic device. This repository snapshot contains code and documentation only; participant recordings, cleaned datasets, identity maps, anchors, run outputs, and model weights are not included.

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

## Pipeline

```text
local ECG + PPG
  -> raw audit and manifest
  -> ECG clock correction, cleaning, MADC, and SQA
  -> Silver and Gold preparation
  -> ECG/PPG peak and IBI agreement validation
  -> participant-safe channel-specific ML
  -> conditional Colab T4 handoff when required
  -> gated ONNX export and parity evidence
```

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
python -m pip install -e .
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

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
python -m pytest PipelinePython/tests/pipeline/ PipelinePython/tests/unit/ --ignore=PipelinePython/tests/pipeline/test_anchors.py --ignore=PipelinePython/tests/pipeline/test_v3_artifact_gates.py -q
python scripts/audit_public_repo.py --history
```

The publication audit checks candidate files and reachable Git history for participant datasets, credential patterns, and notebook outputs. It reports file paths and rule names, never matched secret values.

## Data privacy and publication

This snapshot contains no physiological datasets. Do not commit raw recordings, anchor tables, identity mappings, generated participant-level data, screenshots, model artifacts, or notebook outputs. Clean data still needs an explicit release review; participant IDs are pseudonymous and do not make biosignal data non-sensitive.

Before any push, run `python scripts/audit_public_repo.py --history`. If a dataset is separately approved for release, review its provenance, columns, filenames, and participant IDs before adding it; update the repository audit policy at the same time.

## License

No `LICENSE` file is included. Public visibility alone does not grant reuse rights; add an explicit license before accepting third-party reuse.
