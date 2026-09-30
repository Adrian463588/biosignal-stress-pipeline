# PipelinePython

Canonical local Python pipeline for biosignal stress detection.

```text
01_data_preprocess/  raw audit, cleaning, Silver, Gold
02_peak_to_peak/     ECG-PPG peak detection and validation
03_stress_onnx/      participant-safe ML, Colab handoff, ONNX export
common/              shared domain implementation and contracts
configs/             canonical Python pipeline configuration
tests/               Python pipeline tests
```

Run from the repository root with one UTC run ID:

    $RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    python -m PipelinePython data --run-id $RunId --resume
    python -m PipelinePython peaks --run-id $RunId --resume
    python -m PipelinePython stress-onnx --run-id $RunId --sensor ppg_green --export-mode research --resume

Use python -m PipelinePython all --run-id $RunId --resume --export-mode research for the complete graph. Train each PPG channel independently. Use --smoke to stop before training and export.

Raw data, generated artifacts, reports, and Android source remain outside this
folder. Only the canonical handoff notebook at
03_stress_onnx/notebooks/BiosignalStressTraining_Colab.ipynb is included. Legacy
notebooks and their execution outputs are not published.
