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

Run from the repository root:

```powershell
python -m PipelinePython data --run-id 20260821T120000Z
python -m PipelinePython peaks --run-id 20260821T120000Z --resume
python -m PipelinePython stress-onnx --run-id 20260821T120000Z --sensor ppg_green --resume
python -m PipelinePython all --run-id 20260821T120000Z --resume
```

Raw data, generated artifacts, reports, and Android source remain outside this
folder. The legacy `ColabVersion` notebooks are retained under
`03_stress_onnx/notebooks/legacy`; only the generated canonical notebook at
`03_stress_onnx/notebooks/BiosignalStressTraining_Colab.ipynb` is active.
