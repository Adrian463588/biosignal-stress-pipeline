# Canonical pipeline workflows

All operator-facing end-to-end flows are exposed by the numbered folders under
`PipelinePython`. Domain implementations remain in `PipelinePython/common`;
this folder owns stage order,
dependencies, resume behavior, and the public flow names.

| Flow | Purpose | Main outputs |
| --- | --- | --- |
| `data` | raw audit, ingest, ECG clock correction, segmentation, cleaning, MADC/SQA, peak-ready Gold | Bronze, Sync, Silver, Gold, readiness, EDA |
| `peaks` | development-only detector audit plus ECG--PPG validation | detector selection, 192 validation rows, IBI/HR/HRV agreement reports |
| `stress-onnx` | participant-safe classical ML, conditional Colab handoff, gated ONNX export | model reports, handoff manifest when needed, ONNX/parity/test vectors |
| `all` | complete graph in canonical order | all outputs above |

Run from the repository root:

```powershell
python -m PipelinePython all --run-id 20260821T120000Z --resume
```

For separate, resumable flows, reuse the same `--run-id`:

```powershell
python -m PipelinePython data --run-id 20260821T120000Z --resume
python -m PipelinePython peaks --run-id 20260821T120000Z --resume
python -m PipelinePython stress-onnx --run-id 20260821T120000Z --resume
```

Use `--sensor ppg_green|ppg_ir|ppg_red` only for the stress flow. Use
`--participant ID-XX` only for a data flow that intentionally processes one
participant. Raw inputs remain under `data/raw`; the runner writes only to the
run-scoped artifact/report roots.

`stress-onnx` fails closed when no channel passes the locked development and
reused-internal-test gates. Conditional DL is not executed locally: the flow
writes a checksum-bound handoff for manual Google Colab T4 execution only.
