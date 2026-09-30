# Public Data Contracts

The immutable dataclasses in `PipelinePython/common/contracts.py` are executable
contracts. The canonical model interface is:

- input: `float_input`, `float32[N, n_features]`
- output: `stress_probability`, `float32[N]`
- label order: `relax=0`, `task=1`

The candidate manifest records feature order, preprocessing/feature/model
versions, sensor, threshold, model SHA-256, and training/holdout participants.
Android must reject a missing feature, non-finite tensor, model/hash/schema
mismatch, unsupported sensor, poor/invalid quality, or unavailable model.

Raw timestamps and values are append-only. Corrected ECG timestamps are stored
in a distinct field with affine parameters, method, confidence, overlap, peak
metrics, PAT, and exclusion reason.

