# Architecture

```text
DataRenamedPartisipan (immutable)
  -> ingest and metadata validation
  -> whole-record ECG affine correction and PAT validation
  -> anchor slice (relax/task, 10 s transition buffer)
  -> per-window zero-phase preprocessing and SQA
  -> per-sensor gold features plus masked IBI sequences
  -> 28-participant nested LOSO and frozen 4-participant holdout
  -> signed candidate bundle
  -> FP32 ONNX plus manifest plus parity vectors
  -> Android :inference (fail closed) consumed by :mobile
```

All physical and index artifacts retain participant, session/recording, phase,
sensor, window, source, version, quality, and exclusion lineage. I/O,
transformations, signal algorithms, training, export, and mobile adapters are
separate modules.
