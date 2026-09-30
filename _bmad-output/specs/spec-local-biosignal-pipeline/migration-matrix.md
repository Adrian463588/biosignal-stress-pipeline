# Migration and Reuse Matrix

| Source | Decision | v3 use |
|---|---|---|
| `data/raw/DataRenamedPartisipan` | adapt | Canonical immutable ECG and PPG input. |
| `data/raw/DataTrimmedPartisipan` | compare_only | Audit-only; never fed to v3 transformations. |
| `ProsesData2/PipelinePython/common` | adapt | Algorithm and contract donor; repository remains read-only. |
| `ProsesData2/android/inference` | adapt | Fail-closed module boundary and parity-test donor. |
| Colab notebooks | reject | Not part of local execution or release evidence. |
| `gold_features.csv` | compare_only | Legacy same-subject-fold accuracy baseline. |
| old ONNX/Android model assets | reject | Backed up; parity error was outside v3 gate. |
| old hierarchical Bronze/Silver/Gold | reject | Backed up because it included stale slicing/Routine outputs. |
| old validation metrics | compare_only | Historical evidence only; not a v3 synchronization result. |
| personality training | defer | Optional backlog outside the stress MVP Definition of Done. |

The cleanup baseline contained `gold_features.csv` and
`channel_validation_metrics.csv`. The two flat canonical paths
`canonical_signals.csv` and `canonical_segmented_signals.csv` named by legacy
documentation were not physically present, so cleanup could neither move nor
overwrite them. This absence is retained as migration evidence rather than
silently reconstructing files under legacy names.

