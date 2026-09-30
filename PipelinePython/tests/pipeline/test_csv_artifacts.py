from __future__ import annotations

from pathlib import Path

import pandas as pd

from PipelinePython.common.csv_artifacts import export_run_csv


def test_export_run_csv_mirrors_supported_layers_without_raw(tmp_path: Path) -> None:
    (tmp_path / "silver" / "ID-01" / "relax").mkdir(parents=True)
    (tmp_path / "gold").mkdir()
    (tmp_path / "validation" / "ppg_ecg" / "peak_tables" / "ID-01" / "relax").mkdir(
        parents=True
    )
    pd.DataFrame({"value": [1.0]}).to_parquet(
        tmp_path / "silver" / "ID-01" / "relax" / "ppg_ir.parquet",
        index=False,
    )
    pd.DataFrame({"label": [0]}).to_parquet(
        tmp_path / "gold" / "gold_features_v3.parquet", index=False
    )
    pd.DataFrame({"matched": [True]}).to_parquet(
        tmp_path
        / "validation"
        / "ppg_ecg"
        / "peak_tables"
        / "ID-01"
        / "relax"
        / "ppg_ir.parquet",
        index=False,
    )

    result = export_run_csv(tmp_path)

    assert result["silver"] == 1
    assert result["gold"] == 1
    assert result["validation"] == 1
    assert (
        tmp_path / "csv_export" / "silver" / "ID-01" / "relax" / "ppg_ir.csv"
    ).is_file()
    assert not (tmp_path / "csv_export" / "raw").exists()
