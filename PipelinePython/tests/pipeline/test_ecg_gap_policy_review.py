from pathlib import Path

from PipelinePython.common.data_foundation import (
    CANONICAL_ECG_PPG_VALIDATOR,
    LEGACY_ADAPTER_ONLY,
    DataFoundationBuilder,
)


def test_data_foundation_is_explicitly_legacy_adapter() -> None:
    assert LEGACY_ADAPTER_ONLY is True
    assert DataFoundationBuilder.legacy_adapter_only is True
    assert CANONICAL_ECG_PPG_VALIDATOR.endswith("PpgEcgValidator")


def test_primary_ecg_policy_is_declared_in_validator_source() -> None:
    from PipelinePython.common import peak_validation

    source = peak_validation.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "np.interp" not in text
    assert "silver_missing_clean_value" in text
