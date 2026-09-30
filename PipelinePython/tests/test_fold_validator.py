from PipelinePython.common.cv.fold_validator import FoldValidator


def test_fold_validator():
    validator = FoldValidator(min_ba_threshold=0.3)

    assert validator.validate("ID-01", {"balanced_accuracy": 0.85}) is True
    assert validator.validate("ID-02", {"balanced_accuracy": 0.15}) is False
    assert validator.validate("ID-03", {"balanced_accuracy": 0.35}) is True
