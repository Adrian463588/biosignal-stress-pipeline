import numpy as np
from PipelinePython.common.cv.leakage import LeakageDetector


def test_check_participant_overlap():
    detector = LeakageDetector()
    train_ids = ["ID-01", "ID-02", "ID-03"]
    val_ids = ["ID-04", "ID-05"]
    assert not detector.check_participant_overlap(train_ids, val_ids)

    val_ids_overlap = ["ID-02", "ID-05"]
    assert detector.check_participant_overlap(train_ids, val_ids_overlap)


def test_check_augmentation_leakage():
    detector = LeakageDetector()
    train_aug = np.random.randn(50, 10)
    val_clean = np.random.randn(50, 10)

    # Should not flag random noise
    assert detector.check_augmentation_leakage(train_aug, val_clean) > 1.0

    # Should flag highly similar data
    val_leak = train_aug + np.random.randn(50, 10) * 0.01
    assert detector.check_augmentation_leakage(train_aug, val_leak) < 1.0
