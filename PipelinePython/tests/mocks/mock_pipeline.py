class MockPipeline:
    """Mock for upstream pipeline artifacts in tests."""

    @staticmethod
    def get_mock_metrics():
        return {"balanced_accuracy": 0.81}

    @staticmethod
    def get_failing_mock_metrics():
        return {"balanced_accuracy": 0.50}
