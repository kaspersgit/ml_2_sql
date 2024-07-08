import pytest
import pandas as pd
import logging
from io import StringIO

# Import the functions to be tested
from ml2sql.utils.helper_functions.checks import (
    checkTargetHard,
    checkFeatures,
    checkInputData,
)


# Setup logging capture
@pytest.fixture
def capture_logs():
    logger = (
        logging.getLogger()
    )  # Make sure this matches the logger name in your actual code
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger.addHandler(handler)

    yield log_capture

    logger.removeHandler(handler)
    logger.setLevel(previous_level)


# Tests for checkTargetHard
def test_checkTargetHard_valid():
    target = pd.Series([0, 1, 0, 1])
    checkTargetHard(target)  # Should not raise any exception


def test_checkTargetHard_single_value():
    target = pd.Series([1, 1, 1])
    with pytest.raises(
        ValueError, match="Target column needs more than 1 unique value"
    ):
        checkTargetHard(target)


def test_checkTargetHard_non_binary(capture_logs):
    target = pd.Series([1, 2, 1, 2])
    checkTargetHard(target)
    assert "Target with 2 unique values" in capture_logs.getvalue()


def test_checkTargetHard_with_nulls():
    target = pd.Series([0, 1, None, 0])
    checkTargetHard(target)  # Should not raise any exception


def test_checkTargetHard_list_input():
    target = [0, 1, 0, 1]
    checkTargetHard(target)  # Should not raise any exception


# Tests for checkFeatures
def test_checkFeatures_no_nulls(capture_logs):
    features = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    checkFeatures(features)
    assert "NULL values found" not in capture_logs.getvalue()


def test_checkFeatures_with_nulls(capture_logs):
    features = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
    checkFeatures(features)
    assert "NULL values found" in capture_logs.getvalue()


# Tests for checkInputData
def test_checkInputData_valid():
    data = pd.DataFrame(
        {"target": [0, 1, 0, 1], "feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]}
    )
    config = {"target": "target", "features": ["feature1", "feature2"]}
    checkInputData(data, config)  # Should not raise any exception


def test_checkInputData_invalid_target():
    data = pd.DataFrame(
        {"target": [1, 1, 1, 1], "feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]}
    )
    config = {"target": "target", "features": ["feature1", "feature2"]}
    with pytest.raises(
        ValueError, match="Target column needs more than 1 unique value"
    ):
        checkInputData(data, config)


def test_checkInputData_features_with_nulls(capture_logs):
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "feature1": [1, None, 3, 4],
            "feature2": [5, 6, None, 8],
        }
    )
    config = {"target": "target", "features": ["feature1", "feature2"]}
    checkInputData(data, config)
    assert "NULL values found" in capture_logs.getvalue()
