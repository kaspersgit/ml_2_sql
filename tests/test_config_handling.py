import pytest
import pandas as pd
import numpy as np
from io import StringIO
import logging

# Import the functions to be tested
from ml2sql.utils.helper_functions.config_handling import (
    config_handling,
    _get_col_dtype,
    select_ml_cols,
)


# Setup logging capture
@pytest.fixture
def capture_logs():
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    yield log_capture
    logger.removeHandler(handler)


# Tests for config_handling
def test_config_handling_basic():
    config = {
        "target": "target_column",
        "features": ["feature1", "feature2"],
        "model_params": {"param1": "value1"},
        "pre_params": {"oot_set": "true"},
        "post_params": {"calibration": "auto"},
    }
    data = pd.DataFrame(
        {"target_column": [0, 1], "feature1": [1, 2], "feature2": [3, 4]}
    )

    target_col, feature_cols, model_params, pre_params, post_params = config_handling(
        config, data
    )

    assert target_col == "target_column"
    assert feature_cols == ["feature1", "feature2"]
    assert model_params == {"param1": "value1"}


def test_config_handling_default_values():
    config = {"target": "target_column"}
    data = pd.DataFrame(
        {"target_column": [0, 1], "feature1": [1, 2], "feature2": [3, 4]}
    )

    target_col, feature_cols, model_params, pre_params, post_params = config_handling(
        config, data
    )

    assert target_col == "target_column"
    assert set(feature_cols) == {"feature1", "feature2"}
    assert model_params == {}
    assert pre_params["oot_set"] == "false"
    assert pre_params["cv_type"] == "kfold_cv"
    assert pre_params["upsampling"] == "false"
    assert post_params["calibration"] == "false"
    assert not post_params["sql_split"]
    assert post_params["sql_decimals"] == 15
    assert post_params["file_type"] == "png"


# Tests for _get_col_dtype
def test_get_col_dtype_numeric():
    col = pd.Series([1, 2, 3, 4, 5])
    assert _get_col_dtype(col) == np.dtype("int64")


def test_get_col_dtype_datetime():
    col = pd.Series(["2021-01-01", "2021-01-02", "2021-01-03"])
    assert _get_col_dtype(col) == np.dtype("<M8[ns]")


def test_get_col_dtype_string():
    col = pd.Series(["a", "b", "c", "d"])
    assert _get_col_dtype(col) == "object"


def test_get_col_dtype_mixed():
    col = pd.Series(["a", "2021-01-01", "1", "b"])
    assert _get_col_dtype(col) == "object"


# Tests for select_ml_cols
def test_select_ml_cols_basic(capsys):
    df = pd.DataFrame(
        {
            "id": range(100),
            "constant": ["A"] * 100,
            "high_cardinality_int": np.random.randint(1000000, 9999999, 100),
            "date": pd.date_range(start="2021-01-01", periods=100),
            "high_cardinality_object": [f"value_{i}" for i in range(100)],
            "good_feature1": np.random.rand(100),
            "good_feature2": np.random.choice(["A", "B", "C"], 100),
        }
    )

    selected_features = select_ml_cols(df)

    assert "good_feature1" in selected_features
    assert "good_feature2" in selected_features
    assert "id" not in selected_features
    assert "constant" not in selected_features
    assert "high_cardinality_int" not in selected_features
    assert "date" not in selected_features
    assert "high_cardinality_object" not in selected_features

    captured = capsys.readouterr()
    assert '"id" only has unique values' in captured.out
    assert '"constant" is column with only one value' in captured.out
    assert '"high_cardinality_int" only has unique value' in captured.out
    assert '"date" only has unique values' in captured.out
    assert '"high_cardinality_object" only has unique values' in captured.out


def test_select_ml_cols_all_good_features():
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.choice(["A", "B", "C"], 100),
            "feature3": np.random.randint(1, 5, 100),
        }
    )

    selected_features = select_ml_cols(df)

    assert set(selected_features) == {"feature1", "feature2", "feature3"}


# Run the tests
if __name__ == "__main__":
    pytest.main()
