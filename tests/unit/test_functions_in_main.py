import sys
import os
import shutil
import json
import logging
import pytest
import pandas as pd
import numpy as np
from ml2sql.utils.helper_functions.checks import checkInputData
from ml2sql.utils.helper_functions.config_handling import config_handling
from ml2sql.utils.pre_processing.pre_process import pre_process_kfold


logger = logging.getLogger(__name__)


# Fixtures for test data and configurations
@pytest.fixture
def test_data():
    col1 = np.random.randint(1, 21, size=100)  # Random integers from 1 to 20
    col2 = np.random.randint(1, 21, size=100)  # Random integers from 1 to 20
    target = np.random.randint(0, 2, size=100)  # Random binary values (0 or 1)

    df = pd.DataFrame({"col1": col1, "col2": col2, "target": target})
    csv_path = "input/data/test_data.csv"
    df.to_csv(csv_path, index=False)
    yield csv_path, df
    os.remove(csv_path)


@pytest.fixture
def test_config():
    config = {
        "target": "target",
        "features": ["col1", "col2"],
        "model_params": {},
        "pre_params": {
            "oot_set": "false",
            "cv_type": "kfold_cv",
            "upsampling": "false",
        },
        "post_params": {
            "calibration": "false",
            "sql_split": False,
            "sql_decimals": 15,
            "file_type": "png",
        },
    }
    config_path = "input/configuration/test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    yield config_path, config
    os.remove(config_path)


@pytest.fixture
def test_path():
    OUTPUT_PATH = "trained_models/test_tool"

    # Make directory with current data and model name
    try:
        # For main
        os.makedirs(f"{OUTPUT_PATH}")
        os.makedirs(f"{OUTPUT_PATH}/feature_importance")
        os.makedirs(f"{OUTPUT_PATH}/feature_info")
        os.makedirs(f"{OUTPUT_PATH}/performance")
        os.makedirs(f"{OUTPUT_PATH}/model")

    except FileExistsError:
        sys.exit("Error: Model directory already exists")

    yield OUTPUT_PATH

    # Remove the folder and its contents
    shutil.rmtree(OUTPUT_PATH)


# Test helper functions
def test_config_handling(test_data, test_config):
    csv_path, df = test_data
    config_path, config = test_config

    target_col, feature_cols, model_params, pre_params, post_params = config_handling(
        config, df
    )
    assert target_col == "target"
    assert feature_cols == ["col1", "col2"]
    assert model_params == {}
    assert pre_params == {
        "oot_set": "false",
        "cv_type": "kfold_cv",
        "upsampling": "false",
    }
    assert post_params == {
        "calibration": "false",
        "sql_split": False,
        "sql_decimals": 15,
        "file_type": "png",
    }


def test_checkInputData(test_data, test_config):
    csv_path, df = test_data
    config_path, config = test_config

    checkInputData(df, config)  # No assertion needed, it should not raise an error


def test_pre_process_kfold(test_data, test_config, test_path):
    csv_path, df = test_data
    config_path, config = test_config

    datasets = pre_process_kfold(
        test_path,
        df,
        "target",
        ["col1", "col2"],
        model_name="ebm",
        model_type="classification",
        pre_params=config["pre_params"],
        post_params=config["post_params"],
        random_seed=42,
    )
    assert len(datasets["cv_train"]["X"]) == 5  # Assuming 5-fold cross-validation
