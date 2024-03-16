import pytest
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append("scripts")

from create_config import get_input, create_config

# Test data
test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.fixture
def mock_args():
    class MockArgs:
        def __init__(self, data_path):
            self.data_path = data_path

    return MockArgs("test_data.csv")


def test_get_input(monkeypatch):
    # Test valid input
    user_input = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_input))
    options = ["Option 1", "Option 2"]
    message = "Choose an option"
    result = get_input(options, message)
    assert result == "Option 1"

    # Test invalid input
    user_input = iter(["3", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_input))
    result = get_input(options, message)
    assert result == "Option 1"


def test_create_config(mock_args, monkeypatch, tmp_path):
    # Mock user input
    user_input = iter(["0", "2", "3", "5", "no", "1", "2", "2", "10000", "_"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_input))
    monkeypatch.setattr("builtins.print", lambda _: None)

    # Mock data file
    test_data.to_csv(mock_args.data_path, index=False)
    # Test manual config creation
    config_path = create_config(mock_args)
    assert Path(config_path).exists()

    # Load and test the generated config
    with open(config_path) as f:
        config = json.load(f)
    assert set(config["features"]) == set(test_data.columns) - {"A"}
    assert config["target"] == "A"

    # Clean up
    Path(mock_args.data_path).unlink()
    Path(config_path).unlink()


def test_create_config_automatic(mock_args, monkeypatch, tmp_path):
    # Mock user input
    user_input = iter(["0", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_input))
    monkeypatch.setattr("builtins.print", lambda _: None)

    # Mock data file
    test_data.to_csv(mock_args.data_path, index=False)

    # Mock select_ml_cols function
    with monkeypatch.context() as m:
        m.setattr("create_config.select_ml_cols", lambda _: set(test_data.columns))

        # Test automatic config creation
        config_path = create_config(mock_args)
        assert Path(config_path).exists()

        # Load and test the generated config
        with open(config_path) as f:
            config = json.load(f)
        assert set(config["features"]) == set(test_data.columns) - {"A"}
        assert config["target"] == "A"

    # Clean up
    Path(mock_args.data_path).unlink()
    Path(config_path).unlink()


def test_create_config_with_model_params(mock_args, monkeypatch, tmp_path):
    # Mock user input
    user_input = iter(
        [
            "0",
            "2",
            "3",
            "5",
            "yes",
            "max_depth, n_estimators",
            "5, 100",
            "1",
            "2",
            "2",
            "100000",
            "_",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(user_input))
    monkeypatch.setattr("builtins.print", lambda _: None)

    # Mock data file
    test_data.to_csv(mock_args.data_path, index=False)

    # Test manual config creation with model parameters
    config_path = create_config(mock_args)
    assert Path(config_path).exists()

    # Load and test the generated config
    with open(config_path) as f:
        config = json.load(f)
    assert set(config["features"]) == set(test_data.columns) - {"A"}
    assert config["target"] == "A"
    assert config["model_params"] == {"max_depth": "5", "n_estimators": "100"}

    # Clean up
    Path(mock_args.data_path).unlink()
    Path(config_path).unlink()
