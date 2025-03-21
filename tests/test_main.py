import pytest
from typer.testing import CliRunner
import sys
from ml2sql.main import app
import json
from datetime import datetime
import os
import pandas as pd
from tests.test_constants import PROBLEM_TYPE


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("session_temp")
    yield temp_dir


@pytest.fixture
def setup_run_environment(temp_dir):
    # Create necessary directories and files
    (temp_dir / "input" / "data").mkdir(parents=True, exist_ok=True)
    (temp_dir / "input" / "configuration").mkdir(parents=True, exist_ok=True)
    (temp_dir / "trained_models").mkdir(exist_ok=True)

    # Create a dummy CSV file with some data
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "feature1": [1, 2, 3, 4, 6, 7, 8, 3, 6, 1],
            "feature2": [5, 6, 7, 8, 3, 4, 6, 7, 3, 1],
        }
    )
    df.to_csv(temp_dir / "input" / "data" / "AA_test.csv", index=False)

    # Create a dummy JSON configuration file
    config = {
        "target": "target",
        "features": ["feature1", "feature2"],
        "model_params": {},
    }

    with open(temp_dir / "input" / "configuration" / "AA_config.json", "w") as f:
        json.dump(config, f)

    # Change to the temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    # Change back to the original directory after the test
    os.chdir(original_dir)


def test_version(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ml2sql v" in result.stdout


# Test cli_init
def test_init_command(temp_dir, runner):
    result = runner.invoke(app, ["init", "--dest", str(temp_dir)])
    assert result.exit_code == 0

    # Check if the command output indicates success
    assert "ml2sql project initialized in" in result.stdout

    # Check if the expected folders were created
    assert (temp_dir / "input" / "data").is_dir()
    assert (temp_dir / "input" / "configuration").is_dir()
    assert (temp_dir / "trained_models").is_dir()

    # Check if demo data was copied (you may need to adjust this based on your actual demo data)
    assert any((temp_dir / "input" / "data").iterdir()), "No files found in input/data"
    assert any(
        (temp_dir / "input" / "configuration").iterdir()
    ), "No files found in input/configuration"


def test_init_command_existing_directory(temp_dir, runner):
    # Create a file in the directory to simulate an existing project
    (temp_dir / "existing_file.txt").touch()

    result = runner.invoke(app, ["init", "--dest", str(temp_dir)])
    assert result.exit_code == 0

    # Check if the command completed without errors
    assert "ml2sql project initialized in" in result.stdout

    # Check if the existing file is still there
    assert (temp_dir / "existing_file.txt").exists()


def test_run_command(setup_run_environment, runner, caplog):
    # Avoid I/O error by not having any logger  produce a message
    caplog.set_level(100000)

    temp_dir = setup_run_environment

    # Create the input sequence for test_run
    user_inputs = (
        "1\n"  # Select the first CSV file (AA_test.csv)
        "2\n"  # Select the first JSON file (we created one in setup)
        "1\n"  # Select the first model type (assuming it's "Explainable Boosting Machine")
        "test_model\n"  # Enter model name
    )

    # Run the `run` command to create a model
    try:
        result = runner.invoke(app, ["run"], input=user_inputs)

        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")

        if result.exception:
            print(f"Exception: {result.exception}")
            print(f"Traceback: {result.exc_info}")

        assert result.exit_code == 0
        assert "Starting script to create model" in result.output

        # Check if the model directory was created
        current_date = datetime.today().strftime("%Y%m%d")
        model_dir = temp_dir / "trained_models" / f"{current_date}_test_model"
        assert model_dir.is_dir(), f"Model directory not found: {model_dir}"
        assert (model_dir / "feature_importance").is_dir()
        assert (model_dir / "feature_info").is_dir()
        assert (model_dir / "performance").is_dir()
        assert (model_dir / "model").is_dir()

        # Check if model files were created
        assert any(
            (model_dir / "model").glob("*.sav")
        ), "No pickle file found in model directory"
        assert (
            model_dir / "performance" / "test_roc_plot.png"
        ).exists(), "At least one metrics file not found"
        assert any(
            (model_dir / "feature_importance").glob("*.png")
        ), "No plot found in feature_importance directory"

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        print(f"Contents of current directory: {os.listdir()}", file=sys.stderr)
        raise
