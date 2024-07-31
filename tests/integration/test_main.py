import pytest
from typer.testing import CliRunner
from pathlib import Path
from ml2sql.main import app
import os
import typer
from datetime import datetime

runner = CliRunner()


@pytest.fixture(scope="module")
def setup_folder_structure(tmp_path_factory):
    # Create a temporary directory
    tmp_path = tmp_path_factory.mktemp("data")

    # Convert to pathlib.Path object
    tmp_path = Path(tmp_path)

    # Set up directories and files
    os.chdir(tmp_path)

    # Run the init command to set up the project
    result = runner.invoke(app, ["init", "--dest", tmp_path])

    typer.echo(f"Result: {result.output}")

    if result.exit_code != 0:
        typer.echo(f"Init command failed with exit code {result.exit_code}")
        typer.echo(result.output)
    assert result.exit_code == 0

@pytest.fixture(scope="module")
def setup_environment(tmp_path_factory):
    # Create a temporary directory
    tmp_path = tmp_path_factory.mktemp("data")

    # Convert to pathlib.Path object
    tmp_path = Path(tmp_path)

    # Set up directories and files
    os.chdir(tmp_path)

    # Run the init command to set up the project
    result = runner.invoke(app, ["init", "--dest", tmp_path])

    typer.echo(f"Result: {result.output}")

    if result.exit_code != 0:
        typer.echo(f"Init command failed with exit code {result.exit_code}")
        typer.echo(result.output)
    assert result.exit_code == 0

    # Create the input sequence for test_run
    user_inputs = (
        "1\n"  # Select the first CSV file
        "1\n"  # Select Create New Config File
        "0\n"  # Select the first column as the target
        "1\n"  # Select automatic creation of config
        "2\n"  # Select the first actual JSON file
        "1\n"  # Select the first model type
        "test_model\n"  # Enter model name
    )

    typer.echo(f"User inputs: {user_inputs}")

    # Run the `run` command to create a model
    result = runner.invoke(app, ["run"], input=user_inputs, catch_exceptions=False)

    if result.exit_code != 0:
        typer.echo(f"Run command failed with exit code {result.exit_code}")
        typer.echo(result.output)
        typer.echo(result.exception)
    assert result.exit_code == 0

    yield tmp_path

    # Cleanup after tests
    os.chdir("/")  # Change back to root directory
    tmp_path.rmdir()  # Remove the temporary directory


def test_version():
    result = runner.invoke(app, ["--version"])

    typer.echo(f"Result: {result.output}")

    assert result.exit_code == 0
    assert "ml2sql v" in result.output


def test_init(mocker, tmp_path):
    tmp_path = Path(tmp_path)
    os.chdir(tmp_path)  # Change to temporary directory for testing

    result = runner.invoke(app, ["init", "--dest", tmp_path])

    # Mock the os.system call to prevent actual execution of commands
    # mocker.patch("os.system")

    typer.echo(f"Result: {result.output}")
    typer.echo(f"Current working directory: {os.getcwd()}")
    typer.echo(f"Contents of current directory: {os.listdir()}")

    assert result.exit_code == 0
    assert "ml2sql project initialized in" in result.output

    # Check if directories are created
    input_dir = tmp_path / "input"
    data_dir = input_dir / "data"
    config_dir = input_dir / "configuration"
    trained_models_dir = tmp_path / "trained_models"

    typer.echo(f"input_dir exists: {input_dir.exists()}")
    typer.echo(f"data_dir exists: {data_dir.exists()}")
    typer.echo(f"config_dir exists: {config_dir.exists()}")
    typer.echo(f"trained_models_dir exists: {trained_models_dir.exists()}")

    assert input_dir.exists()
    assert data_dir.exists()
    assert config_dir.exists()
    assert trained_models_dir.exists()


def test_run(mocker, setup_folder_structure):
    tmp_path = setup_folder_structure
    os.chdir(tmp_path)

    # Mock the os.system call to prevent actual execution of commands
    mocker.patch("os.system")

    # Create the input sequence for test_run
    user_inputs = (
        "1\n"  # Select the first CSV file
        "2\n"  # Select the first JSON file
        "1\n"  # Select the first model type
        "test_run_main\n"  # Enter model name
    )

    result = runner.invoke(app, ["run"], input=user_inputs)

    typer.echo(f"Result: {result.output}")

    assert result.exit_code == 0
    assert (
        "CSV file input/data/example_binary_titanic.csv will be used for modelling"
        in result.output
    )
    assert (
        "Configuration file input/configuration/example_binary_titanic.json will be used for modelling"
        in result.output
    )
    assert (
        "Algorithm chosen for modelling: Explainable Boosting Machine" in result.output
    )
    assert "Starting script to create model" in result.output
    assert "Model outputs can be found in folder:" in result.output


def test_check_model(mocker, setup_environment):
    tmp_path = setup_environment
    os.chdir(tmp_path)

    # Mock the os.system call to prevent actual execution of commands
    mocker.patch("os.system")

    # Create the input sequence for test_check_model
    user_inputs = (
        "1\n"  # Select the first CSV file
        "1\n"  # Select the first model file
    )

    # Run CLI command
    result = runner.invoke(
        app, ["check-model"], input=user_inputs, catch_exceptions=False
    )

    # get current date for folder name
    # Current date
    current_date = datetime.today().strftime("%Y%m%d")

    if result.exit_code != 0:
        typer.echo(f"Check model command failed with exit code {result.exit_code}")
        typer.echo("Output:")
        typer.echo(result.output)
        if result.exception:
            typer.echo("Exception:")
            typer.echo(result.exception)
    assert result.exit_code == 0
    assert (
        "CSV file input/data/example_binary_titanic.csv will be used for testing model"
        in result.output
    )
    assert (
        f"Model trained_models/{current_date}_test_model/model/ebm_classification.sav will be used for testing"
        in result.output
    )
    assert "Model performance outputs can be found in folder:" in result.output
