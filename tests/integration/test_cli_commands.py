import sys

sys.path.append("scripts")

import os
import shutil
import subprocess
import pytest


@pytest.fixture(scope="module")
def setup_file_structure():
    print("Start file structure setup function")
    OUTPUT_PATH = "trained_models/test_tool"
    DATA_PATH = "input/data/example_binary_titanic.csv"

    # Make directory with current data and model name
    try:
        # For main
        os.makedirs(f"{OUTPUT_PATH}")
        os.makedirs(f"{OUTPUT_PATH}/feature_importance")
        os.makedirs(f"{OUTPUT_PATH}/feature_info")
        os.makedirs(f"{OUTPUT_PATH}/performance")
        os.makedirs(f"{OUTPUT_PATH}/model")

        # For modeltester
        csv_name = DATA_PATH.split("/")[-1].split(".")[0]

        destination_path = f"{OUTPUT_PATH}/tested_datasets/{csv_name}"
        os.makedirs(destination_path)
        os.makedirs(f"{destination_path}/performance")
        os.makedirs(f"{destination_path}/local_explanations")

    except FileExistsError:
        sys.exit("Error: Model directory already exists")

    print("finish file structure setup function")
    yield OUTPUT_PATH, DATA_PATH

    # Remove the folder and its contents
    shutil.rmtree(OUTPUT_PATH)


def test_main_script(setup_file_structure):
    print("Start main calling function")
    OUTPUT_PATH, DATA_PATH = setup_file_structure

    # Check platform, windows is different from linux/mac
    if sys.platform == "win32":
        executable = ".ml2sql\\Scripts\\python.exe"
        command = [
            executable,
            "scripts\\main.py",
            "--name",
            OUTPUT_PATH,
            "--data_path",
            DATA_PATH,
            "--configuration",
            "input\\configuration\\example_binary_titanic.json",
            "--model",
            "ebm",
        ]
    else:
        executable = ".ml2sql/bin/python"
        command = [
            executable,
            "scripts/main.py",
            "--name",
            OUTPUT_PATH,
            "--data_path",
            DATA_PATH,
            "--configuration",
            "input/configuration/example_binary_titanic.json",
            "--model",
            "ebm",
        ]

    result = subprocess.run(
        command,
        # stdout=subprocess.PIPE,
        capture_output=True,
        text=True,
        check=False,
    )
    # Check exit code (0 is success)
    assert result.returncode == 0

    # Check for more logged strings in output
    assert "Script input arguments:" in result.stderr
    assert "Configuration file content:" in result.stderr
    assert "Target column has 2 unique values" in result.stderr
    assert "This problem will be treated as a classification problem" in result.stderr

    print("Finish calling main function")


def test_modeltester_script(setup_file_structure):
    print("Start calling modeltester function")
    OUTPUT_PATH, DATA_PATH = setup_file_structure
    DATASET_NAME = os.path.split(DATA_PATH)[-1].split(".")[0]

    # Check platform, windows is different from linux/mac
    if sys.platform == "win32":
        executable = ".ml2sql\\Scripts\\python.exe"
        command = [
            executable,
            "scripts\\modeltester.py",
            "--model_path",
            f"{OUTPUT_PATH}\\model\\ebm_classification.sav",
            "--data_path",
            DATA_PATH,
            "--destination_path",
            f"{OUTPUT_PATH}\\tested_datasets\\{DATASET_NAME}",
        ]
    else:
        executable = ".ml2sql/bin/python"
        command = [
            executable,
            "scripts/modeltester.py",
            "--model_path",
            f"{OUTPUT_PATH}/model/ebm_classification.sav",
            "--data_path",
            DATA_PATH,
            "--destination_path",
            f"{OUTPUT_PATH}/tested_datasets/{DATASET_NAME}",
        ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    print("Finish calling modeltester function")
