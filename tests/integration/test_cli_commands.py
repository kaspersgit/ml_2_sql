import sys

sys.path.append("scripts")

import os
import shutil
import subprocess
import pytest


@pytest.fixture(scope="module")
def setup_file_structure():
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

    except FileExistsError:
        sys.exit("Error: Model directory already exists")

    yield OUTPUT_PATH, DATA_PATH

    # Remove the folder and its contents
    shutil.rmtree(OUTPUT_PATH)


def test_main_script(setup_file_structure):
    OUTPUT_PATH, DATA_PATH = setup_file_structure

    # Check platform, windows is different from linux/mac
    if sys.platform == "win32":
        executable = ".ml2sql\\Scripts\\python.exe"
    else:
        executable = ".ml2sql/bin/python"

    result = subprocess.run(
        [
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
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_modeltester_script(setup_file_structure):
    OUTPUT_PATH, DATA_PATH = setup_file_structure
    MODEL_PATH = f"{OUTPUT_PATH}/model/ebm_classification.sav"
    DATASET_NAME = DATA_PATH.split("/")[-1].split(".")[0]
    DESTINATION_PATH = f"{OUTPUT_PATH}/tested_datasets/{DATASET_NAME}"

    # Check platform, windows is different from linux/mac
    if sys.platform == "win32":
        executable = ".ml2sql\\Scripts\\python.exe"
    else:
        executable = ".ml2sql/bin/python"

    result = subprocess.run(
        [
            executable,
            "scripts/modeltester.py",
            "--model_path",
            MODEL_PATH,
            "--data_path",
            DATA_PATH,
            "--destination_path",
            DESTINATION_PATH,
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0
