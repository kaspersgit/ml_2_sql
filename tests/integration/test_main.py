import sys

sys.path.append("scripts")

import os
import shutil
import subprocess


def test_cli_exit_code():
    OUTPUT_PATH = "trained_models/test_tool"

    # Make directory with current data and model name
    try:
        os.makedirs(f"{OUTPUT_PATH}")
        os.makedirs(f"{OUTPUT_PATH}/feature_importance")
        os.makedirs(f"{OUTPUT_PATH}/feature_info")
        os.makedirs(f"{OUTPUT_PATH}/performance")
        os.makedirs(f"{OUTPUT_PATH}/model")
    except FileExistsError:
        sys.exit("Error: Model directory already exists")

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
            "input/data/example_binary_titanic.csv",
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

    # Remove the folder and its contents
    shutil.rmtree(OUTPUT_PATH)
