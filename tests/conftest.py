# test_utils.py

import pytest
import os
from typer.testing import CliRunner
from pathlib import Path
from ml2sql.main import app


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    session_temp = tmp_path_factory.mktemp("ml2sql_session")
    return session_temp


@pytest.fixture(scope="session")
def init_ml2sql(runner, temp_dir):
    original_dir = Path.cwd()
    os.chdir(temp_dir)

    try:
        result = runner.invoke(app, ["init", "--dest", str(temp_dir)])
        assert result.exit_code == 0, f"Init command failed: {result.output}"
    finally:
        os.chdir(original_dir)

    return temp_dir


@pytest.fixture(autouse=True)
def change_test_dir(init_ml2sql):
    original_dir = Path.cwd()
    os.chdir(init_ml2sql)
    yield
    os.chdir(original_dir)


def find_sav_file(directory):
    directory = Path(directory)
    sav_files = list(directory.glob("*.sav"))
    if not sav_files:
        raise FileNotFoundError(f"No .sav file found in {directory}")
    return str(sav_files[0])


def find_data_file(directory, keyword):
    directory = Path(directory)
    matching_files = list(directory.glob(f"*{keyword}*.csv"))
    if not matching_files:
        raise FileNotFoundError(
            f"No .csv file containing '{keyword}' found in {directory}"
        )
    return str(matching_files[0])
