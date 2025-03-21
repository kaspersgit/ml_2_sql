import pytest
import os
import sys
import joblib
import pandas as pd
from ml2sql.main import app
from ml2sql.utils.test_helpers.sql_model import execute_sql_script
from datetime import datetime
from tests.test_constants import PROBLEM_TYPE
from tests.conftest import (
    find_sav_file,
    find_data_file,
)


@pytest.mark.parametrize("problem_type", PROBLEM_TYPE)
def test_lregression_model(init_ml2sql, runner, caplog, problem_type):  # noqa: F811
    caplog.set_level(100000)
    temp_dir = init_ml2sql  # noqa: F811
    model_choice = 3  # Logistic/Linear Regression model

    if problem_type == "binary":
        data_choice = 1
    elif problem_type == "multiclass":
        data_choice = 2
    elif problem_type == "regression":
        data_choice = 3

    config_add = 1
    model_type = "lregression"
    project_name = f"test_{model_type}_{problem_type}_model"
    user_inputs = (
        f"{data_choice}\n"
        f"{data_choice + config_add}\n"
        f"{model_choice}\n"
        f"{project_name}\n"
    )

    try:
        result = runner.invoke(app, ["run"], input=user_inputs)
        assert result.exit_code == 0
        assert "Starting script to create model" in result.output

        current_date = datetime.today().strftime("%Y%m%d")
        model_dir = temp_dir / "trained_models" / f"{current_date}_{project_name}"
        assert model_dir.is_dir(), f"Model directory not found: {model_dir}"
        assert (model_dir / "feature_importance").is_dir()
        assert (model_dir / "feature_info").is_dir()
        assert (model_dir / "performance").is_dir()
        assert (model_dir / "model").is_dir()

        assert any(
            (model_dir / "model").glob("*.sav")
        ), "No pickle file found in model directory"
        assert (model_dir / "performance").glob(
            "*.png"
        ), "No png file found for performance metrics"
        assert any(
            (model_dir / "feature_importance").glob("*.png")
        ), "No plot found in feature_importance directory"

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        print(f"Contents of current directory: {os.listdir()}", file=sys.stderr)
        raise

    MODEL_PATH = find_sav_file(model_dir / "model")
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    DATA_PATH = find_data_file(temp_dir / "input" / "data", problem_type)
    data = pd.read_csv(DATA_PATH, nrows=99)

    SQL_OUTPUT_PATH = model_dir / "model" / f"{model_type}_in_sql.sql"
    with open(SQL_OUTPUT_PATH, "r") as sql_file:
        loaded_sql = sql_file.read()

    if problem_type == "regression":
        pred_column = "prediction"
        model_pred = model.predict(data[model.feature_names_in_])
    elif problem_type == "binary":
        pred_column = "probability"
        model_pred = model.predict_proba(data[model.feature_names_in_])[:, 1]
    elif problem_type == "multiclass":
        pred_column = "probability_Z_Scratch"
        model_pred = model.predict_proba(data[model.feature_names_in_])[:, -1]

    sql_pred = execute_sql_script(loaded_sql, data, pred_column)
    assert sql_pred is not None

    print(f"Max difference SQL - pickled model: {(abs(sql_pred - model_pred)).max()}")
    tolerance = 0.0001
    assert (abs((sql_pred - model_pred) / model_pred) <= tolerance).all()
