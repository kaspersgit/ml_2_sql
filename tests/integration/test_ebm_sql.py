# Make inference using both the pickled model and the SQL model
# make sure these two models have the same output
# use example data for this (titanic, although that will only cover the binary classification)
import sys

sys.path.append("scripts")

import os
import joblib
import logging
import pandas as pd
import numpy as np
import pytest
from utils.test_helpers.sql_model import execute_sql_script
from utils.output_scripts.ebm_as_code import save_model_and_extras

# Assuming your_model.pkl is the pickled model and df is the DataFrame for testing
SQL_OUTPUT_PATH = "tests/model/ebm_in_sql.sql"

# Define a list of models - datasets to test
clf_binary = [
    "tests/model/binary_ebm_classification.sav",
    "input/data/example_binary_titanic.csv",
]

clf_multiclass = [
    "tests/model/multiclass_ebm_classification.sav",
    "input/data/example_multiclass_faults.csv",
]

regr_regression = [
    "tests/model/regression_ebm_regression.sav",
    "input/data/example_regression_used_cars.csv",
]

# combine into 1 list to iterate over
fixture_data = [clf_binary, regr_regression]


@pytest.fixture(params=fixture_data)
def load_model_data(request):
    model_path = request.param[0]
    print(model_path)
    model_type = os.path.basename(model_path).split("_")[
        0
    ]  # Binary, multiclass or regression

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    data_path = request.param[1]
    data_type = os.path.basename(data_path).split("_")[
        1
    ]  # Binary, multiclass or regression

    # Load data for testing
    # Return each element of the list as a fixture value
    data = pd.read_csv(data_path)

    if model_type == data_type:
        return data, model, model_type


# Define a fixture for split parameter
@pytest.fixture(params=[True, False])
def split(request):
    return request.param


def test_model_processing(load_model_data, split, logging=logging.getLogger(__name__)):
    # unpack data and model
    data, model, model_type = load_model_data

    # Generate SQL from the loaded model
    save_model_and_extras(ebm=model, model_name="tests", split=split, logging=logging)

    # Load the SQL version
    with open(SQL_OUTPUT_PATH, "r") as sql_file:
        loaded_sql = sql_file.read()

    # Run SQL against the DataFrame using DuckDB

    if model_type == "multiclass":
        prob_column = "total_score"
    elif model_type == "binary":
        prob_column = "probability"
    elif model_type == "regression":
        prob_column = "prediction"

    sql_prob = execute_sql_script(loaded_sql, data, prob_column)

    # Add assertions to check if the results are as expected
    assert sql_prob is not None

    # Predict scores using pickled model
    if model_type == "multiclass":
        score_pred = model.decision_function(data)
        score_pred = score_pred if score_pred.ndim == 1 else np.sum(score_pred, axis=1)
    elif model_type == "binary":
        score_pred = model.predict_proba(data)[:,1]
    elif model_type == "regression":
        score_pred = model.predict(data)

    logging.info(
        f"Max difference SQL - pickled model: {(abs(sql_prob - score_pred)).max()}"
    )

    # Check if SQL model prediction is same as pickled model prediction
    # use a tolerance of
    tolerance = 0.00001
    assert (abs(sql_prob - score_pred) <= tolerance).all()

    # Clean up: Delete the generated SQL file after the test
    import os

    os.remove(SQL_OUTPUT_PATH)
