# Make inference using both the pickled model and the SQL model
# make sure these two models have the same output
# use example data for this (titanic, although that will only cover the binary classification)
import sys 
sys.path.append('scripts')

import os
import joblib
import logging
import pandas as pd
import numpy as np
import pytest
from utils.test_helpers.sql_model import execute_sql_script
from utils.output_scripts.decision_tree_as_code import save_model_and_extras

# Assuming your_model.pkl is the pickled model and df is the DataFrame for testing
SQL_OUTPUT_PATH = 'tests/model/tree_in_sql.sql'

# Define a list of models - datasets to test
clf_binary = ['tests/model/binary_dt_classification.sav', 'input/data/example_binary_titanic.csv']
regr_regression = ['tests/model/regression_dt_regression.sav', 'input/data/example_regression_used_cars.csv']

# combine into 1 list to iterate over
fixture_data = [clf_binary, regr_regression]


@pytest.fixture(params=fixture_data)
def load_model_data(request):
    model_path = request.param[0]
    print(model_path)
    model_type = os.path.basename(model_path).split('_')[0]  # Binary, multiclass or regression
    
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    
    data_path = request.param[1]
    data_type = os.path.basename(data_path).split('_')[1]   # Binary, multiclass or regression

    # Load data for testing
    # Return each element of the list as a fixture value
    data = pd.read_csv(data_path)

    if model_type == data_type:
        return data, model, model_type


def test_model_processing(load_model_data, sql_split=None, logging=logging.getLogger(__name__)):

    # unpack data and model
    data, model, model_type = load_model_data

    # Generate SQL from the loaded model
    save_model_and_extras(clf=model, model_name='tests', sql_split=sql_split, logging=logging)

    # Load the SQL version
    with open(SQL_OUTPUT_PATH, 'r') as sql_file:
        loaded_sql = sql_file.read()

    # Run SQL against the DataFrame using DuckDB
    
    if model_type == 'multiclass':
        pred_column = 'prediction'
    elif model_type == 'binary':
        pred_column = 'prediction'
    elif model_type == 'regression':
        pred_column = 'prediction'

    sql_pred = execute_sql_script(loaded_sql, data, pred_column)

    # Add assertions to check if the results are as expected
    assert sql_pred is not None

    # Predict scores using pickled model
    if model_type == 'multiclass':
        model_pred = model.predict(data[model.feature_names_in_])
    elif model_type == 'binary':
        model_pred = model.predict(data[model.feature_names_in_])
    elif model_type == 'regression':
        model_pred = model.predict(data[model.feature_names_in_])

    model_pred = model_pred if model_pred.ndim == 1 else np.sum(model_pred, axis=1)
    logging.info(f'Max difference SQL - pickled model: {(abs(sql_pred - model_pred)).max()}')
    print(sql_pred)
    print(model_pred)
    # Check if SQL model prediction is same as pickled model prediction 
    # use a tolerance of 
    tolerance = 0.00001
    assert (abs(sql_pred - model_pred) <= tolerance).all()

    # Clean up: Optionally, you can delete the generated SQL file after the test
    import os
    os.remove(SQL_OUTPUT_PATH)


