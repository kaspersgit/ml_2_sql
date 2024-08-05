import pandas as pd
import logging
from ml2sql.utils.test_helpers.sql_model import execute_sql_script

logger = logging.getLogger(__name__)


def checkTargetHard(target):
    # Convert the target to a pandas Series if it's not already one
    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    # Drop null values from the target
    target_non_null = target.dropna()
    if target_non_null.nunique() == 1:
        raise ValueError("Target column needs more than 1 unique value")
    elif target_non_null.nunique() == 2:
        # Check if the two unique values are 0 and 1
        unique_values = target_non_null.unique()
        if set(unique_values) != set([0, 1]):
            logger.error(
                f"Target with 2 unique values ({unique_values}), these should be converted into 0 and 1"
            )


def checkFeatures(features):
    featNullCount = features.isnull().sum()
    nullf = featNullCount[featNullCount > 0]
    if len(nullf) > 0:
        logger.info(f"NULL values found in the data, for the following: \n{nullf}")


def checkInputData(data, config):
    """
    Checks at start ensuring target and feature columns are good to go
    """
    checkTargetHard(data[config["target"]])
    checkFeatures(data[config["features"]])


def check_sql(loaded_sql, data, prob_column, model_pred):
    sql_pred = execute_sql_script(loaded_sql, data, prob_column)

    # Add assertions to check if the results are as expected
    assert sql_pred is not None

    logger.info(
        f"Max difference SQL - pickled model: {(abs((sql_pred - model_pred)/model_pred)).max()}"
    )

    # Check if SQL model prediction is same as pickled model prediction
    # use a tolerance of 0.001%
    tolerance = 0.00001
    assert (abs((sql_pred - model_pred) / model_pred) <= tolerance).all()
