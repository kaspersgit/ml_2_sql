# Load packages
import logging
import pandas as pd
import json

# Main modelling function
from ml2sql.utils.modelling.main_modeler import make_model

# The translations to SQL (grey as we refer to them dynamically)
from ml2sql.utils.output_scripts import decision_tree_as_code  # noqa: F401
from ml2sql.utils.output_scripts import ebm_as_code  # noqa: F401
from ml2sql.utils.output_scripts import l_regression_as_code  # noqa: F401

from ml2sql.utils.helper_functions.checks import checkInputData
from ml2sql.utils.helper_functions.setup_logger import setup_logger

from ml2sql.utils.helper_functions.config_handling import config_handling
from ml2sql.utils.pre_processing.pre_process import pre_process_kfold


def modelcreater(data_path, config_path, model_name, project_name):
    """
    Main function to train machine learning models and save the trained model along with its SQL representation.
    """

    # Set logger
    setup_logger(project_name + "/logging.log")
    logger = logging.getLogger(__name__)
    logger.info(
        f"Script input arguments: \ndata_path: {data_path} \nconfig_path: {config_path} \nmodel_name: \n {model_name} \nproject_name: {project_name}"
    )

    # Load data
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(
        data_path,
        keep_default_na=False,
        na_values=["", "N/A", "NULL", "None", "NONE"],
    )

    # Load configuration
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path) as json_file:
        configuration = json.load(json_file)

    # Handle the configuration file
    target_col, feature_cols, model_params, pre_params, post_params = config_handling(
        configuration, data
    )

    logger.info(f"Configuration file content: {configuration}")

    # Perform input checks
    checkInputData(data, configuration)

    # Determine model type
    if (data[target_col].dtype == "float") or (
        (data[target_col].dtype == "int") and (data[target_col].nunique() > 10)
    ):
        model_type = "regression"
    else:
        model_type = "classification"

    logger.info(f"Target column has {data[target_col].nunique()} unique values")
    logger.info(f"This problem will be treated as a {model_type} problem")

    # Preprocess data
    logger.info("Preprocessing data...")
    datasets = pre_process_kfold(
        project_name,
        data,
        target_col,
        feature_cols,
        model_name=model_name,
        model_type=model_type,
        pre_params=pre_params,
        post_params=post_params,
        random_seed=42,
    )

    # Train model
    logger.info(f"Training {model_name} model...")
    clf = make_model(
        project_name,
        datasets,
        model_name=model_name,
        model_type=model_type,
        model_params=model_params,
        post_params=post_params,
    )

    # Create SQL version of model and save it
    logger.info(f"Saving {model_name} model and its SQL representation...")
    globals()[f"{model_name}_as_code"].save_model_and_extras(
        clf, project_name, post_params
    )

    logger.info("Script finished.")
