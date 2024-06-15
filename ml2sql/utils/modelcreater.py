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
from ml2sql.utils.helper_functions.parsing_arguments import GetArgs
from ml2sql.utils.pre_processing.pre_process import pre_process_kfold


def main(args):
    """
    Main function to train machine learning models and save the trained model along with its SQL representation.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
    """

    # Set logger
    setup_logger(args.name + "/logging.log")
    logger = logging.getLogger(__name__)
    logger.info(f"Script input arguments: {args}")

    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    data = pd.read_csv(
        args.data_path,
        keep_default_na=False,
        na_values=["", "N/A", "NULL", "None", "NONE"],
    )

    # Load configuration
    logger.info(f"Loading configuration from {args.configuration}...")
    with open(args.configuration) as json_file:
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
        args.name,
        data,
        target_col,
        feature_cols,
        model_name=args.model_name,
        model_type=model_type,
        pre_params=pre_params,
        post_params=post_params,
        random_seed=42,
    )

    # Train model
    logger.info(f"Training {args.model_name} model...")
    clf = make_model(
        args.name,
        datasets,
        model_name=args.model_name,
        model_type=model_type,
        model_params=model_params,
        post_params=post_params,
    )

    # Create SQL version of model and save it
    logger.info(f"Saving {args.model_name} model and its SQL representation...")
    globals()[f"{args.model_name}_as_code"].save_model_and_extras(
        clf, args.name, post_params
    )

    logger.info("Script finished.")


# Run function
if __name__ == "__main__":
    set_env = "prod"  # either prod or dev

    # Check if this script is run from terminal
    if set_env == "prod":
        # (Prod) script is being run through the terminal
        argvals = None
    else:
        # (Dev) script is not being run through the terminal
        # make sure pwd is the root folder (not in scripts)

        # Command line arguments used for testing
        argvals = (
            "--name ../trained_models/test "
            "--data_path ../input/data/example_binary_titanic.csv "
            "--configuration ../input/configuration/example_binary_titanic.json "
            "--model ebm".split()
        )  # example of passing test params to parser

        # settings
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 10)

    # Get arguments from the CLI
    args = GetArgs("main", argvals)

    # Run main with given arguments
    main(args)
