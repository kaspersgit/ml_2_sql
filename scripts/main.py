# Load packages
import logging
import pandas as pd
import json
import sys

# Main modelling function
from utils.modelling.main_modeler import make_model

# The translations to SQL (grey as we refer to them dynamically)
from utils.output_scripts import decision_tree_as_code  # noqa: F401
from utils.output_scripts import decision_rule_as_code  # noqa: F401
from utils.output_scripts import ebm_as_code  # noqa: F401
from utils.output_scripts import l_regression_as_code  # noqa: F401

from utils.helper_functions.checks import checkInputDataHard

from utils.helper_functions.config_handling import config_handling
from utils.helper_functions.parsing_arguments import GetArgs
from utils.pre_processing.pre_process import pre_process_kfold


def main(args):
    # get given name from the first given argument
    given_name = args.name

    # set logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(given_name + "/logging.log"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # ignore matplotlibs font warnings
    logging.info(f"Script input arguments: \n{args}")

    # Load in data
    data = pd.read_csv(
        args.data_path, keep_default_na=False, na_values=["", "N/A", "NULL"]
    )

    # Get configuration file
    with open(args.configuration) as json_file:
        configuration = json.load(json_file)

    # get model name
    model_name = args.model_name

    # Handle the configuration file
    target_col, feature_cols, model_params, pre_params, post_params = config_handling(
        configuration, data, logging
    )

    # Log parameters
    logging.info(f"Configuration file content: \n{configuration}")

    # Perform some basic checks
    checkInputDataHard(data, configuration, logging)

    # set model type based on target value
    if (data[target_col].dtype == "float") | (
        (data[target_col].dtype == "int") & (data[target_col].nunique() > 10)
    ):
        model_type = "regression"
    else:
        model_type = "classification"

        logging.info(f"Target column has {data[target_col].nunique()} unique values")

    logging.info(
        "This problem will be treated as a {model_type} problem".format(
            model_type=model_type
        )
    )

    # pre process data
    datasets = pre_process_kfold(
        given_name,
        data,
        target_col,
        feature_cols,
        model_name=model_name,
        model_type=model_type,
        logging=logging,
        pre_params=pre_params,
        post_params=post_params,
        random_seed=random_seed,
    )

    # train decision tree and figures and save them
    clf = make_model(
        given_name,
        datasets,
        model_name=model_name,
        model_type=model_type,
        model_params=model_params,
        post_params=post_params,
        logging=logging,
    )

    # Create SQL version of model and save it
    globals()[model_name + "_as_code"].save_model_and_extras(
        clf, given_name, post_params, logging
    )

    # Exit with code 0 for success
    sys.exit(0)


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

    # Set random seed
    random_seed = 42

    # Get arguments from the CLI
    args = GetArgs("main", argvals)

    # Run main with given arguments
    main(args)
