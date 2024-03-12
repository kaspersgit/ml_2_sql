import joblib
import pandas as pd
import logging
import sys
import numpy as np

from utils.helper_functions.parsing_arguments import GetArgs
from utils.modelling.performance import (
    plotClassificationCurve,
    plotCalibrationCurve,
    plotProbabilityDistribution,
    plotYhatVsYSave,
    plotQuantileError,
    regressionMetricsTable,
    plotConfusionMatrix,
)


def apply_model(args):
    # Set destination
    destination = args.destination_path

    # Set Logger
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        filename=destination + "/logging.log",
        level=logging.DEBUG,
    )
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # ignore matplotlibs font warnings

    # Load in model
    model = joblib.load(open(args.model_path, "rb"))

    # Load in data
    df = pd.read_csv(args.data_path)

    target_col = model.target
    feature_cols = [x for x in model.feature_names if " x " not in x]

    # Perform inference
    y_true = df[target_col]
    X = df[feature_cols]

    y_pred = model.predict(X)

    # Create performance metrics graphs
    if hasattr(model, "classes_"):
        # If classification make probability prediction
        y_prob = model.predict_proba(X)

        if len(model.classes_) == 2:
            # for binary only take class one probabilities
            y_prob = y_prob[:, 1]

            # Threshold dependant
            plotConfusionMatrix(
                destination,
                y_true,
                y_prob,
                y_pred,
                file_type="html",
                data_type="test",
                logging=logging,
            )

            # Also create pr curve for class 0
            y_neg = np.array([1 - j for j in list(y_true)])
            y_prob_neg = np.array([1 - j for j in list(y_prob)])

            # Threshold independant
            plotClassificationCurve(
                destination,
                y_true,
                y_prob,
                curve_type="roc",
                data_type="test",
                logging=logging,
            )

            plotClassificationCurve(
                destination,
                y_true,
                y_prob,
                curve_type="pr",
                data_type="test_class1",
                logging=logging,
            )
            plotClassificationCurve(
                destination,
                y_neg,
                y_prob_neg,
                curve_type="pr",
                data_type="test_class0",
                logging=logging,
            )

            plotClassificationCurve(
                destination,
                y_true,
                y_prob,
                curve_type="pr",
                data_type="test_data_class1",
                logging=logging,
            )
            plotClassificationCurve(
                destination,
                y_neg,
                y_prob_neg,
                curve_type="pr",
                data_type="test_data_class0",
                logging=logging,
            )

            plotCalibrationCurve(
                destination, y_true, y_prob, data_type="test", logging=logging
            )

            plotProbabilityDistribution(
                destination, y_true, y_prob, data_type="test", logging=logging
            )

        # If multiclass classification
        elif len(model.classes_) > 2:
            # loop through classes
            for c in model.classes_:
                # creating a list of all the classes except the current class
                other_class = [x for x in model.classes_ if x != c]

                # Get index of selected class in model.classes_
                class_index = list(model.classes_).index(c)

                # marking the current class as 1 and all other classes as 0
                y_ova = [0 if x in other_class else 1 for x in y_true]
                y_prob_ova = [x[class_index] for x in y_prob]

                # Threshold independant
                plotClassificationCurve(
                    destination,
                    y_ova,
                    y_prob_ova,
                    curve_type="roc",
                    data_type=f"test_class_{c}",
                    logging=logging,
                )

                plotClassificationCurve(
                    destination,
                    y_ova,
                    y_prob_ova,
                    curve_type="pr",
                    data_type=f"test_class_{c}",
                    logging=logging,
                )

                plotCalibrationCurve(
                    destination,
                    y_ova,
                    y_prob_ova,
                    data_type=f"test_class_{c}",
                    logging=logging,
                )

                plotProbabilityDistribution(
                    destination,
                    y_ova,
                    y_prob_ova,
                    data_type=f"test_class_{c}",
                    logging=logging,
                )

    # if regression
    else:
        plotYhatVsYSave(destination, y_true, y_pred, data_type="test", logging=logging)

        plotQuantileError(
            destination, y_true, y_pred, data_type="test", logging=logging
        )

        regressionMetricsTable(
            destination, y_true, y_pred, X, data_type="test", logging=logging
        )

    # Exit with code 0 for success
    sys.exit(0)

if __name__ == "__main__":
    set_env = "prod"  # either prod or dev

    # Check if this script is run from terminal
    if set_env == "prod":
        # (Prod) script is being run through the terminal
        argvals = None
    else:
        # (Dev) script is not being run through the terminal

        # Command line arguments used for testing
        argvals = (
            "--model_path ../trained_models/20230622_datathon_ebm/model/ebm_regression.sav "
            "--data_path ../input/data/20230525_cq_datathon.csv "
            "--destination_path ../trained_models/20230622_datathon_ebm/tested_datasets/20230525_cq_datathon.csv".split()
        )  # example of passing test params to parser

        # settings
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 10)

    # Get arguments from the CLI
    args = GetArgs("test_model", argvals)

    # Run main with given arguments
    apply_model(args)
