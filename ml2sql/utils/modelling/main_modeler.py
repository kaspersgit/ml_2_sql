import joblib
import random
import numpy as np
from typing import Any, Dict, Tuple

from sklearn.model_selection import train_test_split
from ml2sql.utils.modelling.performance import postModellingPlots
from ml2sql.utils.modelling.calibration import calibrateModel

# Algorithms (imported dynamically)
from ml2sql.utils.modelling.models import ebm  # noqa: F401
from ml2sql.utils.modelling.models import decision_tree  # noqa: F401
from ml2sql.utils.modelling.models import l_regression  # noqa: F401

import logging

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: Dict[str, Any],
    model_type: str,
    model_name: str,
):
    """
    Train a machine learning model.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data target.
        model_params (dict): Model hyperparameters.
        model_type (str): Type of model (classification or regression).
        model_name (str): Name of model (EBM, linear regression or decision tree).

    Returns:
        Trained machine learning model.
    """
    try:
        clf = globals()[model_name].trainModel(
            X_train, y_train, model_params, model_type
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    return clf


def predict(clf, X_test: np.ndarray, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained model.

    Args:
        clf: Trained machine learning model.
        X_test (np.ndarray): Test data features.
        model_type (str): Type of model (classification or regression).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing predicted target values and predicted probabilities (for classification models).
    """
    try:
        y_test_pred = clf.predict(X_test)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

    if model_type == "classification":
        try:
            y_test_prob = clf.predict_proba(X_test)
            if len(clf.classes_) == 2:
                y_test_prob = y_test_prob[:, 1]
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            raise
        return y_test_pred, y_test_prob
    else:
        return y_test_pred, None


def make_model(
    given_name: str,
    datasets: Dict[str, Dict[str, np.ndarray]],
    model_name: str,
    model_type: str,
    model_params: Dict[str, Any],
    post_params: Dict[str, Any],
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Train and save a model, and generate performance plots.

    Args:
        given_name (str): The name of the model.
        datasets (dict): A dictionary containing the training and test datasets.
        model_name (str): The name of the model to be used.
        model_type (str): The type of the model (classification or regression).
        model_params (dict): A dictionary containing the hyperparameters of the model.
        post_params (dict): A dictionary containing the postprocessing parameters.

    Returns:
        Tuple[Any, Dict[str, np.ndarray]]: A tuple containing the trained model and post-processing datasets.
    """
    # Unpack datasets
    X_train = datasets["cv_train"]["X"]
    y_train = datasets["cv_train"]["y"]
    X_test = datasets["cv_test"]["X"]
    y_test = datasets["cv_test"]["y"]
    X_all = datasets["final_train"]["X"]
    y_all = datasets["final_train"]["y"]

    # Check if X is a list (CV should be applied in that case)
    if isinstance(X_train, list):
        y_test_pred_list = []
        y_test_prob_list = []

        # Save all trained models in a dictionary
        model_dict = {"test": {}}

        for fold_id, (X_train_fold, y_train_fold) in enumerate(zip(X_train, y_train)):
            logger.info(f"Fold {fold_id} - Train model on test data")

            # Check if model needs to be calibrated
            if post_params["calibration"] != "false":
                X_train_fold, X_cal, y_train_fold, y_cal = train_test_split(
                    X_train_fold,
                    y_train_fold,
                    test_size=0.2,
                    random_state=random.randint(0, 100),
                )
            else:
                X_cal, y_cal = None, None

            # Train the model
            clf = train_model(
                X_train_fold, y_train_fold, model_params, model_type, model_name
            )

            # Save model of this fold in a dict
            model_dict["test"][fold_id] = clf

            if post_params["calibration"] != "false":
                try:
                    clf = calibrateModel(
                        clf,
                        X_cal,
                        y_cal,
                        method=post_params["calibration"],
                        final_model=False,
                    )
                except Exception as e:
                    logger.error(f"Error calibrating model: {e}")
                    raise

            # Discrete predictions
            y_test_pred_list.append(clf.predict(X_test[fold_id]))

            if model_type == "classification":
                # Probability predictions
                y_test_pred, y_test_prob = predict(clf, X_test[fold_id], model_type)
                y_test_prob_list.append(y_test_prob)

        # Merge lists of predictions into one list
        y_test_pred = np.concatenate(y_test_pred_list, axis=0)

        if model_type == "classification":
            # Merge lists of prediction probabilities into one list
            y_test_prob = np.concatenate(y_test_prob_list, axis=0)

    # If just regular train/test split has been applied
    else:
        clf = train_model(X_train, y_train, model_params, model_type, model_name)

        # Discrete prediction
        y_test_pred = clf.predict(X_test)

        if model_type == "classification":
            # Probability prediction
            y_test_pred, y_test_prob = predict(clf, X_test, model_type)

    # Train model one last time on all samples (upsampled)
    logger.info("Train final model on all data")

    # Check if model needs to be calibrated
    if post_params["calibration"] != "false":
        X_all, X_cal, y_all, y_cal = train_test_split(
            X_all, y_all, test_size=0.2, random_state=123
        )

    # Train final model
    clf = train_model(X_all, y_all, model_params, model_type, model_name)

    # Save in dict
    model_dict["final"] = clf

    # Save target column name as part of model
    clf.target = y_all.name
    # Save feature names as part of the model
    clf.feature_names = X_all.columns

    # Save model in pickled format
    filename = f"{given_name}/model/{model_name}_{model_type}.sav"
    try:
        with open(filename, "wb") as f:
            joblib.dump(clf, f)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

    # Train set prediction of final model
    y_all_pred = clf.predict(X_all)

    if model_type == "classification":
        # Probability predictions
        y_all_pred, y_all_prob = predict(clf, X_all, model_type)

    if post_params["calibration"] != "false":
        try:
            cal_clf, cal_reg = calibrateModel(
                clf, X_cal, y_cal, method=post_params["calibration"], final_model=True
            )
        except Exception as e:
            logger.error(f"Error calibrating final model: {e}")
            raise

        # Save model in pickled format
        filename = f"{given_name}/model/ebm_calibrated_{model_type}.sav"
        try:
            with open(filename, "wb") as f:
                joblib.dump(cal_clf, f)
        except Exception as e:
            logger.error(f"Error saving calibrated model: {e}")
            raise

        # Train set prediction of final model
        y_all_pred = cal_clf.predict(X_all)

        if model_type == "classification":
            # Probability predictions
            y_all_pred, y_all_prob = predict(cal_clf, X_all, model_type)

    # Concatenate y_train, X_test, and X_train into single lists
    y_test_concat = np.concatenate(y_test, axis=0)
    y_train_concat = np.concatenate(y_train, axis=0)
    X_test_concat = np.concatenate(X_test, axis=0)
    X_train_concat = np.concatenate(X_train, axis=0)

    post_datasets = {
        "X_test_list": X_test,
        "X_train_list": X_train,
        "X_test_concat": X_test_concat,
        "X_train_concat": X_train_concat,
        "X_all": X_all,
        "y_train_list": y_train,
        "y_train_concat": y_train_concat,
        "y_all": y_all,
        "y_all_pred": y_all_pred,
        "y_test_concat": y_test_concat,
        "y_test_pred": y_test_pred,
        "y_test_list": y_test,
    }

    if model_type == "classification":
        post_datasets["y_all_prob"] = y_all_prob
        post_datasets["y_test_prob"] = y_test_prob
        post_datasets["y_test_prob_list"] = y_test_prob_list

    # Performance and other post modeling plots
    try:
        postModellingPlots(
            model_dict, model_name, model_type, given_name, post_datasets, post_params
        )
    except Exception as e:
        logger.error(f"Error generating post-modeling plots: {e}")
        raise

    return clf


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example usage
    # Load your datasets and model parameters here
    datasets = {...}
    model_name = "ebm"
    model_type = "classification"
    model_params = {...}
    post_params = {...}

    # Train and save the model
    trained_model, post_datasets = make_model(
        "my_model", datasets, model_name, model_type, model_params, post_params
    )
