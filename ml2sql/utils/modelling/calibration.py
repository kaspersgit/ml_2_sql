# import packages
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


def calibrateModel(model, X, y, method="auto", final_model=False):
    """
    Calibrate a classification model.

    Parameters
    ----------
    model : object
        A trained classification model.
    X : array-like or sparse matrix of shape (n_samples, n_features)
        The input data.
    y : array-like of shape (n_samples,)
        The target values.
    method : str, optional
        The calibration method to use. If 'auto', the method is chosen automatically based on the size of the calibration dataset. If 'sigmoid', the logistic regression method is used. If 'isotonic', the isotonic regression method is used. Default is 'auto'.
    final_model : bool, optional
        If True, a calibrated model and calibration function are returned. Default is False.

    Returns
    -------
    model_cal : object
        A calibrated classifier.
    cal_reg : object, optional
        A calibration function. Returned only if `final_model` is True.
    """
    # set calibration function depending on calibration data size
    if method == "auto":
        if len(X) > 1000:
            method = "isotonic"
            logger.info("Applying isotonic regression as calibration method")
        else:
            method = "sigmoid"
            logger.info("Applying logistic regression as calibration method")

    if final_model:
        if method == "sigmoid":
            y_pred = model.predict_proba(X)[:, 1]
            cal_reg = LogisticRegression().fit(y_pred.reshape(-1, 1), y)

        elif method == "isotonic":
            y_pred = model.predict_proba(X)[:, 1]
            cal_reg = IsotonicRegression().fit(y_pred.reshape(-1, 1), y)

    model_cal = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
    model_cal.fit(X, y)

    # set attributes
    model_cal.feature_names = model.feature_names

    logger.info("Trained calibrated model")

    if final_model:
        return model_cal, cal_reg
    else:
        return model_cal
