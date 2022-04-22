# import packages
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def calibrateModel(model, X, y, logging, method='auto', final_model=False):
    # set calibration function depending on calibration data size
    if method == 'auto':
        if len(X) > 1000:
            method = 'isotonic'
            logging.info('Applying isotonic regression as calibration method')
        else:
            method = 'sigmoid'
            logging.info('Applying logistic regression as calibration method')

    if final_model:
        if method == 'sigmoid':
            y_pred = model.predict_proba(X)[:,1]
            cal_reg = LogisticRegression().fit(y_pred.reshape(-1, 1), y)

        elif method == 'isotonic':
            y_pred = model.predict_proba(X)[:,1]
            cal_reg = IsotonicRegression().fit(y_pred.reshape(-1, 1), y)

    model_cal = CalibratedClassifierCV(base_estimator=model, method=method, cv='prefit')
    model_cal.fit(X, y)

    # set attributes
    model_cal.feature_names = model.feature_names

    print('Trained calibrated model')
    logging.info('Trained calibrated model')

    # plotCalibrationCurve(given_name='trained_models/20220303_henkie', y_true=y_test[part_id], y_prob=y_pred,
    #                      prediction_type='uncal', data_type='')
    #
    # y_pred_cal = iso_reg.predict(y_pred)
    #
    # plotCalibrationCurve(given_name='trained_models/20220303_henkie', y_true=y_test[part_id], y_prob=y_pred_cal,
    #                      prediction_type='cal', data_type='')

    if final_model:
        return model_cal, cal_reg
    else:
        return model_cal
