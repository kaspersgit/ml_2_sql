import joblib
import random
import numpy as np 

from sklearn.model_selection import train_test_split
from utils.modelling.performance import postModellingPlots
from utils.modelling.calibration import calibrateModel

# The actual algorithms (grey as we refer to them dynamically)
from utils.modelling.models import ebm # noqa: F401
from utils.modelling.models import decision_rule # noqa: F401
from utils.modelling.models import decision_tree # noqa: F401
from utils.modelling.models import l_regression # noqa: F401

def make_model(given_name, datasets, model_name, model_type, model_params, post_params, logging):
    """
    Train and save a model, and generate performance plots.

    Parameters
    ----------
    given_name : str
        The name of the model.
    datasets : dict
        A dictionary containing the training and test datasets.
    model_name : str
        The name of the model to be used.
    model_type : str
        The type of the model (classification or regression).
    model_params : dict
        A dictionary containing the hyperparameters of the model.
    post_params : dict
        A dictionary containing the postprocessing parameters.
    logging : logger
        A logger object used for logging.

    Returns
    -------
    A trained machine learning model that can be used to make predictions on new data.

    """
    # unpack datasets
    X_train = datasets['cv_train']['X']
    y_train = datasets['cv_train']['y']
    X_test = datasets['cv_test']['X']
    y_test = datasets['cv_test']['y']
    X_all = datasets['final_train']['X']
    y_all = datasets['final_train']['y']


    # check if X is a list (CV should be applied in that case)
    if isinstance(X_train, list):

        y_test_pred_list = list()
        y_test_prob_list = list()

        # Save all trained models in a dictionary
        clf_dict = {'test': {}}

        for fold_id in range(len(X_train)):
            logging.info(f'Fold {fold_id} \nTrain model on test data')

            # Check if model needs to be calibrated
            if post_params['calibration'] != 'false':
                X_slice_train, X_cal, y_slice_train, y_cal = train_test_split(X_train[fold_id],y_train[fold_id], test_size=0.2,random_state=random.randint(0,100))
            else:
                X_slice_train, y_slice_train = X_train[fold_id], y_train[fold_id]

            # Train the model
            clf = globals()[model_name].trainModel(X_slice_train, y_slice_train, model_params, model_type, logging)

            # Save model of this fold in a dict
            clf_dict['test'][fold_id] = clf

            if post_params['calibration'] != 'false':
                clf = calibrateModel(clf, X_cal, y_cal, logging, method=post_params['calibration'], final_model=False)

            # discrete predictions
            y_test_pred_list.append(clf.predict(X_test[fold_id]))

            if model_type == 'classification':
            # probability predictions
                # Binary classification
                if len(clf.classes_) == 2:
                    y_test_prob_list.append(clf.predict_proba(X_test[fold_id])[:,1])
                elif len(clf.classes_) > 2:
                    y_test_prob_list.append(clf.predict_proba(X_test[fold_id]))

        # Merge list of prediction lists into one list
        y_test_pred = np.concatenate(y_test_pred_list, axis=0)

        if model_type == 'classification':
            # Merge list of prediction probabilities lists into one list
            y_test_prob = np.concatenate(y_test_prob_list, axis=0)

    # If just regular train/test split has been applied
    else:
        clf = globals()[model_name].trainModel(X_train, y_train, model_params, model_type, logging)

        # discrete prediction
        y_test_pred = clf.predict(X_test)

        if model_type == 'classification':
            # probability prediction
            y_test_prob = clf.predict_proba(X_test)[:,1]

    # train model one last time on all samples (upsampled)
    logging.info('Train final model on all data')

    # Check if model needs to be calibrated
    if post_params['calibration'] != 'false':
        X_all, X_cal, y_all, y_cal = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

    # Train final model
    clf = globals()[model_name].trainModel(X_all, y_all, model_params, model_type, logging)

    # Save in dict
    clf_dict['final'] = clf

    # Save target column name as part of model
    clf.target = y_all.name

    # Save model in pickled format
    filename = f'{given_name}/model/{model_name}_{model_type}.sav'
    joblib.dump(clf, open(filename, 'wb'))

    # train set prediction of final model
    y_all_pred = clf.predict(X_all)

    if model_type == 'classification':
    # probability predictions
        # Binary classification
        if len(clf.classes_) == 2:
            y_all_prob = clf.predict_proba(X_all)[:,1]
        elif len(clf.classes_) > 2:
            y_all_prob = clf.predict_proba(X_all)


    if post_params['calibration'] != 'false':
        cal_clf, cal_reg = calibrateModel(clf, X_cal, y_cal, logging, method=post_params['calibration'], final_model=True)

        # Save model in pickled format
        filename = given_name + '/model/ebm_calibrated_{model_type}.sav'.format(model_type=model_type)
        joblib.dump(cal_clf, open(filename, 'wb'))

        # train set prediction of final model
        y_all_pred = cal_clf.predict(X_all)

        if model_type == 'classification':
            # probability predictions
            # Binary classification
            if len(clf.classes_) == 2:
                y_all_prob = clf.predict_proba(X_all)[:, 1]
            elif len(clf.classes_) > 2:
                y_all_prob = clf.predict_proba(X_all)

    # concat y train, X test and X train to single lists
    y_test_concat = np.concatenate(y_test, axis=0)
    y_train_concat = np.concatenate(y_train, axis=0)
    X_test_concat = np.concatenate(X_test, axis=0)
    X_train_concat = np.concatenate(X_train, axis=0)

    post_datasets = {
        'X_test_list': X_test,
        'X_train_list': X_train,
        'X_test_concat': X_test_concat,
        'X_train_concat': X_train_concat,
        'X_all': X_all,
        'y_train_list': y_train,
        'y_train_concat': y_train_concat,
        'y_all': y_all,
        'y_all_pred': y_all_pred,
        'y_test_concat': y_test_concat,
        'y_test_pred': y_test_pred,
        'y_test_list': y_test,
    }

    if model_type == 'classification':
        post_datasets['y_all_prob'] = y_all_prob
        post_datasets['y_test_prob'] = y_test_prob
        post_datasets['y_test_prob_list'] = y_test_prob_list

    # Performance and other post modeling plots
    postModellingPlots(clf_dict, model_name, model_type, given_name, post_datasets, post_params, logging)

    return clf
