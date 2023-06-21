from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import pickle
import random
from sklearn.model_selection import train_test_split
from utils.modelling.performance import *
from utils.modelling.calibration import *

def trainModel(X_train, y_train, params, model_type, logging):
    """
    Trains an Explainable Boosting Machine (EBM) model on the given training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training data features.
    y_train : pandas.Series
        Training data target.
    params : dict
        Parameters for the EBM model.
    model_type : str
        Type of model to train, either 'regression' or 'classification'.
    logging : logging.Logger
        Logger instance for logging information.

    Returns
    -------
    clf : EBM model
        Trained EBM model.

    Raises
    ------
    None

    Notes
    -----
    This function trains an Explainable Boosting Machine (EBM) model on the given training data
    using the provided parameters and model type. The trained model is returned and can be used
    for prediction and feature importance analysis.

    If 'feature_names' is not included in the provided params dictionary, the function will use
    the column names of the X_train DataFrame as feature names.

    If the model_type parameter is not 'regression' or 'classification', an error message is printed
    and a warning is logged.

    The function logs the model parameters and a message indicating that the EBM model has been
    trained successfully.
    """
    if 'feature_names' not in params.keys():
        params['feature_names'] = X_train.columns
    if model_type == 'regression':
        clf = ExplainableBoostingRegressor(**params)
    elif model_type == 'classification':
        clf = ExplainableBoostingClassifier(**params)
    else:
        print('Only regression or classification available')
        logging.warning('Only regression or classification available')

    clf.fit(X_train, y_train)
    logging.info(f'Model params:\n {clf.get_params}')

    print('Trained explainable boosting machine \n')
    logging.info('Trained explainable boosting machine')

    return clf


def featureExplanationSave(clf, given_name, file_type, logging):
    """
    Saves feature-specific and overall feature importance graphs for a given Explainable Boosting Machine (EBM) model.

    Parameters
    ----------
    clf : EBM model
        Trained EBM model for which to generate feature importance graphs.
    given_name : str
        Name for the output files.
    file_type : str
        Type of file to save the output graphs, either 'png' or 'html'.
    logging : logging.Logger
        Logger instance for logging information.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This function generates and saves feature-specific and overall feature importance graphs for a given
    Explainable Boosting Machine (EBM) model. The overall feature importance graph is saved as a png or
    html file, depending on the file_type parameter. The file is saved in a directory with the given_name
    parameter as its name.

    The function also generates and saves feature-specific explanation graphs for each feature group in the
    EBM model. The file is saved in a directory with the given_name parameter as its name. Feature names
    are reformatted by replacing certain characters with underscores.

    The function logs a message indicating how many features have been saved.
    """
    
    clf_global = clf.explain_global()

    # Save overall feature importance graph
    plotly_fig = clf_global.visualize()

    if file_type == 'png':
        plotly_fig.write_image('{given_name}/1_overall_feature_importance.png'.format(given_name=given_name))
    elif file_type == 'html':
        plotly_fig.write_html('{given_name}/1_overall_feature_importance.html'.format(given_name=given_name))

    # Save feature specific explanation graphs
    for index, value in enumerate(clf.feature_groups_):
        plotly_fig = clf_global.visualize(index)

        # reformatting feature name
        feature_name = clf.feature_names[index]
        chars = "\\`./ "
        for c in chars:
            if c in feature_name:
                feature_name = feature_name.replace(c, "_")

        if file_type == 'png':
            plotly_fig.write_image(f'{given_name}/explain_{feature_name}.png')
        elif file_type == 'html':
            # or as html file
            plotly_fig.write_html(f'{given_name}/explain_{feature_name}.html')

    print('Explanation plots of {n_features} features saved'.format(n_features=index+1))
    logging.info('Explanation plots of {n_features} features saved'.format(n_features=index+1))

def postModelPlots(clf, given_name, file_type, logging):
    featureExplanationSave(clf, given_name, file_type, logging)
