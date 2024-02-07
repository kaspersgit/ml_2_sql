from interpret.glassbox import DecisionListClassifier
from utils.modelling.performance import *

def trainModel(X_train, y_train, params, model_type, logging):
    """
    Train a decision list classifier on the given training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The input features for training the model.
    y_train : pandas.Series
        The target variable for training the model.
    params : dict
        A dictionary containing the parameters for the decision list classifier.
    model_type : str
        The type of model to train. Only 'classification' is currently supported.
    logging : logging.Logger
        A logger object to record messages during the training process.

    Returns
    -------
    clf : DecisionListClassifier
        The trained decision list classifier.

    Raises
    ------
    ValueError
        If an unsupported model type is provided.

    """
    if 'feature_names' not in params.keys():
        params['feature_names'] = X_train.columns
    if model_type == 'classification':
        clf = DecisionListClassifier(**params)
    else:
        print('Only classification available')
        logging.warning('Only classification available')

    clf.fit(X_train, y_train)
    print('Trained decision rule \n')
    logging.info('Trained decision rule')

    return clf

def featureImportanceSave(clf, given_name, logging):
    """
    Save the feature importance graphs for the trained model.

    Parameters
    ----------
    clf : object
        The trained decision list classifier object.

    given_name : str
        The name of the file to which the feature importance graphs will be saved.

    logging : logging.Logger
        The logging object.

    Returns
    -------
    None

    Notes
    -----
    The feature importance graphs include the overall feature importance graph and the feature specific explanation graphs.

    The feature importance graphs are saved in the following files:
        - overall feature importance graph - {given_name}/decisions_rule.html
        - feature specific explanation graphs - {given_name}/explain_{feature_name}.html

    """

    clf_global = clf.explain_global()

    # Save overall feature importance graph
    # use clf.feat_rule_map_ to show nr of rules feature is used in

    # Save list of all rules
    plotly_fig = clf_global.visualize()

    with open(f"{given_name}/decisions_rule.html", "w") as file:
        file.write(plotly_fig)

    # Save feature specific explanation graphs
    for index, value in enumerate(clf.feature_names):
        plotly_fig = clf_global.visualize(index)

        # reformatting feature name
        feature_name = clf.feature_names[index]
        chars = "\\`./ "
        for c in chars:
            if c in feature_name:
                feature_name = feature_name.replace(c, "_")

        with open(f"{given_name}/explain_{feature_name}.html", "w") as file:
            file.write(plotly_fig)

    print(f'Decision lists for {index+1} features saved')
    logging.info(f'Decision lists for {index+1} features saved')

def postModelPlots(clf, given_name, file_type, logging):
    featureImportanceSave(clf, given_name, logging)
