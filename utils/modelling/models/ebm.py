from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import pickle
import random
from sklearn.model_selection import train_test_split
from utils.modelling.performance import *
from utils.modelling.calibration import *

def trainModel(X_train, y_train, params, model_type, logging):
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
