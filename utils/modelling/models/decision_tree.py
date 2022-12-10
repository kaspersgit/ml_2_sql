from sklearn import tree
import pickle
from utils.modelling.performance import *

def trainModel(X_train, y_train, params, model_type, logging):
    # if 'feature_names' not in params.keys():
    #     params['feature_names'] = X_train.columns
    if model_type == 'regression':
        clf = tree.DecisionTreeRegressor(**params)
    elif model_type == 'classification':
        clf = tree.DecisionTreeClassifier(**params)
    else:
        print('Only regression or classification available')
        logging.warning('Only regression or classification available')

    clf.fit(X_train, y_train)
    logging.info(f'Model params:\n {clf.get_params}')

    print('Trained decision tree \n')
    logging.info('Trained decision tree')

    return clf

def plotTreeStructureSave(clf, given_name):

    plt.figure(figsize=(30,30))

    tree.plot_tree(clf, fontsize=10, feature_names=clf.feature_names_in_, class_names=clf.classes_)
    plt.savefig(f'{given_name}/tree_plot.png')

    print('Tree structure plot saved')


def featureImportanceSave(clf, given_name, file_type, logging):
    importance_df = pd.DataFrame({'importance':clf.feature_importances_, 'feature':clf.feature_names_in_}).sort_values('importance', ascending=True).reset_index(drop=True)
    importance_non_zero = importance_df[importance_df['importance'] > 0]
    plotly_fig = px.bar(importance_non_zero, x='importance', y='feature')

    # Update size of figure
    plotly_fig.update_layout(xaxis_title='Importance', yaxis_title='Feature',
                      title=f'Feature importance',
                      width=1000,
                      height=800)

    if file_type == 'png':
        plotly_fig.write_image(f'{given_name}/gini_feature_importance.png')
    elif file_type == 'html':
        plotly_fig.write_html(f'{given_name}/gini_feature_importance.html')

    print('Gini feature importance plot saved')

def postModelPlots(clf, given_name, file_type, logging):
    featureImportanceSave(clf, given_name, file_type, logging)