from sklearn import tree
import pickle
from utils.modelling.performance import *

def trainModel(X_train, y_train, params, model_type, logging):
    """
    Trains a decision tree model using the given training data and parameters.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Target variable.
    params : dict
        Parameters to configure the decision tree.
    model_type : str
        Type of the model. Can be 'regression' or 'classification'.
    logging : logging.Logger
        Logger object to record messages.

    Returns
    -------
    clf : DecisionTreeRegressor or DecisionTreeClassifier
        Trained decision tree model.

    Raises
    ------
    ValueError
        If an unsupported model type is provided.

    Notes
    -----
    If the model type is 'regression', an instance of `DecisionTreeRegressor` is created
    and trained. If the model type is 'classification', an instance of `DecisionTreeClassifier`
    is created and trained. If an unsupported model type is provided, a `ValueError` is raised.

    The trained decision tree model is returned.

    Example:
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> params = {'max_depth': 3, 'min_samples_split': 2}
    >>> clf = trainModel(X_train, y_train, params, 'classification', logging)
    """
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
    """
    Generates and saves a bar plot of feature importance using Plotly.
    
    Parameters:
    -----------
    clf: DecisionTreeClassifier or DecisionTreeRegressor object
        The trained decision tree model.
    given_name: str
        The directory name where the plot should be saved.
    file_type: str {'png', 'html'}
        The type of file in which the plot should be saved.
    logging: logging.Logger object
        A logging object to log information or errors.
        
    Returns:
    --------
    None
    
    Raises:
    -------
    None
    
    Notes:
    ------
    - This function requires the Plotly package.
    - The plot is saved in the given directory with the name 'gini_feature_importance.png' if 'file_type' is 'png', 
      or 'gini_feature_importance.html' if 'file_type' is 'html'.
    """
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