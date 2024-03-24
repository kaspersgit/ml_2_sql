from sklearn import tree
import pandas as pd
import numpy as np
import plotly.express as px
import logging

logger = logging.getLogger(__name__)


def trainModel(X_train, y_train, params, model_type):
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

    """
    if model_type == "regression":
        clf = tree.DecisionTreeRegressor(**params)
    elif model_type == "classification":
        clf = tree.DecisionTreeClassifier(**params)
    else:
        logger.warning("Only regression or classification available")

    clf.fit(X_train, y_train)
    logger.info(f"Model params:\n {clf.get_params}")

    logger.info("Trained decision tree")

    return clf


def featureImportanceSave(clf, given_name, file_type):
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
    importance_df = (
        pd.DataFrame(
            {"importance": clf.feature_importances_, "feature": clf.feature_names_in_}
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    importance_non_zero = importance_df[importance_df["importance"] > 0]
    plotly_fig = px.bar(importance_non_zero, x="importance", y="feature")

    # Update size of figure
    plotly_fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        title="Feature importance",
        width=1000,
        height=800,
    )

    if file_type == "png":
        plotly_fig.write_image(f"{given_name}/gini_feature_importance.png")
    elif file_type == "html":
        plotly_fig.write_html(f"{given_name}/gini_feature_importance.html")

    logger.info("Gini feature importance plot saved")


def allClassesHaveLeafNode(clf):
    """
    Check if all classes are represented by a leaf node in a given decision tree classifier.

    Parameters:
    -----------
    clf : sklearn.tree.DecisionTreeClassifier object
        The decision tree classifier to be checked.

    Returns:
    --------
    bool
        True if all classes are represented by a leaf node, False otherwise.
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    # feature = clf.tree_.feature
    # threshold = clf.tree_.threshold
    class_values = clf.tree_.value
    classes = clf.classes_

    end_leaf_classes = []

    for node_id in range(n_nodes):
        if children_left[node_id] == children_right[node_id]:
            # feature_index = feature[node_id]
            class_name = clf.classes_[np.argmax(class_values[node_id])]
            end_leaf_classes.append(class_name)

    if len(set(end_leaf_classes)) == len(classes):
        logger.info("All classes are represented by a leaf node\n")

        return True
    else:
        logger.info(
            "{}/{} classes are represented by a leaf node ({})".format(
                len(set(end_leaf_classes)), len(clf.classes_), len(end_leaf_classes)
            )
        )
        logger.info(
            "Missing class(es): {}\n".format(set(clf.classes_) - set(end_leaf_classes))
        )

        return False


def postModelPlots(clf, given_name, file_type):
    featureImportanceSave(clf, given_name, file_type)
    allClassesHaveLeafNode(clf)
