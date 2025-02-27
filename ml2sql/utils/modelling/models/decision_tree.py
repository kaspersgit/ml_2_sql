from sklearn import tree
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from ml2sql.utils.modelling.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class DecisionTreeModel(BaseModel):
    """
    Decision Tree model implementation.
    
    This class implements the BaseModel interface for the Decision Tree
    model from scikit-learn.
    """
    
    def train(self, X_train, y_train, model_type):
        """
        Train a decision tree model using the given training data and parameters.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Target variable.
        model_type : str
            Type of the model. Can be 'regression' or 'classification'.

        Returns
        -------
        self : DecisionTreeModel
            Trained decision tree model instance.
        """
        self.feature_names = X_train.columns
        
        if model_type == "regression":
            self.model = tree.DecisionTreeRegressor(**self.params)
        elif model_type == "classification":
            self.model = tree.DecisionTreeClassifier(**self.params)
        else:
            logger.warning("Only regression or classification available")
            raise ValueError("Invalid model_type. Must be 'regression' or 'classification'.")

        self.model.fit(X_train, y_train)
        self.target = y_train.name
        
        logger.info(f"Model params:\n {self.model.get_params}")
        logger.info("Trained decision tree")

        return self

    def post_model_plots(self, given_name, file_type):
        """
        Generate and save plots related to the model.
        
        Parameters
        ----------
        given_name : str
            Name for the output files.
        file_type : str
            Type of file to save the output plots.
            
        Returns
        -------
        None
        """
        self._feature_importance_save(given_name, file_type)
        
        if hasattr(self.model, 'classes_'):
            self._all_classes_have_leaf_node()
    
    def _feature_importance_save(self, given_name, file_type):
        """
        Generate and save a bar plot of feature importance using Plotly.

        Parameters
        ----------
        given_name : str
            The directory name where the plot should be saved.
        file_type : str {'png', 'html'}
            The type of file in which the plot should be saved.

        Returns
        -------
        None
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        importance_df = (
            pd.DataFrame(
                {"importance": self.model.feature_importances_, "feature": self.model.feature_names_in_}
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
    
    def _all_classes_have_leaf_node(self):
        """
        Check if all classes are represented by a leaf node in the decision tree classifier.

        Returns
        -------
        bool
            True if all classes are represented by a leaf node, False otherwise.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        if not hasattr(self.model, 'classes_'):
            return True
            
        n_nodes = self.model.tree_.node_count
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        class_values = self.model.tree_.value
        classes = self.model.classes_

        end_leaf_classes = []

        for node_id in range(n_nodes):
            if children_left[node_id] == children_right[node_id]:
                class_name = self.model.classes_[np.argmax(class_values[node_id])]
                end_leaf_classes.append(class_name)

        if len(set(end_leaf_classes)) == len(classes):
            logger.info("All classes are represented by a leaf node\n")
            return True
        else:
            logger.info(
                "{}/{} classes are represented by a leaf node ({})".format(
                    len(set(end_leaf_classes)), len(self.model.classes_), len(end_leaf_classes)
                )
            )
            logger.info(
                "Missing class(es): {}\n".format(set(self.model.classes_) - set(end_leaf_classes))
            )
            return False


# For backward compatibility
def trainModel(X_train, y_train, params, model_type):
    """
    Legacy function for backward compatibility.
    
    Creates and trains a DecisionTreeModel instance.
    
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
    """
    model = DecisionTreeModel(params)
    return model.train(X_train, y_train, model_type).model


def featureImportanceSave(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Generates and saves a bar plot of feature importance using Plotly.
    
    Parameters
    ----------
    clf : DecisionTreeClassifier or DecisionTreeRegressor object
        The trained decision tree model.
    given_name : str
        The directory name where the plot should be saved.
    file_type : str {'png', 'html'}
        The type of file in which the plot should be saved.
        
    Returns
    -------
    None
    """
    model = DecisionTreeModel()
    model.model = clf
    model._feature_importance_save(given_name, file_type)


def allClassesHaveLeafNode(clf):
    """
    Legacy function for backward compatibility.
    
    Check if all classes are represented by a leaf node in a given decision tree classifier.
    
    Parameters
    ----------
    clf : sklearn.tree.DecisionTreeClassifier object
        The decision tree classifier to be checked.
        
    Returns
    -------
    bool
        True if all classes are represented by a leaf node, False otherwise.
    """
    model = DecisionTreeModel()
    model.model = clf
    return model._all_classes_have_leaf_node()


def postModelPlots(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Generates and saves plots related to the model.
    
    Parameters
    ----------
    clf : DecisionTreeClassifier or DecisionTreeRegressor object
        The trained decision tree model.
    given_name : str
        The directory name where the plots should be saved.
    file_type : str
        The type of file in which the plots should be saved.
        
    Returns
    -------
    None
    """
    featureImportanceSave(clf, given_name, file_type)
    if type(clf).__name__ == "DecisionTreeClassifier":
        allClassesHaveLeafNode(clf)