from interpret.glassbox import LinearRegression, LogisticRegression
import logging
from ml2sql.utils.modelling.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LinearRegressionModel(BaseModel):
    """
    Linear/Logistic Regression model implementation.
    
    This class implements the BaseModel interface for the Linear/Logistic Regression
    models from the interpret package.
    """
    
    def __sklearn_tags__(self):
        """
        Get the sklearn tags for this model.
        
        This method is required for compatibility with scikit-learn's estimator interface.
        It delegates to the underlying model if it exists, otherwise returns a default set of tags.
        
        Returns
        -------
        dict
            Dictionary of tags describing the model.
        """
        if self.model is not None and hasattr(self.model, '__sklearn_tags__'):
            return self.model.__sklearn_tags__()
        elif self.model is not None and hasattr(self.model, '_get_tags'):
            # For older scikit-learn versions
            return self.model._get_tags()
        else:
            # Default tags
            return {
                'allow_nan': False,
                'binary_only': False,
                'multilabel': False,
                'multioutput': False,
                'multioutput_only': False,
                'no_validation': False,
                'non_deterministic': False,
                'pairwise': False,
                'preserves_dtype': [],
                'poor_score': False,
                'requires_fit': True,
                'requires_positive_X': False,
                'requires_positive_y': False,
                'requires_y': True,
                'stateless': False,
                'X_types': ['2darray'],
                '_skip_test': False,
                '_xfail_checks': False
            }
    
    @property
    def coef_(self):
        """
        Get the coefficients of the model.
        
        Returns
        -------
        numpy.ndarray
            Coefficients of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'sk_model_'):
            raise AttributeError("Model does not have sk_model_ attribute.")
        return self.model.sk_model_.coef_

    @property
    def intercept_(self):
        """
        Get the intercept of the model.
        
        Returns
        -------
        float or numpy.ndarray
            Intercept of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'sk_model_'):
            raise AttributeError("Model does not have sk_model_ attribute.")
        return self.model.sk_model_.intercept_
    
    def train(self, X_train, y_train, model_type):
        """
        Train a Linear/Logistic Regression model on the given training data.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training data features.
        y_train : pandas.Series
            Training data target.
        model_type : str
            Type of model to train, either 'regression' or 'classification'.

        Returns
        -------
        self : LinearRegressionModel
            The trained model instance.
        """
        self.feature_names = X_train.columns
        
        if model_type == "regression":
            self.model = LinearRegression(**self.params)
            clf_name = "Linear regression"
        elif model_type == "classification":
            self.model = LogisticRegression(**self.params)
            # Hard code classes_
            self.model.classes_ = list(set(y_train))
            clf_name = "Logistic regression"
        else:
            logger.warning("Only regression or classification available")
            raise ValueError("Invalid model_type. Must be 'regression' or 'classification'.")

        self.model.fit(X_train, y_train)
        self.target = y_train.name
        
        logger.info(f"Model non default params:\n {self.model.kwargs}")
        logger.info(f"Trained {clf_name.lower()}")

        return self

    def post_model_plots(self, given_name, file_type):
        """
        Generate and save feature-specific and overall feature importance graphs.

        Parameters
        ----------
        given_name : str
            Name for the output files.
        file_type : str
            Type of file to save the output graphs, either 'png' or 'html'.

        Returns
        -------
        None
        """
        self._feature_explanation_save(given_name, file_type)
    
    def _feature_explanation_save(self, given_name, file_type):
        """
        Save feature-specific and overall feature importance graphs.

        Parameters
        ----------
        given_name : str
            Name for the output files.
        file_type : str
            Type of file to save the output graphs, either 'png' or 'html'.

        Returns
        -------
        None
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        clf_global = self.model.explain_global()

        # Save overall feature importance graph
        plotly_fig = clf_global.visualize()

        if file_type == "png":
            plotly_fig.write_image(
                "{given_name}/1_overall_feature_importance.png".format(
                    given_name=given_name
                )
            )
        elif file_type == "html":
            plotly_fig.write_html(
                "{given_name}/1_overall_feature_importance.html".format(
                    given_name=given_name
                )
            )

        # Save feature specific explanation graphs
        for index, value in enumerate(self.model.feature_names_in_):
            plotly_fig = clf_global.visualize(index)

            # reformatting feature name
            feature_name = self.model.feature_names_in_[index]
            chars = "\\`./ "
            for c in chars:
                if c in feature_name:
                    feature_name = feature_name.replace(c, "_")

            if file_type == "png":
                plotly_fig.write_image(f"{given_name}/explain_{feature_name}.png")
            elif file_type == "html":
                # or as html file
                plotly_fig.write_html(f"{given_name}/explain_{feature_name}.html")

        logger.info(
            "Explanation plots of {n_features} features saved".format(n_features=index + 1)
        )


# For backward compatibility
def trainModel(X_train, y_train, params, model_type):
    """
    Legacy function for backward compatibility.
    
    Creates and trains a Linear/Logistic Regression model directly without using the LinearRegressionModel wrapper.
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training data features.
    y_train : pandas.Series
        Training data target.
    params : dict
        Parameters for the model.
    model_type : str
        Type of model to train, either 'regression' or 'classification'.
        
    Returns
    -------
    clf : LinearRegression or LogisticRegression
        Trained model.
    """
    if model_type == "regression":
        clf = LinearRegression(**params)
        clf_name = "Linear regression"
    elif model_type == "classification":
        clf = LogisticRegression(**params)
        # Hard code classes_
        clf.classes_ = list(set(y_train))
        clf_name = "Logistic regression"
    else:
        logger.warning("Only regression or classification available")
        raise ValueError("Invalid model_type. Must be 'regression' or 'classification'.")

    clf.fit(X_train, y_train)
    
    logger.info(f"Model non default params:\n {clf.kwargs}")
    logger.info(f"Trained {clf_name.lower()}")

    return clf


def featureExplanationSave(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Saves feature-specific and overall feature importance graphs.
    
    Parameters
    ----------
    clf : LinearRegression or LogisticRegression
        Trained model for which to generate feature importance graphs.
    given_name : str
        Name for the output files.
    file_type : str
        Type of file to save the output graphs, either 'png' or 'html'.
        
    Returns
    -------
    None
    """
    model = LinearRegressionModel()
    model.model = clf
    model._feature_explanation_save(given_name, file_type)


def postModelPlots(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Generates and saves plots related to the model.
    
    Parameters
    ----------
    clf : LinearRegression or LogisticRegression
        Trained model for which to generate plots.
    given_name : str
        Name for the output files.
    file_type : str
        Type of file to save the output plots.
        
    Returns
    -------
    None
    """
    featureExplanationSave(clf, given_name, file_type)