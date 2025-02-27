from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
import logging
from ml2sql.utils.modelling.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class EBMModel(BaseModel):
    """
    Explainable Boosting Machine (EBM) model implementation.
    
    This class implements the BaseModel interface for the Explainable Boosting Machine
    model from the interpret package.
    """
    
    def train(self, X_train, y_train, model_type):
        """
        Train an Explainable Boosting Machine (EBM) model on the given training data.

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
        self : EBMModel
            The trained model instance.
        """
        if "feature_names" not in self.params.keys():
            self.params["feature_names"] = X_train.columns
        
        self.feature_names = X_train.columns
        
        if model_type == "regression":
            self.model = ExplainableBoostingRegressor(**self.params)
        elif model_type == "classification":
            self.model = ExplainableBoostingClassifier(**self.params)
        else:
            logger.warning("Only regression or classification available")
            raise ValueError("Invalid model_type. Must be 'regression' or 'classification'.")

        self.model.fit(X_train, y_train)
        self.target = y_train.name
        
        logger.info(f"Model params:\n {self.model.get_params}")
        logger.info("Trained explainable boosting machine")

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
        for index, value in enumerate(self.model.term_features_):
            plotly_fig = clf_global.visualize(index)

            # Formatting feature name
            # if combined feature create combined feature name
            if len(value) == 2:
                feature_name = (
                    f"{self.model.feature_names[value[0]]}_x_{self.model.feature_names[value[1]]}"
                )
            else:
                feature_name = self.model.feature_names[index]

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
    
    Creates and trains an EBMModel instance.
    
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
        
    Returns
    -------
    clf : EBMModel
        Trained EBM model.
    """
    model = EBMModel(params)
    return model.train(X_train, y_train, model_type).model


def featureExplanationSave(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Saves feature-specific and overall feature importance graphs.
    
    Parameters
    ----------
    clf : EBM model
        Trained EBM model for which to generate feature importance graphs.
    given_name : str
        Name for the output files.
    file_type : str
        Type of file to save the output graphs, either 'png' or 'html'.
        
    Returns
    -------
    None
    """
    model = EBMModel()
    model.model = clf
    model._feature_explanation_save(given_name, file_type)


def postModelPlots(clf, given_name, file_type):
    """
    Legacy function for backward compatibility.
    
    Generates and saves plots related to the model.
    
    Parameters
    ----------
    clf : EBM model
        Trained EBM model for which to generate plots.
    given_name : str
        Name for the output files.
    file_type : str
        Type of file to save the output plots.
        
    Returns
    -------
    None
    """
    featureExplanationSave(clf, given_name, file_type)