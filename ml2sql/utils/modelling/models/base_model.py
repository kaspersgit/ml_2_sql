import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the ml2sql package.
    
    This class defines the common interface that all model implementations
    should follow, ensuring consistency across different model types.
    """
    
    def __init__(self, params=None):
        """
        Initialize the model with the given parameters.
        
        Parameters
        ----------
        params : dict, optional
            Parameters for the model.
        """
        self.params = params or {}
        self.model = None
        self.feature_names = None
        self.target = None
    
    @abstractmethod
    def train(self, X_train, y_train, model_type):
        """
        Train the model on the given data.
        
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
        self : BaseModel
            The trained model instance.
        """
        pass
    
    @abstractmethod
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
        pass
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to make predictions on.
            
        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using the trained model.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to make predictions on.
            
        Returns
        -------
        numpy.ndarray
            Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions.")
        return self.model.predict_proba(X)
    
    @property
    def classes_(self):
        """
        Get the classes of the model.
        
        Returns
        -------
        numpy.ndarray
            Classes of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'classes_'):
            raise AttributeError("Model does not have classes.")
        return self.model.classes_
    
    @property
    def feature_importances_(self):
        """
        Get the feature importances of the model.
        
        Returns
        -------
        numpy.ndarray
            Feature importances of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature importances.")
        return self.model.feature_importances_
    
    @property
    def feature_names_in_(self):
        """
        Get the feature names used by the model.
        
        Returns
        -------
        numpy.ndarray
            Feature names used by the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, 'feature_names_in_'):
            return self.feature_names
        return self.model.feature_names_in_
    
    def get_params(self, deep=True):
        """
        Get the parameters of the model.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        dict
            Parameters of the model.
        """
        if self.model is None:
            return self.params
        if hasattr(self.model, 'get_params'):
            return self.model.get_params(deep)
        return self.params
    
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