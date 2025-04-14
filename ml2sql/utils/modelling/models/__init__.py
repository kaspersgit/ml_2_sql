# Import base model class
from ml2sql.utils.modelling.models.base_model import BaseModel

# Import model implementations
from ml2sql.utils.modelling.models.ebm import EBMModel
from ml2sql.utils.modelling.models.decision_tree import DecisionTreeModel
from ml2sql.utils.modelling.models.l_regression import LinearRegressionModel

# For backward compatibility, also import the functional interfaces
from ml2sql.utils.modelling.models.ebm import trainModel as ebm_trainModel
from ml2sql.utils.modelling.models.decision_tree import trainModel as decision_tree_trainModel
from ml2sql.utils.modelling.models.l_regression import trainModel as l_regression_trainModel

# Define the model classes available
MODEL_CLASSES = {
    'ebm': EBMModel,
    'decision_tree': DecisionTreeModel,
    'l_regression': LinearRegressionModel
}

__all__ = [
    'BaseModel',
    'EBMModel',
    'DecisionTreeModel',
    'LinearRegressionModel',
    'MODEL_CLASSES',
    'ebm_trainModel',
    'decision_tree_trainModel',
    'l_regression_trainModel'
]