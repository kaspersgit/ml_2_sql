# Load packages
import logging
import pandas as pd
import json
from pathlib import Path
import importlib

# Main modelling function
from ml2sql.utils.modelling.main_modeler import make_model

# The translations to SQL (imported dynamically)
from ml2sql.utils.output_scripts import decision_tree_as_code  # noqa: F401
from ml2sql.utils.output_scripts import ebm_as_code  # noqa: F401
from ml2sql.utils.output_scripts import l_regression_as_code  # noqa: F401

from ml2sql.utils.helper_functions.checks import checkInputData
from ml2sql.utils.helper_functions.setup_logger import setup_logger

from ml2sql.utils.helper_functions.config_handling import config_handling
from ml2sql.utils.pre_processing.pre_process import pre_process_kfold


class ModelCreator:
    """
    Class for creating and training machine learning models.
    
    This class encapsulates the process of loading data, configuring models,
    preprocessing data, training models, and saving the trained models along
    with their SQL representations.
    """
    
    def __init__(self, data_path, config_path, model_name, project_name):
        """
        Initialize the ModelCreator with the given parameters.
        
        Parameters
        ----------
        data_path : str or Path
            Path to the CSV file containing the data.
        config_path : str or Path
            Path to the JSON file containing the configuration.
        model_name : str
            Name of the model to train.
        project_name : str or Path
            Name of the project.
        """
        # Convert to Path objects
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.project_name = Path(project_name)
        self.model_name = model_name
        
        # Set logger
        setup_logger(self.project_name / "logging.log")
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Script input arguments: \ndata_path: {self.data_path} \nconfig_path: {self.config_path} \nmodel_name: {self.model_name} \nproject_name: {self.project_name}"
        )
        
        # Initialize attributes
        self.data = None
        self.configuration = None
        self.target_col = None
        self.feature_cols = None
        self.model_params = None
        self.pre_params = None
        self.post_params = None
        self.model_type = None
        self.datasets = None
        self.clf = None
    
    def load_data(self):
        """
        Load data from the CSV file.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        self.logger.info(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(
            self.data_path,
            keep_default_na=False,
            na_values=["", "N/A", "NULL", "None", "NONE"],
        )
        return self
    
    def load_configuration(self):
        """
        Load configuration from the JSON file.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        self.logger.info(f"Loading configuration from {self.config_path}...")
        with self.config_path.open() as json_file:
            self.configuration = json.load(json_file)
        
        # Handle the configuration file
        self.target_col, self.feature_cols, self.model_params, self.pre_params, self.post_params = config_handling(
            self.configuration, self.data
        )
        
        self.logger.info(f"Configuration file content: {self.configuration}")
        return self
    
    def check_input_data(self):
        """
        Perform input checks on the data.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        checkInputData(self.data, self.configuration)
        return self
    
    def determine_model_type(self):
        """
        Determine the type of model to train based on the target column.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        if (self.data[self.target_col].dtype == "float") or (
            (self.data[self.target_col].dtype == "int") and (self.data[self.target_col].nunique() > 10)
        ):
            self.model_type = "regression"
        else:
            self.model_type = "classification"
        
        self.logger.info(f"Target column has {self.data[self.target_col].nunique()} unique values")
        self.logger.info(f"This problem will be treated as a {self.model_type} problem")
        return self
    
    def preprocess_data(self):
        """
        Preprocess the data.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        self.logger.info("Preprocessing data...")
        self.datasets = pre_process_kfold(
            self.project_name,
            self.data,
            self.target_col,
            self.feature_cols,
            model_name=self.model_name,
            model_type=self.model_type,
            pre_params=self.pre_params,
            post_params=self.post_params,
            random_seed=42,
        )
        return self
    
    def train_model(self):
        """
        Train the model.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        self.logger.info(f"Training {self.model_name} model...")
        self.clf = make_model(
            self.project_name,
            self.datasets,
            model_name=self.model_name,
            model_type=self.model_type,
            model_params=self.model_params,
            post_params=self.post_params,
        )
        return self
    
    def save_model_as_sql(self):
        """
        Create SQL version of model and save it.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        self.logger.info(f"Saving {self.model_name} model and its SQL representation...")
        # Import the appropriate module dynamically
        module_name = f"ml2sql.utils.output_scripts.{self.model_name}_as_code"
        module = importlib.import_module(module_name)
        
        # Call the save_model_and_extras function
        module.save_model_and_extras(self.clf, self.project_name, self.post_params)
        
        self.logger.info("Script finished.")
        return self
    
    def create(self):
        """
        Create and train the model.
        
        This method chains all the steps of the model creation process.
        
        Returns
        -------
        self : ModelCreator
            The ModelCreator instance.
        """
        return (
            self.load_data()
            .load_configuration()
            .check_input_data()
            .determine_model_type()
            .preprocess_data()
            .train_model()
            .save_model_as_sql()
        )


def modelcreater(data_path, config_path, model_name, project_name):
    """
    Main function to train machine learning models and save the trained model along with its SQL representation.
    
    This is a wrapper around the ModelCreator class for backward compatibility.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the CSV file containing the data.
    config_path : str or Path
        Path to the JSON file containing the configuration.
    model_name : str
        Name of the model to train.
    project_name : str or Path
        Name of the project.
    """
    creator = ModelCreator(data_path, config_path, model_name, project_name)
    creator.create()
    return creator.clf