# Machine learning to SQL
![GitHub Repo stars](https://img.shields.io/github/stars/kaspersgit/ml_2_sql?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/kaspersgit/ml_2_sql?style=flat-square)
![interpret](https://img.shields.io/badge/interpret-v0.5.1-blue)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![GitHub](https://img.shields.io/github/license/kaspersgit/ml_2_sql?style=flat-square)

# Table of Contents

<img src="https://github.com/kaspersgit/ml_2_sql/blob/main/docs/media/ml2sql_logo.png?raw=true" align="right"
     alt="ML2SQL">

1. [What is it?](#what-is-it)
2. [Getting Started](#getting-started)
3. [Input](#input)
4. [Output](#output)
5. [Remarks](#remarks)
6. [Troubleshooting](#troubleshooting)

<br>

# What is it?
An automated machine learning tool which trains, graphs performance and saves the model in SQL. Using interpretable ML models (from [interpretml](https://github.com/interpretml/interpret/)) to train models which are explainable and interpretable, so called 'glassbox' models. With the outputted model in SQL format which can be used to put a model in 'production' in an SQL environment.
This tool can be used by anybody, but is aimed for people who want to do a quick analysis and/or deploy a model in an SQL system. 

<center><img src="https://github.com/kaspersgit/ml_2_sql/blob/main/docs/media/ml2sql_demo.gif?raw=true"
     alt="ML2SQL_demo" height=400 width=600></center>

## Philosophy:
- For a quick analysis: 
  - Automated training and model performance tested
  - Feature correlations, feature importance and model performance metrics
  - EBM gives more insights into feature importance then any other model
- For model deployment in SQL:
  - Output model in SQL code
  - EBM, Decision Tree and linear/logistic regression
- Explainable Boosting Machine (EBM) on par with other boosted methods while fully explainable

## Note
- Limited to 3 models to choose from: 
  - [Explainable Boostin Machine](https://interpret.ml/docs/ebm.html) 
  - [Linear/Logistic regression](https://interpret.ml/docs/lr.html)
  - [Decision tree](https://interpret.ml/docs/dt.html)

</br>

# Getting started
<details> 
<summary><strong>Installation</strong></summary>
<br>

  1. Make sure you have python >= 3.8 and git installed
  2. Clone Github repo to your local machine and cd into folder, run:
      ```
      git clone git@github.com:kaspersgit/ml_2_sql.git
      cd ml_2_sql
      ```
  3. Create virtual environment and install packages, run: 
        
        Windows:
        ```
        python -m venv .ml2sql
        .ml2sql/Scripts/python -m pip install -r docs/requirements.txt
        ```
        
        Mac/Linux:
        ```
        python3 -m venv .ml2sql
        .ml2sql/bin/python -m pip install -r docs/requirements.txt
        ```
  4. Wait until all packages are installed (could take a few minutes)
  5. You are ready to go (the virtual env does not need to be activated to use this tool)

<br>
</details> 
<details> 
<summary><strong>Quick Demo</strong></summary>
<br>

  1. In the terminal in the root of this folder run: 
      - `python3 run.py` (Mac/Linux)
      - `python run.py` (Windows)
  2. Follow the instructions on screen by selecting the example data and similarly named config file
  3. Check the output in the newly created folder

<br>
</details> 
<details> 
<summary><strong>Quick Usage with Your Own Data</strong></summary>
<br>

  1. Save csv file containing target and all features in the `input/data/` folder (more info on [input data](#data))
  2. In the terminal in the root of this folder run: 
      - `python3 run.py` (Mac/Linux)
      - `python run.py` (Windows)
  3. Select your CSV file
  4. Select `Create a new config` and choose `Automatic` option (a config file will be made and can be edited later) (more info on [config json](#configuration-json))
  5. Select newly created config
  6. Choose a model (EBM is advised)
  7. Give a name for this model
  8. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
  9. The `.sql` file in the `model` folder will contain a SQL written model  

<br>
</details> 
<details> 
<summary><strong>Testing a Trained Model on a New Dataset</strong></summary>
<br>

  1. Make sure the new dataset has the same variables as the dataset the model was trained on (same features and target)
  2. Save dataset in the `input/data/` folder (more info on [input data](#data))
  3. In the terminal in the root of this folder run: 
      - `python3 check_model.py` (Mac/Linux)
      - `python check_model.py` (Windows)
  4. Follow the instructions on screen
  5. The output will be saved in the folder `trained_models/<selected_model>/tested_datasets/<selected_dataset>/`

<br>
</details>
</br>

# Input
## Data
The csv file containing the data has to fulfill some basic assumptions:
- Save the .csv file in the `input/data` folder
- Target column should have more than 1 unique value
- For binary classification (target with 2 unique values) these values should be 0 and 1
- File name should be .csv and not consist of any spaces

#### Additional information
- EBM can handle categorical values (these will be excluded when choosing decision tree or linear/logistic regression)
- EBM can handle missing values

## Configuration json ([example](https://github.com/kaspersgit/ml_2_sql/blob/main/input/configuration/example_binary_titanic.json))
This file will inform the script which column is the target, which are the features and several other parameters for pre and post training.
You can copy and edit a config file from the already existing example in `input/configuration/` or select `Create a new config` file in the second step 
when running the `run.py` file.

Configs are saved in `input/configuration/`.

<details> 
<summary><strong>Configuration file content</strong></summary>

  ### features
  List with names of the columns which should be used as features

  ### model_params
  Dictionary of parameters that can be used with model of choice (optional). Check the model's documentation:
  - EBM ([model documentation](https://interpret.ml/docs/ebm.html))
  - Linear/Logistic regression ([model documentation](https://interpret.ml/docs/lr.html))
  - Decision tree ([model documentation](https://interpret.ml/docs/dt.html))

  `sql_split` options:
  - `false`, outputs the SQL model as one SELECT statement, using column aliases within the same select statement
  - `true`, outputs the SQL model as several CTEs, this can be used if column aliases can't be referenced within the same SELECT statement

  `sql_decimals` options:
  - Any whole positive number, rounds the 'scores' in the SQL file to X decimal places. Can be lowered to avoid any data type overflow problems, but will decrease precision.

  `file_type` options (optional):
  - `png`, output of features importance graphs will be static .png (smaller file).
  - `html`, output of features importance graphs will be dynamic .html (bigger file and opens in browser).

  ### pre_params
  `cv_type` options (optional):
  - `timeseriesplit`, perform 5 fold timeseries split ([sklearn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html))
  - any other value, perform 5 fold stratified cross validation

  `max_rows` options:
  - Any kind of whole positive number, will limit the data set in order to train faster (as simple as that)

  `time_sensitive_column` options (optional):
  - Name of date column to do the time serie split over
    - used when `cv_type = timeseriesplit`  

  ### target
  Name of target column (required)

</details> 
</br>

# Output ([example](https://github.com/kaspersgit/ml_2_sql/tree/main/trained_models/example_titanic))
The output consists of 4 parts:
- Correlation matrix of the input features
- Feature importance graphs (model specific)
- Model performance graphs
- The model itself in pickled and SQL form

## Correlation matrices 
Can be found in the created model's folder under `/feature_info`

### Pearson Correlation Matrix (Numerical Features)
- A Pearson correlation matrix for numerical features in the input data.
- Visualized as a clustermap and saved as `numeric_clustermap.png` or `numeric_clustermap.html`.

### Cramer's V Correlation Matrix (Categorical Features)
- A Cramer's V correlation matrix for categorical features (object, category, boolean) in the input data.
- Visualized as a clustermap and saved as `categorical_clustermap.png` or `categorical_clustermap.html`.


## Feature importance
Can be found in the created model's folder under `/feature_importance`

### For EBM and logistic/linear regression
- an overview of the top important features
- seperate feature importance graph per feature

### For Decision tree
- graph with gini index

## Model performance
Can be found in the created model's folder under `/performance`

### For Classification Models:

1. Confusion Matrix
- A confusion matrix is plotted and saved in both static (PNG) and interactive (HTML) formats for binary classification problems.
- For multiclass classification, separate confusion matrices are plotted for each class.
2. ROC Curve and Precision-Recall Curve
- The tool plots the Receiver Operating Characteristic (ROC) curve and Precision-Recall curve for binary classification problems.
- For multiclass classification, these curves are plotted for each class versus the rest.
3. Calibration Plot
- A calibration plot (reliability curve) is generated to assess the calibration of the predicted probabilities.
4. Probability Distribution Plot
- A probability distribution plot is created to visualize the distribution of predicted probabilities for the positive and negative classes (binary classification) or for each class (multiclass classification).

### For Regression Models:

1. Scatter Plot of Predicted vs. True Values
- A scatter plot is generated to compare the predicted values against the true values.
2. Quantile Error Plot
- A box plot is created to visualize the prediction error across different quantiles of the true values.
3. Regression Metrics Table
- A table summarizing various regression performance metrics, such as:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared
  - Adjusted R-squared
  - Mean Absolute Percentage Error (MAPE)
  - Explained Variance Score (EVS)
  - Max Error
  - Median Absolute Error (MedAE)
  - Mean Squared Log Error (MSLE)
  - Root Mean Squared Log Error (RMSLE)

## The model
Can be found in the created model's folder under `/model`

- Pickled version of the model is saved as `.sav` file
- SQL version of the model is saved as `.sql` file

<br>

# Remarks

## Notes
- Limited to 3 models (EBM, linear/logistic regression, and Decision Tree).
- Data imbalance treatments (e.g., oversampling + model calibration) are not implemented.
- Only accepts CSV files.
- Interactions with more than 2 variables are not supported.

## TODO list
Check docs/TODO.md for an extensive list of planned features and improvements.
Feel free to open an issue in case a feature is missing or not working properly.

# Troubleshooting
If you encounter an unclear error message after following the instructions above, feel free to create an Issue on the GitHub repository.
