# Machine learning to SQL

![GitHub Repo stars](https://img.shields.io/github/stars/kaspersgit/ml_2_sql?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/kaspersgit/ml_2_sql?style=flat-square)
![interpret](https://img.shields.io/badge/interpret-v0.5.1-blue)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![GitHub](https://img.shields.io/github/license/kaspersgit/ml_2_sql?style=flat-square)


# Table of Contents
1. [What is it?](#what-is-it)
2. [Getting Started](#getting-started)
3. [Input](#input)
4. [Output](#output)
5. [Remarks](#remarks)
6. [Troubleshooting](#troubleshooting)

<br>

# What is it?
## In short
This project tries to make the process simple enough for anyone to train a model, check the performance and deploy that model in SQL.

## Philosophy:
- Automated training, easy to use and clear performance metrics
- Model written in SQL (avoid Data Science debt for a complex pipeline)
- Explainable Boosting Machine on par with other boosted methods while fully explainable

## Background
An automated machine learning tool which trains, graphs performance and saves the model in SQL. Using interpretable ML models (from interpretML) to train models which are explainable and interpretable, so called 'glassbox' models. SQL infrastructure is the only requirement to put a model into production.
This tool can be used by anybody, but is aimed for people who want to easily train a model, understand what the impact of the features is and deploy it in a SQL system. 

## Note
- Limited to 3 models to choose from: 
  - [Explainable Boostin Machine](https://interpret.ml/docs/ebm.html) 
  - [Linear/Logistic regression](https://interpret.ml/docs/lr.html)
  - [Decision tree](https://interpret.ml/docs/dt.html)
- Only accepts CSV files

</br>

# Getting started
## Pre requisites
1. Make sure you have python >= 3.8 install 
2. Clone Github repo to your local machine and cd into folder
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

## Try it out demo
1. In the terminal in the root of this folder run: 
  - `python3 run.py` (Mac/Linux)
  - `python run.py` (Windows)
2. Follow the instructions on screen by selecting the example data and config file
3. Check the output in the newly created folder

## Try it out using own data
1. Save csv file containing target and all features in the `input/data/` folder (more info on [input data](#data))
2. Save a configuration json file in the `input/configuration/` (or create one during the next step) (more info on [config json](#configuration-json))
3. In the terminal in the root of this folder run: 
  - `python3 run.py` (Mac/Linux)
  - `python run.py` (Windows)
4. Follow the instruction on screen
5. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
6. The `.sql` file in the `model` folder will contain a SQL written model  

## Testing already trained model on a new dataset
1. Make sure new dataset has the same features as the dataset the model was trained on (same features)
2. Save dataset in the `input/data/` folder (more info on [input data](#data))
3. In the terminal in the root of this folder run: 
  - `python3 check_model.py` (Mac/Linux)
  - `python check_model.py` (Windows)
4. Follow the instructions on screen
5. The output will be saved in the folder `trained_models/<selected_model>/tested_datasets/<selected_dataset>/`

</br>

# Input
## Data
The csv file containing the data has to fulfill some basic assumptions:
- Target column should have more than 1 unique value
- For binary classification (target with 2 unique values) these values should be 0 and 1
- File name should be .csv and not consist of any spaces

Missing values are allowed

## Configuration json
This file will inform the script which column is the target, which are the features and several other parameters for pre and post training.
You can copy and edit a config file from the already existing example in `input/configuration/` or select `Create a new config` file in the second step 
when running the `run.py` file.

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

</br>

# Output
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
3. alibration Plot
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
- Limited to 3 models (EBM, lin./log. regression and decision tree)
- Data imbalance treatments (e.g. oversampling + model calibration) not fully implemented
- Decision rule not implemented yet
- Only accepts CSV files
- Interactions with >2 variables not supported

## TODO list
Check docs/TODO.md for an extensive list.

# Troubleshooting
If error message is not clear and instructions above are followed, feel free to create an Issue.
