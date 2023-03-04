# Machine learning to SQL

![GitHub last commit](https://img.shields.io/github/last-commit/kaspersgit/ml_2_sql?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/kaspersgit/ml_2_sql?style=flat-square)
![GitHub](https://img.shields.io/github/license/kaspersgit/ml_2_sql?style=flat-square)


# Table of Contents
1. [What is it?](#what-is-it)
2. [Getting Started](#getting-started)
3. [Input](#input)
4. [Remarks](#remarks)
5. [Troubleshooting](#troubleshooting)


# What is it?
This project tries to make the process simple enough for any SQL user to train a model, check the performance and deploy that model in SQL.

## Background
An automated machine learning tool which trains, graphs performance and saves the model in SQL. Using interpretable ML models (from interpretML) to train models which are explainable and transparent in how they come to their prediction. SQL infrastructure is the only requirement to put a model into production.
This tool can be used by anybody, but is aimed for people who want to easily train a model and want to use it in their (company) SQL system. 

Advantages:
- Automated model training
- Easy usage
- Collection of model performance metrics generated
- Model saved in SQL file
- Avoid Python production models 
- Explainable Boosting Machine on par with other boosted methods

Disadvantages:
- No neural networks and other complex models (boosted trees)
- Tool runs locally
- Some values are hardcoded (e.g. kfold: k = 5)

## Current state
- Only EBM is implemented (decision tree, logistic regression and rule set not yet)
- Automated model training is working for binary classification and regression
- SQL creation of model is working fully for binary clasification
- SQL for regression and the whole process for multiclass classification is wip
- Up/down sampling not fully implemented yet 

</br>

# Getting started
## Pre requisites
1. Create virtual environment and install packages, run:
   
    Windows:
   ```
   python -m venv .ml2sql
   .ml2sql/Scripts/activate
   pip install -r requirements.txt
   ```
   
    Mac/Linux:
   ```
   python3 -m venv .ml2sql
   source .ml2sql/bin/activate
   pip install -r requirements.txt
   ```
2. Wait until all packages are installed (could take a few minutes)
3. Activate or deactivate the created virtual environment by running:
   
    Windows:
   `.ml2sql/Scripts/activate` or `deactivate`
   
    Mac/Linux:
   `source .ml2sql/bin/activate` or `deactivate`

## Try it out demo
1. Make sure virtual environment is active (step 3 of Pre requisites)
2. In the terminal in the root of this folder run: `python run.py`
3. Follow the instructions on screen by selecting the demo data and (similarly named) config file
4. Check the output in the newly created folder

## Try it out using own data
1. Make sure virtual environment is active (step 3 of Pre requisites)
2. Save csv file containing target and all features in the `input/data/` folder (more on input data at [input data](#data))
3. Save a settings json file in the `input/configuration/` (explained below at [configuration json](#configuration-json))
4. In the terminal run: `bash run.sh`
5. Follow the instruction on screen
6. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
7. The `.sql` file will contain a SQL Case When statement imitating the decision tree/EBM  

</br>

# Input
## Data
The csv file containing the data has to fulfill some basic assumptions:
- No empty values (e.g. NULL, Na, NaN, etc.)
- Target columns should have more than 1 unique value

## Configuration json
### features
List with names of the columns which should be used as feature (optional)

### model_params
Dictionary of parameters that can be used with model of choice (optional). Check the model's documentation:
- EBM ([model documentation](https://interpret.ml/docs/ebm.html))
- Decision tree ([model documentation](https://interpret.ml/docs/dt.html))
- Decision rule ([model documentation](https://interpret.ml/docs/dr.html))

### post_params
`calibration` options (optional, not fully implemented):
- `sigmoid`, platt scaling applied
- `isotonic`, isotonic regression applied
- `auto`/`true`, either platt scaling or isotonic regression applied based on datasize
- any other value, no calibration applied

`sql_split` options:
- `false`, outputs the SQL model as one SELECT statement, using column aliases within the same select statement
- `true`, outputs the SQL model as several CTEs, this can be used if column aliases can't be referenced within the same SELECT statement

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
- Name of date column
  - used when `cv_type = timeseriesplit`  
  - used when out-of-time dataset is created (not implemented yet)

`upsampling` options (optional, should not be used without calibration):
- `true`, applying the SMOTE(NC) algorithm on the minority class to balance the data
- `false`, not applying any resampling technique

### target
Name of target column (required)

</br>

# Remarks

## Notes
- Any NULL values should be imputed before using this script
- Data imbalance treatments (e.g. oversampling + model calibration) not fully implemented
- Resampling (almost) always makes the trained model ill calibrated
- Multiclass and regression are experimental

## TODO list
- Add decision tree
- Add logistic regression
- Add Skope rules
- Add calibration (platt scaling/isotonic regression)
- Add changelog and versioning
- Implement null handling (there is an implementation mentioned [here](https://github.com/interpretml/interpret/issues/18))
- Make multi class classification EBM work fully
- Make regression EBM work fully
- Removing outliers by using quantiles (e.g. only keeping 1 - 99 % quantiles)
- Spatial Cross-validation discovery
- Extend logging granularity (add model parameters)
- Add target single unique value check
- Replace/improve `classification_report` and `confusion_matrix` due to dependance on threshold
- Add MCC, cohen kappa and other metrics plotted with threshold
- Add SQL translation for decision rule
- Add random seed to config file
- Make a proper testing procedure
- Improve ReadMe
- Add more example files (binary/regression/multiclass)
- Add tool to make configuration json
- Make distribution plot grouped instead of overlaid or stacked (maybe switch to plotly histogram)
