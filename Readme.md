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


# In short
This project tries to make the process simple enough for any SQL user to train a model, check the performance and deploy that model in SQL.

## Philosophy:
- Automated and easy to use 
- Model in SQL (avoid Data Science debt for a complex pipeline)
- Explainable Boosting Machine on par with other boosted methods

## Background
An automated machine learning tool which trains, graphs performance and saves the model in SQL. Using interpretable ML models (from interpretML) to train models which are explainable and transparent in how they come to their prediction. SQL infrastructure is the only requirement to put a model into production.
This tool can be used by anybody, but is aimed for people who want to easily train a model and want to use it in their (company) SQL system. 

## Current state
- Only EBM is implemented (decision tree, logistic regression and rule set not yet)
- Automated model training is working for binary classification and regression
- SQL creation of model is working fully for binary clasification
- SQL for regression and the whole process for multiclass classification is work in progress
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
4. For usage of this tool the virtual environment can be deactivated.

## Try it out demo
1. In the terminal in the root of this folder run: 
  - `python3 run.py` (Mac/Linux)
  - `python run.py` (Windows)
2. Follow the instructions on screen by selecting the demo data and (similarly named) config file
3. Check the output in the newly created folder

## Try it out using own data
1. Save csv file containing target and all features in the `input/data/` folder (more on input data at [input data](#data))
2. Save a settings json file in the `input/configuration/` (explained below at [configuration json](#configuration-json))
3. In the terminal in the root of this folder run: 
  - `python3 run.py` (Mac/Linux)
  - `python run.py` (Windows)
4. Follow the instruction on screen
5. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
6. The `.sql` file in the `model` folder will contain a SQL written model  

## Testing already trained model on a new dataset
1. Make sure new dataset is exactly the same format as the dataset the model was trained on
2. Save dataset in the `input/data/` folder (more on input data at [input data](#data))
3. In the terminal in the root of this folder run: 
  - `python3 test_model.py` (Mac/Linux)
  - `python test_model.py` (Windows)
4. Follow the instructions on screen
5. The output will be saved in the folder `trained_models/<selected_model>/tested_datasets/<selected_dataset>/`

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
- Multiclass and regression are experimental

## TODO list
- EBM
  - SQL interaction terms first term with least values
  - SQL interaction terms group by to merge similar score bounds
  - Make multi class classification EBM work fully
  - Make regression EBM work fully

- Checks and config
  - Add check if variables and target are finite 
  - Add variables being NULL checked
  - Add random seed to config file
  - Implement null handling (there is an implementation mentioned [here](https://github.com/interpretml/interpret/issues/18))

- Performance monitoring
  - Add performance summary for easy and quick comparison (including label count, auc pr & roc, best f1-score, etc)
  - Add feature over/under fitting plot (https://towardsdatascience.com/which-of-your-features-are-overfitting-c46d0762e769)
  - Make distribution plot grouped instead of overlaid or stacked (maybe switch to plotly histogram)

- Other 
  - Add calibration (platt scaling/isotonic regression)
  - Add changelog and versioning
  - Extend logging granularity 
  - Add SQL translation for decision rule
  - Make a proper testing procedure
  - Add more example files (binary/regression/multiclass)

# Troubleshooting