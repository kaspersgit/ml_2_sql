# Machine learning to SQL
## Background
Due to SQL being the main language being used for data manipulation and thus has extensive support in terms of compute and scheduling, 
why not perform inference with a machine learning model written in SQL code? The big limitation here is SQL itself, that's why we attempt to use 
machine learning models which have a simple structure it is writable in SQL. One additional benefit of this is that the model is interpretable, 
if you can write down the model in a basic logical language (SQL) you should be able to understand it (with limitation ofcourse).

This project tries to make the process simple enough for any SQL user to train a model, check the performance and deploy that model in SQL.

## Pre requisites
1. Create virtual environment and install packages, on mac run:
   ```
   python3 -m venv .ml2sql
   source .ml2sql/bin/activate
   pip install -r requirements.txt
   ```

## How to use main script
1. Save csv file containing target and all features in the `input/data/` folder
2. Save a settings json file in the `input/configuration/` folder which can contain:
   1. "target" -- name of target column (required)
   2. "features" -- list of names of features used (optional,  otherwise all columns except for target are used)
   3. "model_params" --  parameters used for the Explainable Boosting Machine (optional, [model documentation](https://interpret.ml/docs/ebm.html))
   4. "pre_params" --  parameters used for data pre processing (optional, check example file for parameters)
   5. "post_params" --  parameters used for post modeling (optional, check example file for parameters)
3. NULL handling is currently done in a hacky way which only works if numeric variables are >= 0 (as they are imputed with -1)
4. In the terminal run: `bash run.sh`
5. Follow the instruction on screen
6. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
7. The `.sql` file will contain a SQL Case When statement imitating the decision tree/EBM

### TODO list
- Add functionality to specify model parameters (done)
- Add AUC plots for classification (done)
- Add calibration plot (done)
- Save actual model (pickled) (done)
- Extend EBM to do multiclass (done)
- Extend EBM SQL for interaction terms (done)
- Allow for categorical features (done)
- Remove last bound and last score instead of adding inf (done)
- When training final model, make sure to train it on 1 copy of upsampled data (not un multiple copies of it) (done)
- Add logger in python files and save in folder (done)
- Add constant for EBM (as seen in the explain local plots) (done)
- Make EBM SQL code handle null values (filtered out atm) (done kind of by doing a hacky null impute by -1)
- make sure calibration data is not share of upsampled dataset (done)
- Time series splitting fix this one (done)
- Collect all param handling in one script (done)
- Group all X_ and y_ datasets into one dict (done)
- Get csv file from s3 link
- Add regression next to classification (also for SQL)
- Add MCC-F1 (curve?)
- plots for regression (idea https://towardsdatascience.com/visualizing-linear-ridge-and-lasso-regression-performance-6dda7affa251)
- Discovery on calibration (and how it can be written in SQL)
- Simplify interaction lookup df (make separate column per feature and perform a groupby)
- Use sklearn pipelines to simplify and streamline whole modelling process
- Spatial Cross-validation
- Extend logging granularity (add model parameters)
- Improve logging file clarity
- Add platt scaling and isotonic regression
- Add calibrated classifier from sklearn for model.sav and performance
- Add isotonic or logistic regression for sql version of model
- Make upsampling optional as a parameter
- Use menu function bash for model type choosing
