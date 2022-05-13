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
2. Save a settings json file in the `input/configuration/` (explained below at `Configuration json`)
3. In the terminal run: `bash run.sh`
4. Follow the instruction on screen
5. The output will be saved in the folder `trained_models/<current_date>_<your_model_name>/`
6. The `.sql` file will contain a SQL Case When statement imitating the decision tree/EBM

### Configuration json
#### features
List with names of the columns which should be used as feature (optional)

#### model_params
Dictionary of parameters that can be used with model of choice (optional). Check the model's documentation:
- EBM ([model documentation](https://interpret.ml/docs/ebm.html))

#### post_params
`calibration` options (optional, not fully implemented):
- `sigmoid`, platt scaling applied
- `isotonic`, isotonic regression applied
- `auto`/`true`, either platt scaling or isotonic regression applied based on datasize
- any other value, not calibration applied

`sql_split` options:
- `false`, outputs the SQL model as one column by adding all separate scores up directly
- `true`, outputs the SQL model as one column for each feature and a total score columns afterwards. This might be needed to avoid some memory related (stackoverflow) error.

#### pre_params
`cv_type` options (optional):
- `timeseriesplit`, perform 5 fold timeseries split ([sklearn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html))
- any other value, perform 5 fold stratified cross validation

`max_rows` options (not used currently):
- Any kind of whole positive number, will limit the data set in order to train faster (as simple as that)

`time_sensitive_column` options (optional):
- Name of date column
  - used when `cv_type = timeseriesplit`  
  - used when out-of-time dataset is created (not implemented yet)

`upsampling` options (optional):
- `true`, applying the SMOTE(NC) algorithm on the minority class to balance the data
- `false`, not applying any resampling technique

#### target 
Name of target column (required)

### Notes
- Any NULL values should be imputed before using this script
- Data imbalance treatments (e.g. oversampling + model calibration) not fully implemented
- Resampling (almost) always makes the trained model ill calibrated
- Multiclass and regression are experimental

### TODO list
- Get csv file from s3 link
- Make regression EBM work fully
- Make multi class classification EBM work fully
- Use sklearn pipelines to simplify and streamline whole modelling process
- Spatial Cross-validation
- Extend logging granularity (add model parameters)
- Add platt scaling and isotonic regression
- Add calibrated classifier from sklearn for model.sav and performance
- Add isotonic or logistic regression for sql version of model
- Make upsampling optional as a parameter
- Use menu function bash for model type choosing
- Implement null handling (there is an implementation mentioned [here](https://github.com/interpretml/interpret/issues/18))
