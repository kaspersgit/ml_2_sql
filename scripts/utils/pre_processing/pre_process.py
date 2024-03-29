import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from utils.feature_selection.correlations import (
    plotPearsonCorrelation,
    plotCramervCorrelation,
)
import logging

logger = logging.getLogger(__name__)


def cleanAndCastColumns(data, feature_cols, target_col, model_name, model_type):
    # make copy of data
    _data = data.copy()

    # clean out where target is NaN
    logger.info(
        "Rows being removed due to NaN in target column: {nans} \n".format(
            nans=len(_data[_data[target_col].isna()])
        )
    )

    _data = _data.loc[_data[target_col].notna(), :].reset_index(drop=True)

    # Clean out feature which only has 1 unique value
    one_nunique = _data[feature_cols].columns[_data[feature_cols].nunique() == 1]

    if len(one_nunique) > 0:
        _data = _data.loc[:, ~_data.columns.isin(one_nunique)]
        logger.info(
            f"Features being removed due to single unique value: {one_nunique} \n"
        )

        # Remove feature from feature_cols if in there
        for f in one_nunique:
            if f in feature_cols:
                feature_cols.remove(f)

    if model_type == "classification":
        # Remove classes with only one occurence
        one_occurence_class = (
            _data[target_col]
            .value_counts()[_data[target_col].value_counts() == 1]
            .index
        )
        if len(one_occurence_class) > 0:
            _data = _data[~_data[target_col].isin(one_occurence_class)]

            logger.info(
                "Removed class {classification} due to having only 1 observation".format(
                    classification=one_occurence_class
                )
            )

    # Imputing missing values and casting
    # assuming only int/float and bool column types
    nan_cols = _data[feature_cols][
        _data[feature_cols].columns[_data[feature_cols].isna().sum() > 0]
    ]
    if len(nan_cols) > 0:
        logger.info(
            "Columns with NaN values: \n{nans}".format(nans=nan_cols.isna().sum())
        )

    # Remove columns where more than 50% of rows are NaN/None/Null
    nan_cols = _data.columns[_data.isna().mean() > 0.5]
    if len(nan_cols) > 0:
        _data.drop(nan_cols, axis=1, inplace=True)
        print(f"Columns with more than 50% NaN values removed: {', '.join(nan_cols)}")

    # Adjust this to allow for categorical features
    for col in _data[feature_cols].select_dtypes(include=["object"]).columns:
        # Check if values are true/false/None then boolean
        if all(val in [True, False, None, np.NaN] for val in _data[col].unique()):
            _data[col] = _data[col].astype(int)
        else:  # otherwise assume categorical
            _data[col] = _data[col].astype({col: "category"})

    if model_name != "ebm":  # otherwise model can't handle categorical features
        cat_col = _data[feature_cols].select_dtypes(include=["category"]).columns
        logger.info(
            f"Features being removed due to type being categorical: {cat_col} \n"
        )
        # Remove feature from feature_cols if in there
        for f in cat_col:
            if f in feature_cols:
                feature_cols.remove(f)

    # Overview of _data types
    logger.info(
        "Column types in data set (including target)\n{col_types} \n".format(
            col_types=_data[feature_cols].dtypes.value_counts()
        )
    )

    # change boolean into 0's and 1's
    for col in _data[feature_cols].select_dtypes(include=["bool"]).columns:
        _data[col] = _data[col].astype(int)

    return _data.reset_index(drop=True)


def imbalanceness(labels):
    classes_count = labels.value_counts()
    max_class_size = classes_count.max()
    min_class_size = classes_count.min()
    total_size = classes_count.sum()
    nclasses = len(classes_count)

    return (max_class_size - min_class_size) / (total_size - nclasses)


def pre_process_kfold(
    given_name,
    data,
    target_col,
    feature_cols,
    model_name,
    model_type,
    pre_params,
    post_params,
    random_seed=42,
):
    # clean and cast
    data_clean = cleanAndCastColumns(
        data, feature_cols, target_col, model_name, model_type
    )

    # Limit dataset with respect to the max_rows parameter
    if "max_rows" in pre_params:
        max_rows = min(pre_params["max_rows"], len(data_clean))
        data_clean = data_clean.sample(n=max_rows).reset_index(drop=True)
        logger.info(f"Limited dataset to {max_rows}")

    # Create correlation plots
    plotPearsonCorrelation(
        data_clean[feature_cols], given_name, post_params["file_type"]
    )
    plotCramervCorrelation(
        data_clean[feature_cols], given_name, post_params["file_type"]
    )

    # create kfolds in a statified manner
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

    # initiate kfold
    if pre_params["cv_type"] == "timeseriesplit":
        logger.info("Performing time series split cross validation")
        data.sort_values(pre_params["time_sensitive_column"], inplace=True)
        kfold = TimeSeriesSplit(n_splits=5)
    elif model_type == "classification":
        logger.info("Performing stratified kfold cross validation")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    elif model_type == "regression":
        logger.info("Performing normal kfold cross validation")
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Create initial dict to collect datasets
    datasets = {}

    # set X and y data apart
    y = data_clean[target_col]
    X = data_clean[feature_cols]

    if pre_params["upsampling"] != "false":
        #### Create upsampled version of full dataset for final training
        # Make sure total rows * columns after upsampling won't hit x nr of cells
        max_cells = 50000
        X_trim, y_trim = trimPreUpsampleDataRows(X, y, max_cells)

        # upsample by trying SMOTE algo
        X_ups, y_ups = upsampleData(X, y_trim, model_type, random_seed=42)

        # Add to datasets
        datasets["final_train"] = {"X": X_ups, "y": y_ups}
    else:
        # Add to datasets
        datasets["final_train"] = {"X": X, "y": y}
    ####

    #### Create OOT dataset if wanted
    if pre_params["oot_set"] != "false":
        oot_df = data_clean.sort_values(
            pre_params["time_sensitive_column"], ascending=True
        ).tail(pre_params["oot_rows"])
        X_oot = oot_df[feature_cols]
        y_oot = oot_df[target_col]

        datasets["oot"] = {"X": X_oot, "y": y_oot}

        # set new X and y data apart (oot excluded)
        data_wo_oot = pd.concat([data_clean, oot_df]).drop_duplicates(keep=False)
        X = data_wo_oot[feature_cols]
        y = data_wo_oot[target_col]
    ####

    #### Create datasets based on the different folds
    # listing the different folds
    X_train_list, X_test_list, y_train_list, y_test_list = (
        list(),
        list(),
        list(),
        list(),
    )

    # enumerate the splits and summarize the distributions
    kfold_nr = 0
    for train_ix, test_ix in kfold.split(X, y):
        # Record kfold
        logger.info(f"Creating fold nr {kfold_nr+1}")

        # select rows
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        if pre_params["upsampling"] != "false":
            # report on nr rows
            logger.info(f"Nr rows pre trimming: {len(X_train)}")
            logger.info(f"imbalanceness pre trimming: \n {y_train.value_counts()}")

            X_train_trim, y_train_trim = trimPreUpsampleDataRows(
                X_train, y_train, max_cells, logger
            )

            # Nr rows after trimming down dataset
            logger.info(f"Nr rows pre upsampling: {len(X_train_trim)}")
            logger.info(
                f"imbalanceness pre upsampling: \n {y_train_trim.value_counts()}"
            )

            # upsample by trying SMOTE algo
            X_train, y_train = upsampleData(
                X_train_trim, y_train_trim, model_type, random_seed=42
            )

        # Nr rows of training set
        logger.info(f"Nr rows train set: {len(X_train)}")
        logger.info(f"Nr rows test set: {len(X_test)}")
        logger.info(f"imbalanceness train: \n {y_train.value_counts()}")

        # append to the lists
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

        kfold_nr += 1

    # add datasets
    datasets["cv_train"] = {"X": X_train_list, "y": y_train_list}
    datasets["cv_test"] = {"X": X_test_list, "y": y_test_list}
    ####

    return datasets


def upsampleData(X, y, model_type, random_seed=42):
    if model_type == "classification":
        # oversample train data
        # nested try except (https://stackoverflow.com/questions/17015230/are-nested-try-except-blocks-in-python-a-good-programming-practice)

        try:
            categorical_cols = X.dtypes == "category"
            if categorical_cols.sum() > 0:
                ros = SMOTENC(
                    categorical_features=categorical_cols, random_state=random_seed
                )
                X_ups, y_ups = ros.fit_resample(X, y)
                logger.info("SMOTE-NC oversampling")
            else:
                ros = SMOTE(random_state=random_seed)
                X_ups, y_ups = ros.fit_resample(X, y)
                logger.info("SMOTE oversampling")
        except Exception:
            ros = RandomOverSampler(random_state=random_seed)
            X_ups, y_ups = ros.fit_resample(X, y)
            logger.info("Random oversampling")

    elif model_type == "regression":
        X_ups, y_ups = X, y

    return X_ups, y_ups


def trimDownDataRows(X, y, max_cells):
    # Trim dataset if necessary based on amount of cells (columns x rows)
    nr_cells = X.shape[0] * X.shape[1]
    if nr_cells > max_cells:
        logger.info(
            f"Dataset shape {X.shape} resulting in {nr_cells} cells \nTrimming down..."
        )
        df_pretrim = X.join(y)
        df_posttrim = df_pretrim.sample(n=round(max_cells / X.shape[1]))
        X_trim = df_posttrim[X.columns].reset_index(drop=True)
        y_trim = df_posttrim[y.name].reset_index(drop=True)
        nr_cells_trim = X_trim.shape[0] * X_trim.shape[1]
        logger.info(
            f"Trimmed down to {X_trim.shape} resulting in {nr_cells_trim} cells."
        )
    else:
        X_trim = X
        y_trim = y

    return X_trim, y_trim


def trimPreUpsampleDataRows(X, y, max_cells):
    # reset index
    X_ = X.reset_index(drop=True)
    y_ = y.reset_index(drop=True)

    # Trim dataset if necessary based on amount of cells (columns x rows)
    classes_counts = y_.value_counts()
    size_majority_class = classes_counts.max()

    exp_nr_cells = size_majority_class * X_.shape[1]  # * nr_classes
    max_rows = round(max_cells / (X_.shape[1]))  # * nr_classes

    if exp_nr_cells > max_cells:
        logger.info(
            f"Expecting {exp_nr_cells} cells, more than set limit of {max_cells}"
        )
        big_classes = classes_counts.index[classes_counts > max_rows]

        y_trim = y_[~y_.isin(big_classes)]
        for c in big_classes:
            y_big = y_[y_ == c]
            y_big_trim = y_big.sample(max_rows)
            y_trim = y_trim.append(y_big_trim)
        y_trim = y_trim.sort_index()
        X_trim = X_[X_.index.isin(y_trim.index)]
    else:
        y_trim = y_
        X_trim = X_

    logger.info(f"Original nr of rows {len(X_)} \nNew nr of rows {len(X_trim)}")

    return X_trim.reset_index(drop=True), y_trim.reset_index(drop=True)
