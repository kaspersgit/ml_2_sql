import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from ml2sql.utils.feature_selection.correlations import plot_correlations
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def remove_rows_with_nan_target(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Remove rows from the DataFrame where the target column has NaN values.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        pd.DataFrame: The DataFrame with rows removed where the target column has NaN values.
    """
    nan_rows = data[data[target_col].isna()]
    logger.info(f"Rows being removed due to NaN in target column: {len(nan_rows)}")
    return data.loc[data[target_col].notna(), :].reset_index(drop=True)


def remove_single_unique_value_features(
    data: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features from the DataFrame that have only a single unique value.

    Args:
        data (pd.DataFrame): The input DataFrame.
        feature_cols (List[str]): The list of feature column names.

    Returns:
        Tuple[pd.DataFrame, List[str]]: The DataFrame with single unique value features removed, and the updated list of feature columns.
    """
    one_nunique = data[feature_cols].columns[data[feature_cols].nunique() == 1]
    if len(one_nunique) > 0:
        data = data.loc[:, ~data.columns.isin(one_nunique)]
        logger.info(f"Features being removed due to single unique value: {one_nunique}")
        feature_cols = [f for f in feature_cols if f not in one_nunique]
    return data, feature_cols


def remove_one_occurrence_classes(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Remove classes from the DataFrame that have only one occurrence in the target column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        pd.DataFrame: The DataFrame with classes having only one occurrence removed.
    """
    one_occurrence_classes = (
        data[target_col].value_counts()[data[target_col].value_counts() == 1].index
    )
    if len(one_occurrence_classes) > 0:
        data = data[~data[target_col].isin(one_occurrence_classes)]
        logger.info(
            f"Removed class {one_occurrence_classes} due to having only 1 observation"
        )
    return data


def impute_and_cast_data(
    data: pd.DataFrame, feature_cols: List[str], model_name: str
) -> pd.DataFrame:
    """
    Impute missing values and cast data types for the feature columns in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        feature_cols (List[str]): The list of feature column names.
        model_name (str): The name of the model.

    Returns:
        pd.DataFrame: The DataFrame with imputed values and casted data types.
    """
    # Remove columns where more than 50% of rows are NaN/None/Null
    nan_cols = data.columns[data.isna().mean() > 0.5]
    if len(nan_cols) > 0:
        data.drop(nan_cols, axis=1, inplace=True)
        logger.info(
            f"Columns with more than 50% NaN values removed: {', '.join(nan_cols)}"
        )

    # Adjust data types
    for col in data[feature_cols].select_dtypes(include=["object"]).columns:
        # Check if values are true/false/None then boolean
        if all(val in [True, False, None, np.nan] for val in data[col].unique()):
            data[col] = data[col].astype(int)
        else:  # otherwise assume categorical
            data[col] = data[col].astype({col: "category"})

    if model_name != "ebm":  # otherwise model can't handle categorical features
        cat_cols = data[feature_cols].select_dtypes(include=["category"]).columns
        logger.info(f"Features being removed due to type being categorical: {cat_cols}")
        feature_cols = [f for f in feature_cols if f not in cat_cols]

    # Overview of data types
    logger.info(
        f"Column types in data set (including target)\n{data[feature_cols].dtypes.value_counts()}"
    )

    # Change boolean into 0's and 1's
    for col in data[feature_cols].select_dtypes(include=["bool"]).columns:
        data[col] = data[col].astype(int)

    return data


def cleanAndCastColumns(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    model_type: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean and cast columns in the DataFrame based on the provided parameters.

    Args:
        data (pd.DataFrame): The input DataFrame.
        feature_cols (List[str]): The list of feature column names.
        target_col (str): The name of the target column.
        model_name (str): The name of the model.
        model_type (str): The type of the model ('classification' or 'regression').

    Returns:
        Tuple[pd.DataFrame, List[str]]: The cleaned and casted DataFrame, and the updated list of feature columns.
    """
    _data = data.copy()

    # Clean out where target is NaN
    _data = remove_rows_with_nan_target(_data, target_col)

    # Clean out features with only one unique value
    _data, feature_cols = remove_single_unique_value_features(_data, feature_cols)

    if model_type == "classification":
        # Remove classes with only one occurrence
        _data = remove_one_occurrence_classes(_data, target_col)

    # Impute missing values and cast data types
    _data = impute_and_cast_data(_data, feature_cols, model_name)

    return _data, feature_cols


def pre_process_kfold(
    given_name: str,
    data: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    model_name: str,
    model_type: str,
    pre_params: Dict[str, str],
    post_params: Dict[str, str],
    random_seed: int = 42,
) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """
    Preprocess the data and create cross-validation folds.

    Args:
        given_name (str): The name of the dataset.
        data (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        feature_cols (List[str]): The list of feature column names.
        model_name (str): The name of the model.
        model_type (str): The type of the model ('classification' or 'regression').
        pre_params (Dict[str, str]): The preprocessing parameters.
        post_params (Dict[str, str]): The post-processing parameters.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        Dict[str, Dict[str, List[pd.DataFrame]]]: A dictionary containing the preprocessed datasets for final training, cross-validation, and out-of-time validation (if applicable).
    """
    # Clean and cast data types
    data_clean, feature_cols = cleanAndCastColumns(
        data, feature_cols, target_col, model_name, model_type
    )

    # Limit dataset with respect to the max_rows parameter
    if "max_rows" in pre_params:
        max_rows = min(pre_params["max_rows"], len(data_clean))
        data_clean = data_clean.sample(n=max_rows).reset_index(drop=True)
        logger.info(f"Limited dataset to {max_rows}")

    # Create correlation plots
    plot_correlations(data_clean[feature_cols], given_name, post_params["file_type"])

    # Create cross-validation folds
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

    # Initiate cross-validation method
    if pre_params["cv_type"] == "timeseriesplit":
        logger.info("Performing time series split cross-validation")
        data.sort_values(pre_params["time_sensitive_column"], inplace=True)
        kfold = TimeSeriesSplit(n_splits=5)
    elif model_type == "classification":
        logger.info("Performing stratified kfold cross-validation")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    elif model_type == "regression":
        logger.info("Performing normal kfold cross-validation")
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Create initial dictionary to collect datasets
    datasets = {}

    # Set X and y data apart
    y = data_clean[target_col]
    X = data_clean[feature_cols]

    if pre_params["upsampling"] != "false":
        # Create upsampled version of full dataset for final training
        X_ups, y_ups = upsample_data(X, y, model_type, random_seed=random_seed)

        # Add to datasets
        datasets["final_train"] = {"X": X_ups, "y": y_ups}
    else:
        # Add to datasets
        datasets["final_train"] = {"X": X, "y": y}

    # Create out-of-time (OOT) dataset if requested
    if pre_params["oot_set"] != "false":
        oot_df = data_clean.sort_values(
            pre_params["time_sensitive_column"], ascending=True
        ).tail(pre_params["oot_rows"])
        X_oot = oot_df[feature_cols]
        y_oot = oot_df[target_col]

        datasets["oot"] = {"X": X_oot, "y": y_oot}

        # Set new X and y data apart (excluding OOT)
        data_wo_oot = pd.concat([data_clean, oot_df]).drop_duplicates(keep=False)
        X = data_wo_oot[feature_cols]
        y = data_wo_oot[target_col]

    # Create datasets based on the different folds
    X_train_list, X_test_list, y_train_list, y_test_list = (
        list(),
        list(),
        list(),
        list(),
    )

    # Enumerate the folds and summarize the distributions
    kfold_nr = 0
    for train_ix, test_ix in kfold.split(X, y):
        # Record fold number
        logger.info(f"Creating fold nr {kfold_nr + 1}")

        # Select rows for train and test sets
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        if pre_params["upsampling"] != "false":
            # Report on number of rows
            logger.info(f"Number of rows before trimming: {len(X_train)}")
            logger.info(f"Imbalance before trimming: \n {y_train.value_counts()}")

            X_train_trim, y_train_trim = trim_pre_upsample_data(
                X_train, y_train, max_cells=50000, logger=logger
            )

            # Number of rows after trimming dataset
            logger.info(f"Number of rows before upsampling: {len(X_train_trim)}")
            logger.info(
                f"Imbalance before upsampling: \n {y_train_trim.value_counts()}"
            )

            # Upsample using SMOTE algorithm
            X_train, y_train = upsample_data(
                X_train_trim, y_train_trim, model_type, random_seed=random_seed
            )

        # Report on train and test set sizes
        logger.info(f"Number of rows in train set: {len(X_train)}")
        logger.info(f"Number of rows in test set: {len(X_test)}")

        if y_train.nunique() > 10:
            logger.info(f"Mean: {np.mean(y_train)} \nStd dev: {np.std(y_train)}")
        else:
            logger.info(f"Imbalance in train set: \n {y_train.value_counts()}")

        # Append to lists
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

        kfold_nr += 1

    # Add datasets to the dictionary
    datasets["cv_train"] = {"X": X_train_list, "y": y_train_list}
    datasets["cv_test"] = {"X": X_test_list, "y": y_test_list}

    return datasets


def upsample_data(
    X: pd.DataFrame, y: pd.Series, model_type: str, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Upsample the data using appropriate oversampling techniques based on the model type.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        model_type (str): The type of the model ('classification' or 'regression').
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The upsampled feature and target data.
    """
    if model_type == "classification":
        # Oversample train data
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
        except Exception as e:
            logger.warning(f"Error occurred during SMOTE/SMOTE-NC oversampling: {e}")
            ros = RandomOverSampler(random_state=random_seed)
            X_ups, y_ups = ros.fit_resample(X, y)
            logger.info("Random oversampling")

    elif model_type == "regression":
        X_ups, y_ups = X, y

    return X_ups, y_ups


def trim_pre_upsample_data(
    X: pd.DataFrame, y: pd.Series, max_cells: int, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Trim the dataset before upsampling to ensure the total number of cells (rows * columns) does not exceed a specified limit.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        max_cells (int): The maximum number of cells allowed.
        logger (logging.Logger): The logger instance.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The trimmed feature and target data.
    """
    # Reset indices
    X_ = X.reset_index(drop=True)
    y_ = y.reset_index(drop=True)

    # Trim dataset if necessary based on the amount of cells (columns x rows)
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

    logger.info(
        f"Original number of rows: {len(X_)} \nNew number of rows: {len(X_trim)}"
    )

    return X_trim.reset_index(drop=True), y_trim.reset_index(drop=True)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load data
    data = pd.read_csv("path/to/data.csv")

    # Define features and target
    target_col = "target_column"
    feature_cols = ["feature1", "feature2", ...]

    # Define model parameters
    model_name = "model_name"
    model_type = "classification"  # or "regression"

    # Define preprocessing parameters
    pre_params = {
        "max_rows": 10000,
        "upsampling": "true",
        "oot_set": "true",
        "oot_rows": 1000,
        "time_sensitive_column": "timestamp",
        "cv_type": "timeseriesplit",
    }

    # Define post-processing parameters
    post_params = {"file_type": "png"}

    # Call pre-processing function
    datasets = pre_process_kfold(
        "dataset_name",
        data,
        target_col,
        feature_cols,
        model_name,
        model_type,
        pre_params,
        post_params,
        random_seed=42,
    )

    # Access the preprocessed datasets
    X_train, y_train = datasets["final_train"]["X"], datasets["final_train"]["y"]
    X_test, y_test = datasets["cv_test"]["X"], datasets["cv_test"]["y"]

    # Train and evaluate model
    # ...
