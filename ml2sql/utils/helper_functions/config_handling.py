import pandas as pd
import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
logger = logging.getLogger(__name__)


# Handle the configuration file
def config_handling(configuration, data):
    """
    Handles the configuration file and extracts necessary information.

    Parameters:
    -----------
    configuration : dict
        A dictionary containing configuration information such as target column, features columns,
        model related parameters, pre processing related parameters, and post modeling related parameters.

    logging : logger object
        A logger object to log the progress and debugging information.

    Returns:
    --------
    tuple
        A tuple containing the following:
        - target_col : str
            The name of the target column.
        - feature_cols : list
            A list containing the names of the feature columns.
        - model_params : dict
            A dictionary containing model related parameters.
        - pre_params : dict
            A dictionary containing pre processing related parameters.
        - post_params : dict
            A dictionary containing post modeling related parameters.
    """
    # target column
    target_col = configuration["target"]

    # features columns
    if "features" in configuration.keys():
        feature_cols = configuration["features"]
        logger.info(f"{len(feature_cols)} features specified in file")
    else:
        # treat all other columns as features
        feature_cols = list(data.columns)
        feature_cols.remove(target_col)
        logger.info(f"Using {len(feature_cols)} features (all columns except target)")

    # model related parameters
    if "model_params" in configuration.keys():
        model_params = configuration["model_params"]
    else:
        model_params = {}

    # pre processing related parameters
    if "pre_params" in configuration.keys():
        pre_params = configuration["pre_params"]
    else:
        pre_params = {}

    if not ("oot_set" in pre_params.keys()) & ("oot_rows" in pre_params.keys()):
        pre_params["oot_set"] = "false"

    # Cross validation type to perform
    if "cv_type" not in pre_params.keys():
        pre_params["cv_type"] = "kfold_cv"

    # If not present set upsamplling to false
    if "upsampling" not in pre_params.keys():
        pre_params["upsampling"] = "false"

    # post modeling related parameters
    if "post_params" in configuration.keys():
        post_params = configuration["post_params"]
    else:
        post_params = {}

    # # unpack calibration from params
    if "calibrate" in post_params.keys():
        if post_params["calibration"] in ["auto", "sigmoid", "isotonic", "true"]:
            post_params["calibration"] = (
                "auto"
                if post_params["calibration"] == "true"
                else post_params["calibration"]
            )
        else:
            post_params["calibration"] = "false"
    else:
        post_params["calibration"] = "false"

    # If not present set calibration check if we upsample
    if "calibration" not in post_params.keys():
        if pre_params["upsampling"] == "true":
            post_params["calibration"] = "auto"
        else:
            post_params["calibration"] = "false"

    # If not present set split to false
    if "sql_split" in post_params.keys():
        if post_params["sql_split"] == "true":
            post_params["sql_split"] = True
        else:
            post_params["sql_split"] = False
    else:
        post_params["sql_split"] = False

    # If not present set sql decimals to 15
    if "sql_decimals" not in post_params.keys():
        post_params["sql_decimals"] = 15

    # If not present set file type to png
    if "file_type" in post_params.keys():
        if post_params["file_type"].lower() == "html":
            post_params["file_type"] = "html"
        else:
            post_params["file_type"] = "png"
    else:
        post_params["file_type"] = "png"

    return target_col, feature_cols, model_params, pre_params, post_params


def _get_col_dtype(col):
    """
    Sourced: https://stackoverflow.com/questions/35003138/python-pandas-inferring-column-datatypes
    Infer datatype of a pandas column, process only if the column dtype is object.
    input:   col: a pandas Series representing a df column.
    """

    if col.dtype == "object":
        # try datetime
        try:
            col_new = pd.to_datetime(col.dropna().unique())
            return col_new.dtype
        except ValueError:
            # try numeric
            try:
                col_new = pd.to_numeric(col.dropna().unique())
                return col_new.dtype
            except ValueError:
                try:
                    col_new = pd.to_timedelta(col.dropna().unique())
                    return col_new.dtype
                except ValueError:
                    return "object"
    else:
        return col.dtype


def select_ml_cols(df):
    # Create a dictionary to store the features and their indices
    features_set = set(df.columns)

    print("Columns excluded for feature list:")

    # Check certain column name
    check_date_cols = ["date", "dt"]

    # Check uniqeness
    for col in df.columns:
        # Check share of unq values in the columns
        uniqueness_ratio = len(df[col].unique()) / len(df[col])

        # Check if all elements in the column have the same length
        all_same_length = df[col].apply(lambda x: len(str(x))).nunique() == 1

        if (uniqueness_ratio == 1) & (df[col].dtypes != "float"):
            features_set.discard(col)
            print(f'"{col}" only has unique values')

        elif (uniqueness_ratio > 0.9) & (df[col].dtypes == "int") & (all_same_length):
            features_set.discard(col)
            print(f'"{col}" is int column with high cardinality but same length')

        # TODO remove this and rely on the data type inferring
        elif any([cdc in col.lower() for cdc in check_date_cols]):
            for cdc in check_date_cols:
                if cdc in col.lower():
                    features_set.discard(col)
                    print(f'"{col}" is a date column')

        elif (uniqueness_ratio > 0.4) & (df[col].dtypes == "object"):
            features_set.discard(col)
            print(f'"{col}" is object column with high cardinality')

        elif df[col].nunique() == 1:
            features_set.discard(col)
            print(f'"{col}" is column with only one value')

        else:
            inferred_dtype = _get_col_dtype(df[col])
            if all(
                ele not in str(inferred_dtype)
                for ele in ["object", "string", "int", "float"]
            ):
                features_set.discard(col)
                print(f'"{col}" with datetype {inferred_dtype}')

    return features_set
