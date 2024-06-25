import pandas as pd
import numpy as np
import re
import os
import warnings

warnings.simplefilter("ignore", UserWarning)


# Helper function to remove spaces from numbers
def remove_spaces(value):
    if isinstance(value, str):
        return int(value.replace(" ", ""))
    return value


# Helper function to split monetary values into amount and currency
def split_money(value):
    if isinstance(value, str):
        match = re.match(r"([\$£€¥]+\s?)?([0-9,. ]+)([a-zA-Z]+)", value)
        if match:
            amount = match.group(2).replace(",", "").replace(" ", "")
            currency = match.group(3) if match.group(3) else match.group(1)
            return float(amount), currency
    return np.nan, np.nan


def quick_clean_data(data_path: str = None):
    # if no data path is given then list files in input/data
    if data_path is None:
        # List files in input/data/ directory
        data_dir = "input/data/"
        files = []
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                files.append(f)
        files.sort()

        print("Files in input/data/:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")

        # Ask for CSVPATH
        while True:
            try:
                csv_file_index = input("\nSelect CSV file for training the model: ")
                csv_file_index = int(csv_file_index) - 1
                data_path = os.path.join(data_dir, files[csv_file_index])
                break
            except IndexError:
                print("Invalid index. Please try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        print(f"CSV file {data_path} will be cleaned and saved")

    # df = pd.read_csv(data_path, dtype='object', dtype_backend='pyarrow')
    df = pd.read_csv(data_path)
    print(f"\nOriginal variables and dtypes:\n{df.dtypes}")

    # Iterate over the columns and handle specific cases
    for col in df.columns:
        # Check if the column contains numbers with spaces (thousands separators)
        if (
            df[col]
            .dropna()
            .apply(
                lambda x: isinstance(x, str)
                and x.replace(",", "").replace(".", "").replace(" ", "").isdigit()
                and " " in x
            )
        ).any():
            df[f"{col}_numeric"] = df[col].apply(remove_spaces)

        # Check for null values represented as strings
        null_values = ["nan", "null", "none"]
        if (df[col].isin(null_values)).any():
            df[col] = df[col].lower().replace(null_values, np.nan)

        # Check for columns with both numeric and string values
        numeric_count = pd.to_numeric(df[col], errors="coerce").notnull().sum()
        total_count = df[col].notnull().sum()
        if numeric_count > 0 and numeric_count / total_count >= 0.25:
            df[f"{col}_numeric"] = pd.to_numeric(df[col], errors="coerce")
            if numeric_count < total_count:
                df[f"{col}_non_numeric"] = df[col][
                    pd.to_numeric(df[col], errors="coerce").isna()
                ]
            continue  # Skipping a numeric being interpreted as a datetime

        # Check for boolean values
        bool_values = ["True", "False", "true", "false"]
        if (df[col].dropna().isin(bool_values)).all():
            df[f"{col}_boolean"] = df[col].apply(
                lambda x: True
                if x in ["True", "true"]
                else False
                if x in ["False", "false"]
                else np.nan
            )

        # Check for monetary values
        if (
            df[col]
            .dropna()
            .apply(
                lambda x: isinstance(x, str)
                and re.match(r"([\$£€¥]+\s?)?[0-9,. ]+[a-zA-Z]+", x)
            )
        ).any():
            df[[f"{col}_amount", f"{col}_currency"]] = (
                df[col].apply(split_money).apply(pd.Series)
            )
            df.drop(col, axis=1, inplace=True)

        # Check for date columns
        if pd.to_datetime(df[col], errors="coerce").notnull().all():
            df[f"{col}_datetime"] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[f"{col}_datetime"].dt.year
            df[f"{col}_month"] = df[f"{col}_datetime"].dt.month
            df[f"{col}_day"] = df[f"{col}_datetime"].dt.day
            df[f"{col}_dayofweek"] = df[f"{col}_datetime"].dt.day_name()
            df[f"{col}_dayofweek_int"] = df[f"{col}_datetime"].dt.day_of_week
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12.0)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12.0)
            df[f"{col}_dayofweek_sin"] = np.sin(
                2 * np.pi * df[f"{col}_dayofweek_int"].map(lambda x: x // 7) / 7.0
            )
            df[f"{col}_dayofweek_cos"] = np.cos(
                2 * np.pi * df[f"{col}_dayofweek_int"].map(lambda x: x // 7) / 7.0
            )
            df = pd.get_dummies(df, columns=[f"{col}_month", f"{col}_dayofweek"])
            df.drop(col, axis=1, inplace=True)

    # Remove columns where more than 50% of rows are NaN/None/Null
    nan_cols = df.columns[df.isna().mean() > 0.5]
    if len(nan_cols) > 0:
        df.drop(nan_cols, axis=1, inplace=True)
        print(f"\nColumns with more than 50% NaN values removed: {', '.join(nan_cols)}")

    # Print the inferred data types
    print(f"\nNew variables and inferred dtypes:\n{df.dtypes}")

    # Save processed file
    new_data_path = f"{data_path.split('.csv')[0]}_processed.csv"
    df.to_csv(new_data_path, index=False)
    print(f"\nCleaned file saved as {new_data_path}")
