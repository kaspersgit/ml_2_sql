import pandas as pd
import json
import random
from ml2sql.utils.helper_functions.config_handling import select_ml_cols
from ml2sql.utils.helper_functions.parsing_arguments import GetArgs


def get_input(options, message):
    print(f"\n{message}")
    while True:
        for i, option in enumerate(options):
            print(f"{i + 1}. {option}")
        response = input("Enter the number of your choice: ")
        try:
            choice = options[int(response) - 1]
            break
        except IndexError:
            print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    return choice


def create_config(args):
    # Load in data
    csv_path = args.data_path

    # Read in sample of data
    #  try different %, until the nr of rows is at least 500 (or until we use 50% sample size)
    nrows = 0
    p = 0.005
    while nrows < 500:
        # if sample is 50% of all rows then continue
        if p >= 0.5:
            break

        # increase p 10 fold with every loop
        p = p * 10

        # if random from [0,1] interval is greater than p the row will be skipped
        data = pd.read_csv(
            args.data_path,
            keep_default_na=False,
            na_values=["", "N/A", "NULL", "None", "NONE"],
            header=0,
            skiprows=lambda i: i > 0 and random.random() > p,
        )
        nrows = len(data)

    # Create a dictionary to store the features and their indices
    features_dict = {}
    for i, header in enumerate(data.columns):
        features_dict[i] = header

    # Ask the user to choose a target
    print("\nWhich column would you like to use as the target?")
    for i, header in features_dict.items():
        print(f"{i}. {header}")

    while True:
        try:
            target_ind = int(input("\nEnter the index of the target column: "))
            if target_ind in features_dict:
                break
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Remove the target from the list of features
    target_column = features_dict.pop(target_ind)

    config_creation = get_input(
        ["Automatic", "Manual"],
        "Do you want to create config manually or automatically?",
    )
    print(f"{config_creation}\n")

    if config_creation == "Automatic":
        ## Call other script (mainly to not have to execute this script with an active venv)
        # Filter out features which are ID's (all values are unique) and dates
        features_set = select_ml_cols(data)
        features_set.discard(target_column)

        print("\nFinal config:")
        input_params = {
            "features": list(features_set),
            "model_params": {},
            "post_params": {
                "calibration": "false",
                "sql_split": "true",
                "sql_decimals": 15,
                "file_type": "png",
            },
            "pre_params": {
                "cv_type": "notimeseriesplit",
                "max_rows": 50000,
                "time_sensitive_column": "_",
                "upsampling": "false",
            },
            "target": target_column,
        }

        csv_file_name = csv_path.split("/")[-1].split(".")[0]
        config_name = f"{csv_file_name}_auto_config"

    elif config_creation == "Manual":
        ## Original way of creating config

        # Initialize the selected features
        selected_features = []

        # Main loop to prompt the user for feature selection
        while True:
            print("\nFeatures:")
            for i, header in features_dict.items():
                if i in selected_features:
                    print(f"{i}: {header} (selected)")
                else:
                    print(f"{i}: {header}")
            print("\nSelect features:")
            print("1. Choose a feature by index")
            print("2. Deselect a feature by index")
            print("3. Select all features")
            print("4. Deselect all features")
            print("5. Done")
            choice = input("Enter your choice (1-5): ")

            # Check the user's choice
            if choice == "1":
                index = int(
                    input("Enter the index of the feature you want to select: ")
                )
                if index in features_dict.keys() and index not in selected_features:
                    selected_features.append(index)
            elif choice == "2":
                index = int(
                    input("Enter the index of the feature you want to deselect: ")
                )
                if index in selected_features:
                    selected_features.remove(index)
            elif choice == "3":
                selected_features = list(features_dict.keys())
            elif choice == "4":
                selected_features = []
            elif choice == "5":
                break
            else:
                print("Invalid choice. Please try again.")

        # Create the features list from the selected features indices
        features = [features_dict[i] for i in selected_features]

        # Non used columns
        [features_dict.pop(i) for i in selected_features]

        # Fill in the JSON template with the selected features
        input_params = {
            "features": features,
            "model_params": {},
            "post_params": {
                "calibration": "false",
                "sql_split": "true",
                "file_type": "html",
            },
            "pre_params": {
                "cv_type": "notimeseriesplit",
                "max_rows": 100000,
                "time_sensitive_column": "date_column",
                "upsampling": "false",
            },
            "target": target_column,
        }

        # Ask the user to input the value for each key in input_params
        print("\nLet's fill in the rest of the JSON file together!\n")

        for key in input_params:
            if key in ["features", "target"]:
                continue
            elif key == "model_params":
                input_value = input(
                    "Do you have any specific model parameters to set? (yes/no): "
                )
                if input_value.lower() == "yes":
                    model_params = {}
                    model_params_keys = input(
                        "Enter the key for each model parameter, separated by comma: "
                    ).split(",")
                    model_params_values = input(
                        "Enter the value for each model parameter, separated by comma: "
                    ).split(",")
                    for i in range(len(model_params_keys)):
                        model_params[model_params_keys[i].strip()] = (
                            model_params_values[i].strip()
                        )
                    input_params[key] = model_params
                else:
                    print("No model parameters were set.\n")
            else:
                if len(input_params[key]) > 1:
                    for key2 in input_params[key]:
                        if key2 == "calibration":
                            # input_value = get_input(["true","false"], "Calibration (true/false): ")
                            input_value = "false"
                        elif key2 == "sql_split":
                            input_value = get_input(
                                ["true", "false"], "SQL split (true/false): "
                            )
                        elif key2 == "file_type":
                            input_value = get_input(
                                ["html", "png"], "File type (html/png): "
                            )
                        elif key2 == "cv_type":
                            input_value = get_input(
                                ["timeseriessplit", "kfold"],
                                "CV type (timeseriessplit/kfold): ",
                            )
                        elif key2 == "max_rows":
                            input_value = int(input("\nMax rows (number): "))
                        elif key2 == "time_sensitive_column":
                            if (
                                input_params["pre_params"]["cv_type"]
                                == "timeseriessplit"
                            ):
                                if len(features_dict) == 0:
                                    print(
                                        "No columns to select as time sensitive, switching to kfold"
                                    )
                                    input_params["pre_params"]["cv_type"] = "kfold"
                                print(
                                    "\nWhich column would you like to use as the time sensitive column?"
                                )
                                for i, header in features_dict.items():
                                    print(f"{i}. {header}")
                                date_col_index = int(
                                    input(
                                        "Enter the index of the time sensitive column: "
                                    )
                                )
                                input_value = features_dict[date_col_index]
                            else:
                                input_value = "_"
                        elif key2 == "upsampling":
                            # input_value = get_input(["true","false"], "Upsampling (true/false): ")
                            input_value = "false"

                        # Fill in value in dict
                        input_params[key][key2] = input_value

        # name config file
        config_name = input("\nName config file:")
        config_name = config_name.split(".")[0]

    # convert the dictionary to a JSON string
    json_string = json.dumps(input_params, indent=4)

    # Create path to save config file to
    config_path = f"input/configuration/{config_name}.json"

    # write the JSON string to a file
    with open(config_path, "w") as file:
        file.write(json_string)

    # Print the resulting JSON
    print(json_string)

    # print path
    print(f"Config saved in: {config_path}")

    return config_path


if __name__ == "__main__":
    # Get arguments from the CLI
    args = GetArgs("create_config", None)

    # Run main with given arguments
    create_config(args)
