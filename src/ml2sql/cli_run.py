import os
import sys
import re
from datetime import datetime
import time
from ml2sql.utils.modelcreater import modelcreater
from ml2sql.utils.create_config import create_config


def cli_run():
    # ASCII art
    ml2sql = r"""
    `7MMM.     ,MMF'`7MMF'                        .M"'"bgd   .g8""8q. `7MMF'
    MMMb    dPMM    MM                         ,MI    "Y .dP'    `YM. MM
    M YM   ,M MM    MM             pd*"*b.     `MMb.     dM'      `MM MM
    M  Mb  M' MM    MM            (O)   j8       `YMMNq. MM        MM MM
    M  YM.P'  MM    MM      ,         ,;j9     .     `MM MM.      ,MP MM      ,
    M  `YM'   MM    MM     ,M      ,-='        Mb     dM `Mb.    ,dP' MM     ,M
    .JML. `'  .JMML..JMMmmmmMMM     Ammmmmmm     P"Ybmmd"    `"bmmd"' .JMMmmmmMMM
                                                                MMb
                                                                `bood'
    """

    print(ml2sql)
    print("\n\n")

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
            csv_path = os.path.join(data_dir, files[csv_file_index])
            break
        except IndexError:
            print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"CSV file {csv_path} will be used for modelling")

    # Ask for JSONPATH
    while True:
        # List files in input/configuration/ directory
        files = []
        configuration_dir = "input/configuration/"
        for f in os.listdir(configuration_dir):
            if f.endswith(".json"):
                files.append(f)
        files.sort()

        # Include option to create a new config file
        files.insert(0, "Create New Config File")

        # List the configuration files
        print(f"\n\nFiles in {configuration_dir}:")
        for i, file in enumerate(files, 1):
            if file != "Create New Config File":
                last_mod_ts = os.path.getmtime(configuration_dir + file)
            else:
                last_mod_ts = 0

            if time.time() - last_mod_ts < 10:
                print(f"{i}. {file} (New)")
            else:
                print(f"{i}. {file}")

        # prompt user to select config file
        json_file_index = input("\nSelect JSON file for training configuration: ")
        try:
            json_file_index = int(json_file_index) - 1

            if json_file_index == 0:
                create_config(csv_path)
            else:
                json_path = os.path.join(configuration_dir, files[json_file_index])
                break
        except IndexError:
            print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"Configuration file {json_path} will be used for modelling")

    # Ask for MODEL_TYPE
    model_types = [
        "Explainable Boosting Machine",
        "Decision Tree",
        "Logistic/Linear regression",
    ]
    model_type = None
    while model_type is None:
        print("\nWhat type of model do you want?")
        for i, t in enumerate(model_types, 1):
            print(f"{i}. {t}")

        model_type_index = input("\nChoose a number: ")
        try:
            model_type_index = int(model_type_index) - 1
            model_type = model_types[model_type_index]

        except (ValueError, IndexError):
            print("Invalid option")

    print(f"Algorithm chosen for modelling: {model_type}")

    # Rename model choice for easier reference in code later on
    model_type = "ebm" if model_type == "Explainable Boosting Machine" else model_type
    model_type = (
        "l_regression" if model_type == "Logistic/Linear regression" else model_type
    )
    model_type = model_type.lower().replace(" ", "_")

    # Model name
    unique_name = False
    while not unique_name:
        # ask user to give model run a name
        model_name = input("\nGive it a name: ")

        # Make sure model name only has letters numbers and underscores
        model_name = model_name.lower().replace(" ", "_")
        model_name = re.sub("[^0-9a-zA-Z_]+", "", model_name)

        # Current date
        current_date = datetime.today().strftime("%Y%m%d")

        full_model_name = f"{current_date}_{model_name}"

        # Check if folder already exists with this name
        if os.path.isdir(f"trained_models/{full_model_name}"):
            print("Folder with this name already exists please try another")
        else:
            unique_name = True

    print(f"\nProject name will be: {full_model_name}")

    # Make directory with current data and model name
    try:
        os.makedirs(f"trained_models/{full_model_name}")
        os.makedirs(f"trained_models/{full_model_name}/feature_importance")
        os.makedirs(f"trained_models/{full_model_name}/feature_info")
        os.makedirs(f"trained_models/{full_model_name}/performance")
        os.makedirs(f"trained_models/{full_model_name}/model")
    except FileExistsError:
        sys.exit("Error: Model directory already exists")

    print("\nStarting script to create model")

    modelcreater(
        data_path=csv_path,
        config_path=json_path,
        model_name=model_type,
        project_name=f"trained_models/{full_model_name}",
    )

    print("\nModel outputs can be found in folder:")
    print(f"{os.getcwd()}/trained_models/{full_model_name}")
