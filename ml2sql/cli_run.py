import sys
import re
from datetime import datetime
import time
from pathlib import Path
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
    data_dir = Path("input") / "data"
    files = sorted([f for f in data_dir.glob("*.csv")])

    print("Files in input/data/:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name}")

    # Ask for CSVPATH
    while True:
        try:
            csv_file_index = (
                int(input("\nSelect CSV file for training the model: ")) - 1
            )
            csv_path = files[csv_file_index]
            break
        except (IndexError, ValueError):
            print("Invalid input. Please enter a valid integer.")

    print(f"CSV file {csv_path} will be used for modelling")

    # Ask for JSONPATH
    while True:
        configuration_dir = Path("input") / "configuration"
        files = sorted(list(configuration_dir.glob("*.json")))
        files.insert(0, Path("Create New Config File"))

        print(f"\n\nFiles in {configuration_dir}:")
        for i, file in enumerate(files, 1):
            if file.name != "Create New Config File":
                last_mod_ts = file.stat().st_mtime
            else:
                last_mod_ts = 0

            if time.time() - last_mod_ts < 10:
                print(f"{i}. {file.name} (New)")
            else:
                print(f"{i}. {file.name}")

        try:
            json_file_index = (
                int(input("\nSelect JSON file for training configuration: ")) - 1
            )

            if json_file_index == 0:
                create_config(csv_path)
            else:
                json_path = files[json_file_index]
                break
        except (IndexError, ValueError):
            print("Invalid input. Please enter a valid integer.")

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
        model_name = input("\nGive it a name: ")
        model_name = re.sub(r"[^0-9a-zA-Z_]+", "", model_name.lower().replace(" ", "_"))
        current_date = datetime.today().strftime("%Y%m%d")
        full_model_name = f"{current_date}_{model_name}"

        model_dir = Path("trained_models") / full_model_name
        if model_dir.exists():
            print("Folder with this name already exists please try another")
        else:
            unique_name = True

    print(f"\nProject name will be: {full_model_name}")

    # Make directory with current data and model name
    try:
        model_dir.mkdir(parents=True)
        (model_dir / "feature_importance").mkdir()
        (model_dir / "feature_info").mkdir()
        (model_dir / "performance").mkdir()
        (model_dir / "model").mkdir()
    except FileExistsError:
        sys.exit("Error: Model directory already exists")

    print("\nStarting script to create model")

    modelcreater(
        data_path=csv_path,
        config_path=json_path,
        model_name=model_type,
        project_name=model_dir,
    )

    print("\nModel outputs can be found in folder:")
    print(model_dir.resolve())
