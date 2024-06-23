import os
import re
import sys
from ml2sql.utils.modeltester import modeltester


def cli_check_model():
    # Select data
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
    csv_path = None
    while csv_path is None:
        csv_file_index = input("\nSelect CSV file for testing the model: ")
        try:
            csv_file_index = int(csv_file_index) - 1
            csv_path = os.path.join(data_dir, files[csv_file_index])
        except (ValueError, IndexError):
            print("Invalid option, try again.")

    print(f"CSV file {csv_path} will be used for testing model")

    # Select model
    # List files in trained_models/ directory
    model_dir = "trained_models/"
    files = []

    for root, dirs, filenames in os.walk(model_dir):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            for filename in os.listdir(subdir):
                if filename.endswith(".sav"):
                    files.append(os.path.join(subdir, filename))

    files.sort()

    print(f"\nModels in {model_dir}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {re.sub(model_dir, '', file)}")

    # Ask for ModelPath
    model_path = None
    while model_path is None:
        model_file_index = input("\nSelect model to apply on test data: ")
        try:
            model_file_index = int(model_file_index) - 1
            model_path = os.path.join(model_dir, files[model_file_index])
            model_path = files[model_file_index]
            model_path = model_path.replace(os.sep, "/")

        except (ValueError, IndexError):
            print("Invalid option, try again.")

    print(f"Model {model_path} will be used for testing")

    # Create extra folder in trained model folder
    # Make directory with current data and model name
    model_folder = model_path.split("trained_models/")[1].split("/")[0]
    csv_name = csv_path.split("/")[-1].split(".")[0]
    destination_path = f"trained_models/{model_folder}/tested_datasets/{csv_name}"

    if not os.path.exists(destination_path.split(csv_name)[0]):
        os.makedirs(destination_path.split(csv_name)[0])

    try:
        os.makedirs(destination_path)
        os.makedirs(f"{destination_path}/performance")
        os.makedirs(f"{destination_path}/local_explanations")

    except FileExistsError as e:
        sys.exit(f"{e}\nDataset is already tested on this model: {destination_path}")

    print("Starting script to test model")

    modeltester(
        data_path=csv_path, model_path=model_path, destination_path=destination_path
    )

    print("\nModel performance outputs can be found in folder:")
    print(f"{os.getcwd()}/trained_models/{model_folder}/tested_datasets/{csv_name}/")


if __name__ == "__main__":
    cli_check_model()
