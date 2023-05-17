import os
import sys
from datetime import datetime

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
files = os.listdir(data_dir)
files.sort()

print("Files in input/data/:")
for i, file in enumerate(files, 1):
    print(f"{i}. {file}")

# Ask for CSVPATH
csv_path = None
while csv_path is None:
    csv_file_index = input("\nSelect path to csv file for training the model: ")
    try:
        csv_file_index = int(csv_file_index) - 1
        csv_path = os.path.join(data_dir, files[csv_file_index])
    except (ValueError, IndexError):
        print("Invalid option, try again.")

print(f"CSV file {csv_path} will be used for modelling")


# List files in input/configuration/ directory
configuration_dir = "input/configuration/"
files = os.listdir(configuration_dir)
files.sort()

print(f"\n\nFiles in {configuration_dir}:")
for i, file in enumerate(files, 1):
    print(f"{i}. {file}")

# Ask for JSONPATH
json_path = None
while json_path is None:
    json_file_index = input("\nSelect path to json file for model configuration: ")
    try:
        json_file_index = int(json_file_index) - 1
        json_path = os.path.join(configuration_dir, files[json_file_index])
    except (ValueError, IndexError):
        print("Invalid option, try again.")

print(f"Configuration file {json_path} will be used for modelling")

# Ask for MODEL_TYPE
model_types = ["Explainable Boosting Machine", "Decision Tree", "Decision Rule", "Logistic/Linear regression"]
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
model_type = 'ebm' if model_type == 'Explainable Boosting Machine' else model_type
model_type = 'l_regression' if model_type == 'Logistic/Linear regression' else model_type
model_type = model_type.lower().replace(" ", "_")

# Model name
model_name = input("\nGive it a name: ")

# Current date
current_date = datetime.today().strftime('%Y%m%d')

full_model_name = f"{current_date}_{model_name}"

# Make directory with current data and model name
try:
    os.makedirs(f"trained_models/{full_model_name}")
    os.makedirs(f"trained_models/{full_model_name}/feature_importance")
    os.makedirs(f"trained_models/{full_model_name}/feature_info")
    os.makedirs(f"trained_models/{full_model_name}/performance")
    os.makedirs(f"trained_models/{full_model_name}/model")
except FileExistsError:
    sys.exit("Error: Model directory already exists")

print("Starting script to create model")

# Run script with given input using python in the venv (so venv does not need to be activated beforehand)
print("\nRunning command:")
if sys.platform == 'win32':
    command = f".ml2sql/Scripts/python main.py --name trained_models/{full_model_name} --data_path {csv_path} --configuration {json_path} --model {model_type}"
else:
    command = f".ml2sql/bin/python main.py --name trained_models/{full_model_name} --data_path {csv_path} --configuration {json_path} --model {model_type}"

print(command)

os.system(command)

print("\nModel outputs can be found in folder:")
print(f"{os.getcwd()}/trained_models/{full_model_name}")