import argparse


def SetArgParser():
    """
    Create and configure an argument parser for command-line interface.

    Returns:
    parser: An argparse.ArgumentParser object.

    Usage example:
    parser = SetArgParser()
    args = parser.parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="Enter project name",
        nargs="?",
        default="no_name",
        const="no_name",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Enter path to csv file",
        nargs="?",
        default="no_data",
        const="no_data",
    )
    parser.add_argument(
        "--configuration", type=str, help="Enter path to json file", nargs="?"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Enter model type",
        nargs="?",
        default="decision_tree",
        const="full",
    )

    return parser


def GetArgs(script, argv):
    """
    Parse command-line arguments using argparse and return the parsed arguments.

    Args:
    - script (str): 'main' or 'test_model' dependent on what will run.
    - argv (list of str): List of command-line arguments to parse.

    Returns:
    - args (argparse.Namespace): An object containing the parsed arguments.
      The object has the following attributes:
      - name (str): The name of the project. Default: 'no_name'.
      - data_path (str): The path to the CSV file containing the data.
      - configuration (str): The path to the JSON file containing the model configuration.
      - model_name (str): The name of the model to use.
    """
    parser = argparse.ArgumentParser()

    if script == "main":
        parser.add_argument(
            "--name",
            type=str,
            help="Enter project name",
            nargs="?",
            default="no_name",
            const="no_name",
        )
        parser.add_argument(
            "--data_path",
            type=str,
            help="Enter path to csv file",
            nargs="?",
            const="no_data",
        )
        parser.add_argument(
            "--configuration",
            type=str,
            help="Enter path to json file",
            nargs="?",
            const="no_config",
        )
        parser.add_argument(
            "--model_name", type=str, help="Enter model type", nargs="?", const="full"
        )

    elif script == "create_config":
        parser.add_argument(
            "--data_path",
            type=str,
            help="Enter path to csv file",
            nargs="?",
            const="no_data",
        )

    elif script == "test_model":
        parser.add_argument(
            "--model_path",
            type=str,
            help="Enter path to model",
            nargs="?",
            const="no_name",
        )
        parser.add_argument(
            "--data_path",
            type=str,
            help="Enter path to csv file",
            nargs="?",
            const="no_data",
        )
        parser.add_argument(
            "--destination_path",
            type=str,
            help="Enter path to folder for saving files",
            nargs="?",
            const="no_data",
        )

    args = parser.parse_args(argv)

    return args
