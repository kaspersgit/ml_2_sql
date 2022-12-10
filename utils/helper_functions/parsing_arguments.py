import argparse

def SetArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Enter project name",
                        nargs='?', default='no_name', const='no_name')
    parser.add_argument("--data_path", type=str, help="Enter path to csv file",
                        nargs='?', default='no_data', const='no_data')
    parser.add_argument("--configuration", type=str, help="Enter path to json file",
                        nargs='?')
    parser.add_argument("--model_name", type=str, help="Enter model type",
                        nargs='?', default='decision_tree', const='full')

    return parser

def GetArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Enter project name",
                        nargs='?', default='no_name', const='no_name')
    parser.add_argument("--data_path", type=str, help="Enter path to csv file",
                        nargs='?', default='no_data', const='no_data')
    parser.add_argument("--configuration", type=str, help="Enter path to json file",
                        nargs='?')
    parser.add_argument("--model_name", type=str, help="Enter model type",
                        nargs='?', default='decision_tree', const='full')

    args = parser.parse_args(argv)

    return args