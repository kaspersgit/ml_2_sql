# Load packages
import sys
import pandas as pd
import json
import logging
import random
from utils.modelling import decision_tree
from utils.modelling import ebm
from utils.modelling import decision_rule
from utils.config_handling import *
from utils.pre_process import *
from utils.output_scripts import decision_tree_as_code
from utils.output_scripts import ebm_as_code

# parse command line arguments
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Enter project name",
                        nargs='?', default='no_name', const='no_name')
    parser.add_argument("--data_path", type=str, help="Enter path to csv file",
                        nargs='?', default='no_data', const='no_data')
    parser.add_argument("--configuration", type=str, help="Enter path to json file",
                        nargs='?')
    parser.add_argument("--model_name", type=str, help="Enter model type",
                        nargs='?', default='decision_tree', const='full')

    args = parser.parse_args()

    # settings
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 10)
    random_seed = 42

    # get given name from the first given argument
    given_name = args.name

    # set logger
    logging.basicConfig(format='%(asctime)s %(message)s', filename=given_name+'/logging.log', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True # ignore matplotlibs font warnings

    # path to data
    data_ = pd.read_csv(args.data_path)

    # get target and features columns
    with open(args.configuration) as json_file:
        configuration = json.load(json_file)

    # get model name
    model_name = args.model_name

    #############################################
    # for debugging
    # given_name='trained_models/kasper'
    # logging.basicConfig(format='%(asctime)s %(message)s', filename=given_name+'/logging.log', encoding='utf-8', level=logging.DEBUG)
    # data_ = pd.read_csv('input/data/example_titanic.csv')
    # with open('input/configuration/example_titanic.json') as json_file:
    #     configuration = json.load(json_file)
    # model_name = 'ebm'
    #############################################

    # Handle the configuration file
    target_col, feature_cols, model_params, pre_params, post_params = config_handling(configuration, logging)

    # pre processing
    # make copy (doesn't change anything but for future use)
    data = data_.copy()

    # Log parameters
    logging.info(f'Configuration file content: \n{configuration}')

    # set model type based on target value
    if data[target_col].nunique() == 1:
        raise Exception("Target column needs more than 1 unique value")
    elif (data[target_col].dtype == 'float') | ((data[target_col].dtype == 'int') & (data[target_col].nunique() > 10)):
        model_type = 'regression'
    else:
        model_type = 'classification'

        print(f'\nTarget column has {data[target_col].nunique()} unique values')
        logging.info(f'\nTarget column has {data[target_col].nunique()} unique values')

    print('\nThis problem will be treated as a {model_type} problem'.format(model_type=model_type))
    logging.info('This problem will be treated as a {model_type} problem'.format(model_type=model_type))

    # pre process data
    datasets = pre_process_kfold(given_name, data, target_col, feature_cols
                                                 , model_name=model_name
                                                 , model_type=model_type
                                                 , logging=logging
                                                 , pre_params=pre_params
                                                 , post_params=post_params
                                                 , random_seed=random_seed)

    # train decision tree and figures and save them
    clf = globals()[model_name].make_model(given_name, datasets, model_type=model_type, model_params=model_params, post_params=post_params, logging=logging)

    # Create SQL version of model and save it
    globals()[model_name + '_as_code'].save_model_and_extras(clf, given_name, post_params['sql_split'], logging)

# Run function
if __name__ == '__main__':
    main()