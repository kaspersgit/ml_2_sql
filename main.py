# Load packages
import logging

from utils.modelling.main_modeler import *

# The translations to SQL (grey as we refer to them dynamically)
from utils.output_scripts import decision_tree_as_code
from utils.output_scripts import decision_rule_as_code
from utils.output_scripts import ebm_as_code

from utils.helper_functions.config_handling import *
from utils.helper_functions.parsing_arguments import *
from utils.pre_process import *

def main(args):
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
    clf = make_model(given_name, datasets, model_name=model_name, model_type=model_type, model_params=model_params, post_params=post_params, logging=logging)
    # clf = globals()[model_name].make_model(given_name, datasets, model_type=model_type, model_params=model_params, post_params=post_params, logging=logging)

    # Create SQL version of model and save it
    globals()[model_name + '_as_code'].save_model_and_extras(clf, given_name, post_params['sql_split'], logging)

# Run function
if __name__ == '__main__':

    # settings
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 10)
    random_seed = 42

    # Command line arguments used for testing
    argvals = '--name trained_models/20221210_titaninc_comeon ' \
              '--data_path input/data/example_titanic.csv ' \
              '--configuration input/configuration/example_titanic.json ' \
              '--model ebm'.split() # example of passing test params to parser

    # For production (only comment for testing)
    argvals = None

    # Get arguments from the CLI
    args = GetArgs(argvals)

    # Run main with given arguments
    main(args)