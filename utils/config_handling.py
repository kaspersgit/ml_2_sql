# Handle the configuration file
import json

def config_handling(configuration, logging):
    # target column
    target_col = configuration['target']

    # features columns
    if 'features' in configuration.keys():
        feature_cols = configuration['features']
        logging.info(f'{len(feature_cols)} features specified in file')
    else:
        # treat all other columns as features
        feature_cols = list(data_.columns)
        feature_cols.remove(target_col)
        logging.info(f'Using {len(feature_cols)} features (all columns except target)')

    # model related parameters
    if 'model_params' in configuration.keys():
        model_params = configuration['model_params']
    else:
        model_params = {}

    # pre processing related parameters
    if 'pre_params' in configuration.keys():
        pre_params = configuration['pre_params']
    else:
        pre_params = {}

    if not ('oot_set' in pre_params.keys()) & ('oot_rows' in pre_params.keys()):
        pre_params['oot_set'] = 'false'

    # Cross validation type to perform
    if 'cv_type' not in pre_params.keys():
        pre_params['cv_type'] = 'kfold_cv'

    # If not present set upsamplling to false
    if 'upsampling' not in pre_params.keys():
        pre_params['upsampling'] = 'false'

    # post modeling related parameters
    if 'post_params' in configuration.keys():
        post_params = configuration['post_params']
    else:
        post_params = {}

    # # unpack calibration from params
    if 'calibrate' in post_params.keys():
        if post_params['calibration'] in ['auto','sigmoid','isotonic','true']:
            post_params['calibration'] = 'auto' if post_params['calibration'] == 'true' else post_params['calibration']
        else:
            post_params['calibration'] = 'false'
    else:
        post_params['calibration'] = 'false'

    # If not present set calibration check if we upsample
    if 'calibration' not in post_params.keys():
        if pre_params['upsampling'] == 'true':
            post_params['calibration'] = 'auto'
        else:
            post_params['calibration'] = 'false'

    return target_col, feature_cols, model_params, pre_params, post_params