import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC

def cleanAndCastColumns(data, feature_cols, target_col, model_type, logging):
    # make copy of data
    _data = data.copy()

    # clean out where target is NaN
    print('\nRows being removed due to NaN in target column: {nans} \n'.format(nans=len(_data[_data[target_col].isna()])))
    logging.info('Rows being removed due to NaN in target column: {nans} \n'.format(nans=len(_data[_data[target_col].isna()])))

    _data = _data.loc[_data[target_col].notna(), :].reset_index(drop=True)

    # Clean out feature which only has 1 unique value
    one_nunique = _data[feature_cols].columns[_data[feature_cols].nunique() == 1]

    if len(one_nunique) > 0:
        _data = _data.loc[:, ~_data.columns.isin(one_nunique)]
        print(f'\nFeatures being removed due to single unique value: {one_nunique} \n')
        logging.info(f'Features being removed due to single unique value: {one_nunique} \n')

        # Remove feature from feature_cols if in there
        for f in one_nunique:
            if f in feature_cols:
                feature_cols.remove(f)

    if model_type == 'classification':
        # Remove classes with only one occurence
        one_occurence_class = _data[target_col].value_counts()[_data[target_col].value_counts() == 1].index
        if len(one_occurence_class) > 0:
            _data = _data[~_data[target_col].isin(one_occurence_class)]

            print('Removed class {classification} due to having only 1 observation \n'.format(classification=one_occurence_class))
            logging.info('Removed class {classification} due to having only 1 observation'.format(classification=one_occurence_class))


    # Imputing missing values and casting
    # assuming only int/float and bool column types
    nan_cols = _data[feature_cols][_data[feature_cols].columns[_data[feature_cols].isna().sum() > 0]]
    if len(nan_cols) > 0:
        print('Columns with NaN values: \n{nans} \n'.format(nans=nan_cols.isna().sum()))
        logging.info('Columns with NaN values: \n{nans}'.format(nans=nan_cols.isna().sum()))

    # Adjust this to allow for categorical features
    for col in _data[feature_cols].select_dtypes(include=['object']).columns:
        # Check if values are true/false/None then boolean
        if all(val in [True, False, None, np.NaN] for val in _data[col].unique()):
            _data[col] = _data[col].astype(int)
        else:  # otherwise assume categorical
            _data[col] = _data[col].astype({col: 'category'})

    # Overview of _data types
    print('Column types in data set (including target)\n{col_types} \n'.format(col_types=_data[feature_cols].dtypes.value_counts()))

    # change boolean into 0's and 1's
    for col in _data[feature_cols].select_dtypes(include=['bool']).columns:
        _data[col] = _data[col].astype(int)

    return _data.reset_index(drop=True)


def imbalanceness(labels):
    classes_count = labels.value_counts()
    max_class_size = classes_count.max()
    min_class_size = classes_count.min()
    total_size = classes_count.sum()
    nclasses = len(classes_count)

    return (max_class_size - min_class_size)/(total_size - nclasses)

def pre_process_simple(data, target_col, feature_cols, logging, random_seed=42):
    data = cleanAndCastColumns(data, feature_cols, target_col)

    # Warning is caused when a class has very few records and stratisfy is used
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[target_col], test_size=0.25, random_state=random_seed, stratify=data[target_col])

    # pre oversampling
    print('Counts of train labels before resampling: \n{labels} \n\n'.format(labels = y_train.value_counts()))
    logging.info('Counts of train labels before resampling: \n{labels} \n\n'.format(labels = y_train.value_counts()))

    # oversample train data
    try:
        ros = SMOTE(random_state=random_seed)
        X_train_ups, y_train_ups = ros.fit_resample(X_train, y_train)
        print('SMOTE oversampling')
        logging.info('SMOTE oversampling')
    except:
        ros = RandomOverSampler(random_state=random_seed)
        X_train_ups, y_train_ups = ros.fit_resample(X_train, y_train)
        print('Random oversampling')
        logging.info('Random oversampling')

    # post oversampling
    print('Counts of train labels after resampling: \n{labels} \n\n'.format(labels = y_train_ups.value_counts()))
    logging.info('Counts of train labels after resampling: \n{labels}'.format(labels = y_train_ups.value_counts()))

    return X_train, X_train_ups, X_test, y_train, y_train_ups, y_test

def pre_process_kfold(data, target_col, feature_cols, model_type, logging, pre_params, random_seed=42):

    # clean and cast
    data_clean = cleanAndCastColumns(data, feature_cols, target_col, model_type, logging)

    # Limit dataset with respect to the max_rows parameter
    if 'max_rows' in pre_params:
        max_rows = pre_params['max_rows']
        data_clean = data_clean.sample(n=max_rows).reset_index(drop=True)
        print(f'Limited dataset to {max_rows}')
        logging.info(f'Limited dataset to {max_rows}')

    # create kfolds in a statified manner
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

    # initiate kfold
    if pre_params['cv_type'] == 'timeseriesplit':
        print('Performing time series split cross validation')
        logging.info('Performing time series split cross validation')
        data.sort_values(pre_params['time_sensitive_column'], inplace=True)
        kfold = TimeSeriesSplit(n_splits=5)
    elif model_type == 'classification':
        print('Performing stratified kfold cross validation')
        logging.info('Performing stratified kfold cross validation')
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    elif model_type == 'regression':
        print('Performing normal kfold cross validation')
        logging.info('Performing normal kfold cross validation')
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Create initial dict to collect datasets
    datasets = {}

    # set X and y data apart
    y = data_clean[target_col]
    X = data_clean[feature_cols]

    if pre_params['upsampling'] != 'false':
        #### Create upsampled version of full dataset for final training
        # Make sure total rows * columns after upsampling won't hit x nr of cells
        max_cells = 50000
        X_trim, y_trim = trimPreUpsampleDataRows(X, y, max_cells, logging)

        # upsample by trying SMOTE algo
        X_ups, y_ups = upsampleData(X, y_trim, model_type, logging, random_seed=42)

        # Add to datasets
        datasets['final_train'] = {'X': X_ups, 'y': y_ups}
    else:
        # Add to datasets
        datasets['final_train'] = {'X': X, 'y': y}
    ####


    #### Create OOT dataset if wanted
    if pre_params['oot_set'] != 'false':
        oot_df = data_clean.sort_values(pre_params['time_sensitive_column'], ascending=True).tail(pre_params['oot_rows'])
        X_oot = oot_df[feature_cols]
        y_oot = oot_df[target_col]

        datasets['oot'] = {'X': X_oot, 'y': y_oot}

        # set new X and y data apart (oot excluded)
        data_wo_oot = pd.concat([data_clean, oot_df]).drop_duplicates(keep=False)
        X = data_wo_oot[feature_cols]
        y = data_wo_oot[target_col]
    ####


    #### Create datasets based on the different folds
    # listing the different folds
    X_train_list, X_train_ups_list, X_test_list, y_train_list, y_train_ups_list, y_test_list = list(), list(), list(), list(), list(), list()

    # enumerate the splits and summarize the distributions
    kfold_nr = 0
    for train_ix, test_ix in kfold.split(X, y):
        # Record kfold
        print(f'\nCreating fold nr {kfold_nr+1}')
        logging.info(f'Creating fold nr {kfold_nr+1}')

        # select rows
        X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
        y_train, y_test = y[train_ix], y[test_ix]

        if pre_params['upsampling'] != 'false':
            # report on nr rows
            print(f'Nr rows pre trimming: {len(X_train)}')
            print(f'imbalanceness pre trimming: \n {y_train.value_counts()}')
            logging.info(f'Nr rows pre trimming: {len(X_train)}')
            logging.info(f'imbalanceness pre trimming: \n {y_train.value_counts()}')

            X_train_trim, y_train_trim = trimPreUpsampleDataRows(X_train, y_train, max_cells, logging)

            # Nr rows after trimming down dataset
            print(f'Nr rows pre upsampling: {len(X_train_trim)}')
            print(f'imbalanceness pre upsampling: \n {y_train_trim.value_counts()}')
            logging.info(f'Nr rows pre upsampling: {len(X_train_trim)}')
            logging.info(f'imbalanceness pre upsampling: \n {y_train_trim.value_counts()}')

            # upsample by trying SMOTE algo
            X_train, y_train = upsampleData(X_train_trim, y_train_trim, model_type, logging, random_seed=42)

        # Nr rows of training set
        print(f'Nr rows train set: {len(X_train)}')
        print(f'Nr rows test set: {len(X_test)}')
        print(f'imbalanceness train: \n {y_train.value_counts()}')
        logging.info(f'Nr rows train set: {len(X_train)}')
        logging.info(f'Nr rows test set: {len(X_test)}')
        logging.info(f'imbalanceness train: \n {y_train.value_counts()}')

        # append to the lists
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

        kfold_nr += 1

    # add datasets
    datasets['cv_train'] = {'X': X_train_list, 'y': y_train_list}
    datasets['cv_test'] = {'X': X_test_list, 'y': y_test_list}
    ####

    return datasets


def upsampleData(X, y, model_type, logging, random_seed=42):
    if model_type == 'classification':
        # oversample train data
        # nested try except (https://stackoverflow.com/questions/17015230/are-nested-try-except-blocks-in-python-a-good-programming-practice)

        try:
            categorical_cols = X.dtypes == 'category'
            if categorical_cols.sum() > 0:
                ros = SMOTENC(categorical_features=categorical_cols, random_state=random_seed)
                X_ups, y_ups = ros.fit_resample(X, y)
                print('SMOTE-NC oversampling')
                logging.info('SMOTE-NC oversampling')
            else:
                ros = SMOTE(random_state=random_seed)
                X_ups, y_ups = ros.fit_resample(X, y)
                print('SMOTE oversampling')
                logging.info('SMOTE oversampling')
        except:
            ros = RandomOverSampler(random_state=random_seed)
            X_ups, y_ups = ros.fit_resample(X, y)
            print('Random oversampling')
            logging.info('Random oversampling')

    elif model_type == 'regression':
        X_ups, y_ups = X, y

    return X_ups, y_ups

def trimDownDataRows(X, y, max_cells, logging):
    # Trim dataset if necessary based on amount of cells (columns x rows)
    nr_cells = X.shape[0] * X.shape[1]
    if nr_cells > max_cells:
        print(f'Dataset shape {X.shape} resulting in {nr_cells} cells \nTrimming down...')
        logging.info(f'Dataset shape {X.shape} resulting in {nr_cells} cells \nTrimming down...')
        df_pretrim = X.join(y)
        df_posttrim = df_pretrim.sample(n=round(max_cells / X.shape[1]))
        X_trim = df_posttrim[X.columns].reset_index(drop=True)
        y_trim = df_posttrim[y.name].reset_index(drop=True)
        nr_cells_trim = X_trim.shape[0] * X_trim.shape[1]
        print(f'Trimmed down to {X_trim.shape} resulting in {nr_cells_trim} cells.')
        logging.info(f'Trimmed down to {X_trim.shape} resulting in {nr_cells_trim} cells.')
    else:
        X_trim = X
        y_trim = y

    return X_trim, y_trim

def trimPreUpsampleDataRows(X, y, max_cells, logging):
    # reset index
    X_ = X.reset_index(drop=True)
    y_ = y.reset_index(drop=True)

    # Trim dataset if necessary based on amount of cells (columns x rows)
    classes_counts = y_.value_counts()
    nr_classes = len(classes_counts)
    size_majority_class = classes_counts.max()

    exp_nr_cells = size_majority_class * X_.shape[1] # * nr_classes
    max_rows = round(max_cells/(X_.shape[1])) # * nr_classes

    if exp_nr_cells > max_cells:
        print(f'Expecting {exp_nr_cells} cells, more than set limit of {max_cells}')
        logging.info(f'Expecting {exp_nr_cells} cells, more than set limit of {max_cells}')
        big_classes = classes_counts.index[classes_counts > max_rows]

        y_trim = y_[~y_.isin(big_classes)]
        for c in big_classes:
            y_big = y_[y_==c]
            y_big_trim = y_big.sample(max_rows)
            y_trim = y_trim.append(y_big_trim)
        y_trim = y_trim.sort_index()
        X_trim = X_[X_.index.isin(y_trim.index)]
    else:
        y_trim = y_
        X_trim = X_

    print(f'Original nr of rows {len(X_)} \nNew nr of rows {len(X_trim)}')
    logging.info(f'Original nr of rows {len(X_)} \nNew nr of rows {len(X_trim)}')

    return X_trim.reset_index(drop=True), y_trim.reset_index(drop=True)
