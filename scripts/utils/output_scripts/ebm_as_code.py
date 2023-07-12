import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from contextlib import redirect_stdout

def ReduceSingleFeature(df_):
    # For safety
    df = df_.copy()

    # drop any duplicates (which should not exist)
    df_feat_types = df[['feature','feat_type']].drop_duplicates().reset_index(drop=True).copy()

    # Set index
    df = df.set_index(['feature'])

    # Explode per column 
    lookup_df = df[['feat_bound', 'score',
    'score_lower_bound', 'score_upper_bound']].apply(lambda x: x.apply(pd.Series).stack()).reset_index().drop('level_1', axis=1)

    # add feature type back in
    lookup_df = lookup_df.merge(df_feat_types, on='feature', how='left')

    # sort_values but None will be in the bottom
    lookup_df.sort_values(['feature','feat_bound'], inplace=True, ascending=False)

    # So reverse the ordering
    lookup_df = lookup_df.iloc[::-1]

    lookup_df['nr_features'] = 1

    # being able for score arrays to be grouped
    # for categorical features do not group by score hence we include value in score_hash
    lookup_df['pre_hash_column'] = np.where(lookup_df['feat_type'] == 'categorical'
                                            , lookup_df['score'].astype(str) + lookup_df['feat_bound'].astype(str)
                                            , lookup_df['score'].astype(str))
    lookup_df['score_hash'] = pd.util.hash_pandas_object(lookup_df['pre_hash_column'], index=False)
    hash_lookup_df = lookup_df[['score_hash','score']].drop_duplicates('score_hash').reset_index(drop=True).copy()

    # to group by score but keep them in order (not grouping similar scores for different feature values)
    adj_check = (lookup_df.score_hash != lookup_df.score_hash.shift()).cumsum()

    lookup_df['adj_score'] = adj_check
    lookup_df_simple = lookup_df.groupby(['feature', 'feat_type', 'score_hash', 'adj_score']).agg({'feat_bound':'first'
                                                ,'score_lower_bound':'first'
                                                ,'score_upper_bound':'last'
                                                }, sort=False).reset_index(drop=False)

    lookup_df_grouped = lookup_df_simple.merge(hash_lookup_df, on='score_hash', how='left')
    # ordering sets None in last place
    lookup_df_grouped.sort_values(['feature', 'feat_bound'], inplace=True, ascending=False)
    # So reverse the ordering
    lookup_df_grouped = lookup_df_grouped.iloc[::-1].reset_index(drop=True)

    return lookup_df_grouped


def RestructureReduceInteractions(df_):

    # for safety
    df = df_.copy()

    # Restructure
    # loop over rows
    for index, row in df.iterrows():
        # Swap features position if first feature has more bound values then second feature
        if row['score'].shape[0] > row['score'].shape[1]:
            print(str(row['feature']) + ' will be transponsed')

            # loop over column needed to be transposed
            for c in ['feature', 'feat_bound', 'score', 'feat_type']:
                if isinstance(df.loc[index,c], list):
                    df.at[index,c] = [df.at[index,c][1], df.at[index,c][0]]
                elif isinstance(df.loc[index,c], np.ndarray):
                    df.at[index,c] = np.transpose(df.at[index,c])

    # Reduce

    # Split list valued columns into two columns
    df[['feat_1', 'feat_2']] = df['feature'].apply(pd.Series)
    df[['feat_bound_1', 'feat_bound_2']] = df['feat_bound'].apply(pd.Series)
    df[['feat_type_1', 'feat_type_2']] = df['feat_type'].apply(pd.Series)

    # Drop unused column 
    df = df.drop(['feat_bound','feat_type'], axis=1)

    # Set index
    df = df.set_index(['feature'])

    # first explode on score and feat_bound_1 (same dimensions)
    df_explode_1 = df.explode(['score','feat_bound_1'])
    lookup_df = df_explode_1.explode(['score','feat_bound_2'])

    # sort_values but None will be in the bottom
    lookup_df = lookup_df.sort_values(['feat_1', 'feat_2','feat_bound_1','feat_bound_2'], ascending=False)

    # So reverse the ordering
    lookup_df = lookup_df.iloc[::-1]

    lookup_df['nr_features'] = 2

    # being able for score arrays to be grouped
    # for categorical features do not group by score hence we include value in score_hash
    lookup_df['pre_hash_column'] = lookup_df['score'].astype(str) + lookup_df['feat_bound_1'].astype(str) + lookup_df['feat_bound_2'].astype(str)

    lookup_df['score_hash'] = pd.util.hash_pandas_object(lookup_df['pre_hash_column'], index=False)
    hash_lookup_df = lookup_df[['score_hash','score']].drop_duplicates('score_hash').reset_index(drop=True).copy()

    # to group by score but keep them in order (not grouping similar scores for different feature values)
    adj_check = (lookup_df.score_hash != lookup_df.score_hash.shift()).cumsum()

    lookup_df['adj_score'] = adj_check
    lookup_df_simple = lookup_df.groupby(['feat_1', 'feat_2', 'feat_type_1', 'feat_type_2', 'score_hash', 'adj_score']).agg({'feat_bound_1':'first'
                                                , 'feat_bound_2':'first'
                                                ,'score_lower_bound':'first'
                                                ,'score_upper_bound':'last'
                                                }, sort=False).reset_index(drop=False)

    lookup_df_grouped = lookup_df_simple.merge(hash_lookup_df, on='score_hash', how='left')

    # ordering sets None in last place
    lookup_df_grouped.sort_values(['feat_1', 'feat_2','feat_bound_1','feat_bound_2'], inplace=True, ascending=False)

    # So reverse the ordering
    lookup_df_grouped = lookup_df_grouped.iloc[::-1].reset_index(drop=True)

    return lookup_df_grouped
    
    

 
def extractLookupTable(ebm):
    """ Provides reformatted structure of EBM

    Args:
        ebm: A trained EBM

    Returns:
        A dict of dataframes including information to create SQL code
    """

    # Add per feature graph
    lookup_dicts = []


    for feature_group_index, feature_indexes in enumerate(
            ebm.term_features_
    ):

        # below includes treatment for missing values
        model_graph = ebm.term_scores_[feature_group_index]

        # NOTE: This uses stddev. for bounds, consider issue warnings.
        errors = ebm.standard_deviations_[feature_group_index]

        if len(feature_indexes) == 1:

            # hack. remove the 0th index which is for missing values
            model_graph = model_graph[1:-1]
            errors = errors[1:-1]

            feature_bins = ebm.bins_[feature_group_index][0]

            scores = list(model_graph)
            upper_bounds = list(model_graph + errors)
            lower_bounds = list(model_graph - errors)

            if isinstance(feature_bins, dict):
                # categorical
                feat_bound = list(feature_bins.keys())
                feat_type = 'categorical'
        
            else:
                feat_bound = np.append(feature_bins, np.PINF)
                feat_type = 'numeric'


            score = scores
            score_lower_bound = lower_bounds
            score_upper_bound = upper_bounds

            lookup_dict = {'nr_features': 1
                           , 'feature':ebm.feature_names[feature_indexes[0]]
                           , 'feat_bound': feat_bound
                           , 'score': score
                           , 'score_lower_bound': score_lower_bound
                           , 'score_upper_bound': score_upper_bound
                           , 'feat_type': feat_type
                           }

            lookup_dicts.append(lookup_dict)

        elif len(feature_indexes) == 2:
            # hack. remove the 0th index which is for missing values and remove last value which is just zeros
            model_graph = model_graph[1:-1, 1:-1]

            score = model_graph

            feat_bound = [[]] * len(feature_indexes)
            feat_type = [[]] * len(feature_indexes)

            # Make loop over labels to update the score and features bounds
            # Check for non binary variable
            for b in range(len(feature_indexes)):

                bin_levels = ebm.bins_[feature_indexes[b]]
                feature_bins = bin_levels[min(len(feature_indexes), len(bin_levels)) - 1]
            
                if isinstance(feature_bins, dict):
                    feat_bound[b] = list(feature_bins.keys())
                    feat_type[b] = 'categorical'
                else:
                    feat_bound[b] = np.append(feature_bins, np.PINF) 
                    feat_type[b] = 'numeric'

            lookup_dict = {'nr_features': 2
                           , 'feature': [ebm.feature_names[feature_indexes[0]]
                                        , ebm.feature_names[feature_indexes[1]]]
                           , 'feat_bound': feat_bound
                           , 'score': score
                           , 'feat_type': feat_type
                            # not implemented for interaction terms
                           # , 'score_lower_bound': score_lower_bound
                           # , 'score_upper_bound': score_upper_bound
                           }

            lookup_dicts.append(lookup_dict)

        else:  # pragma: no cover
            raise Exception("Interactions greater than 2 not supported.")


    df_of_lists = pd.DataFrame(lookup_dicts)

    # we treat single and double features differently
    df_of_lists_single = df_of_lists[df_of_lists['nr_features']==1].reset_index(drop=True)
    df_of_lists_double = df_of_lists[df_of_lists['nr_features']==2].reset_index(drop=True)

    # Reduce Single featuers lookup
    lookup_df_single = ReduceSingleFeature(df_of_lists_single)

    # Restructure and reduce double feature df 
    lookup_df_double = RestructureReduceInteractions(df_of_lists_double)

    return {'intercept': ebm.intercept_, 'feature_single': lookup_df_single, 'feature_double': df_of_lists_double}

def lookup_df_to_sql(model_name, df_dict, split=True):
    # Create list for all feature score names
    feature_list = []

    # Start with intercept term
    intercept = df_dict['intercept'][0]
    if not split:
        print(f'"{model_name}" AS model_name \n, ')
        print(f'{intercept} AS intercept')
    elif split:
        # Creating CTE to create table aliases
        print('WITH feature_scores AS (\nSELECT')
        print(f'"{model_name}" AS model_name \n, ')
        print(f'{intercept} AS intercept')

    feature_list.append('intercept')

    # for single feature
    single_features = single_features_2_sql(df_dict['feature_single'])
    feature_list = feature_list + single_features

    if len(df_dict['feature_double']) > 0:
        double_features = double_features_2_sql(df_dict['feature_double'])
        feature_list = feature_list + double_features

    if not split:
        # Sum up all separate scores
        print(', ', end='')
        print(*feature_list, sep=' + ')
        print(' AS score')
        
        # Applying softmax
        print(', EXP(score)/(EXP(score) + 1) AS probability')

    elif split:
        # Add placeholder for source table
        print('FROM <source_table> -- TODO replace with correct table')

        # Close CTE and create next SELECT statement
        print('), add_sum_scores AS (')
        print('SELECT *')

        # Sum up all separate scores
        print(', ', end='')
        print(*feature_list, sep=' + ')
        print(' AS score')
        print('FROM feature_scores')

        # Close CTE and make final Select statement
        print(')')
        print('SELECT *')
        # Applying softmax
        print(', EXP(score)/(EXP(score) + 1) AS probability')
        print('FROM add_sum_scores')

def single_features_2_sql(df):
    feature_nr = 0
    feature_list = []
    for f in df['feature'].unique():
        feature_df = df[df['feature'] == f].reset_index(drop=True)

        # Each feature score as seperate column
        print(',\nCASE')

        single_feature_2_sql(feature_df, f)
        feature_nr += 1
        feature_list.append(f'{f}_score')

        # feature score alias
        print(f'AS {f}_score')

    return feature_list

def single_feature_2_sql(df, feature):
    for index, row in df.iterrows():
        # check if string/category
        if row['feat_type'] == 'categorical':
            # check for manual imputed null values
            print(" WHEN {feature} = '{lb}' THEN {score}".format(feature=feature, lb=row['feat_bound'],
                                                                 score=row['score']))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check for last numeric bound
        elif row['feat_bound'] == np.PINF:
            print(' WHEN {feature} >= {lb} THEN {score}'.format(feature=feature,
                                                                lb=df.iloc[index - 1, :]['feat_bound'],
                                                                score=row['score']))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # otherwise it should be a float
        elif isinstance(row['feat_bound'], float):
            # First bound
            if index == 0:
                print(' WHEN {feature} <= {ub} THEN {score}'.format(feature=feature,
                                                                    ub=row['feat_bound'],
                                                                    score=row['score']))

            else:
                print(' WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}'.format(feature=feature,
                                                                                         lb=
                                                                                         df.iloc[index - 1, :][
                                                                                             'feat_bound'],
                                                                                         ub=row['feat_bound'],
                                                                                         score=row['score']))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        else:
            raise "not sure what to do"

    print('END')

def double_features_2_sql(df):

    feature_list = []

    for df_index, df_row in df.iterrows():

        feature_name = '_x_'.join(df_row['feature'])
        
        # Each feature score in seperate column
        print(',\nCASE')

        double_feature_2_sql(df_index, df_row)

        feature_list.append(f'{feature_name}_score')

        # Feature score as alias
        print(f'AS {feature_name}_score')

    return feature_list

def double_feature_2_sql(df_index, df_row):
    first_feature = df_row['feature'][0]
    second_feature = df_row['feature'][1]

    first_feat_bound = df_row['feat_bound'][0]
    second_feat_bound = df_row['feat_bound'][1]

    first_feat_type = df_row['feat_type'][0]
    second_feat_type = df_row['feat_type'][1]

    first_feature_nbounds = len(df_row['feat_bound'][0])
    second_feature_nbounds = len(df_row['feat_bound'][1])

    scores = pd.DataFrame(df_row['score'])

    # first feature
    for f_ind in range(first_feature_nbounds):
        # check if string/category
        if first_feat_type == 'categorical':
            print(" WHEN {feature} = '{b}' THEN \n  CASE".format(feature=first_feature
                                                                 , b=first_feat_bound[f_ind]))
        elif first_feat_bound[f_ind] == np.PINF:
            print(' WHEN {feature} > {b} THEN \n   CASE'.format(feature=first_feature,
                                                                b=first_feat_bound[f_ind - 1]
                                                                ))

        # otherwise it should be a float
        elif isinstance(first_feat_bound[f_ind], float):
            # Check if first bound
            if f_ind == 0:
                print(' WHEN {feature} <= {b}'.format(feature=first_feature,
                                                      b=first_feat_bound[f_ind]), end='')
                print(' THEN \n  CASE')
            else:
                # After first bound set interval
                print(' WHEN {feature} > {lb} AND {feature} <= {ub} THEN \n     CASE'.format(feature=first_feature,
                                                                                             lb=first_feat_bound[
                                                                                                 f_ind - 1],
                                                                                             ub=first_feat_bound[
                                                                                                 f_ind]))
        else:
            raise "not sure what to do"

        # second feature
        for s_ind in range(second_feature_nbounds):
            # check if string/category
            if second_feat_type == 'categorical':
                print("         WHEN {feature} = '{b}' THEN {score}".format(feature=second_feature,
                                                                            b=second_feat_bound[s_ind],
                                                                            score=scores.iloc[f_ind, s_ind]
                                                                            ))

            elif second_feat_bound[s_ind] == np.PINF:
                print('         WHEN {feature} > {b} THEN {score}'.format(feature=second_feature,
                                                                          b=second_feat_bound[s_ind - 1],
                                                                          score=scores.iloc[f_ind, s_ind]
                                                                          ))

            # otherwise it should be a float
            elif isinstance(second_feat_bound[s_ind], float):
                # still use either first or second bound (based on if they are for made for null handling or not
                if s_ind == 0:
                    print('         WHEN {feature} <= {b} THEN {score}'.format(feature=second_feature,
                                                                               b=second_feat_bound[s_ind],
                                                                               score=scores.iloc[
                                                                                   f_ind, s_ind]))
                else:
                    print('         WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}'.format(
                        feature=second_feature,
                        lb=second_feat_bound[
                            s_ind - 1],
                        ub=second_feat_bound[
                            s_ind],
                        score=scores.iloc[
                            f_ind, s_ind]))


            else:
                raise "not sure what to do"

            # Add ELSE 0.0 as last entry
            if s_ind == (second_feature_nbounds - 1):
                print("         ELSE 0.0")

        print('     END')
    # Catch anything else
    print(' ELSE 0.0')
    print('END')


def lookup_df_to_sql_multiclass(model_name, df, classes, split=True):
    
    print(f'"{model_name}" AS model_name \n, ')
    
    class_nr = 0
    feature_list = {}
    for c in classes:
        feature_nr = 0
        feature_list[c] = []
        for f in df['feature'].unique():
            feature_df = df[df['feature'] == f].reset_index(drop=True)

            if (feature_nr == 0) & (class_nr == 0):
                print('CASE')
            elif (feature_nr == 0) & (class_nr > 0):
                print(', CASE')
            else:
                print(',\nCASE')

            single_feature_2_sql_multiclass(feature_df, f, class_nr)

            # Feature score as alias
            print(f'AS {f}_score_{c}')
            feature_list[c].append(f'{f}_score_{c}')

            feature_nr += 1
        class_nr += 1

    if split:
        # Add placeholder for source table
        print('FROM <source_table> -- TODO replace with correct table')
        # Close CTE and create next SELECT statement
        print('), add_sum_scores AS (')
        print('SELECT *')

    
    if not split:
        for c in classes:
            # Sum up all separate scores
            print(', ', end='')
            print(*feature_list[c], sep=' + ')
            print(f' AS score_{c}')

    elif split:
        for c in classes:
            # Create CTE with sums of all features scores per class
            print(', ', end='')
            print(*feature_list[c], sep=' + ')
            print(f' AS score_{c}')
        # From CTE
        print('FROM feature_scores')

    # Summing feature scores to total score per class
    if not split:
        for c in classes:
            if c == classes[0]:
                print(f', EXP(score_{c})', end='')
            else:
                print(f' + EXP(score_{c})', end='')
        print(' AS total_score')

    elif split:
        # Close CTE and create next SELECT statement
        print('), add_sum_all_scores AS (')
        print('SELECT *')

        for c in classes:
            if c == classes[0]:
                print(f', EXP(score_{c})', end='')
            else:
                print(f' + EXP(score_{c})', end='')
        print(' AS total_score')
        
        # Close CTE and create final SELECT statement
        print('FROM add_sum_scores')
        print(')')
        print('SELECT *')

        
    # Applying softmax
    if not split:
        for c in classes:
            print(f', EXP(score_{c}) / (total_score) AS probability_{c}', end='\n')
    elif split: 
        for c in classes:
            print(f', EXP(score_{c}) / (total_score) AS probability_{c}', end='\n')
        print('FROM add_sum_all_scores')

def single_feature_2_sql_multiclass(df, feature, class_nr):
    for index, row in df.iterrows():
        # check if string/category
        if row['feat_type'] == 'categorical':
            # check for manual imputed null values
            print(" WHEN {feature} = '{lb}' THEN {score}".format(feature=feature, lb=row['feat_bound'],
                                                                 score=row['score'][class_nr]))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check for last numeric bound
        elif row['feat_bound'] == np.PINF:
            print(' WHEN {feature} >= {lb} THEN {score}'.format(feature=feature,
                                                                lb=df.iloc[index - 1, :]['feat_bound'],
                                                                score=row['score'][class_nr]))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # otherwise it should be a float
        elif isinstance(row['feat_bound'], float):
            # First bound
            if index == 0:
                print(' WHEN {feature} <= {ub} THEN {score}'.format(feature=feature,
                                                                    ub=row['feat_bound'],
                                                                    score=row['score'][class_nr]))

            else:
                print(' WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}'.format(feature=feature,
                                                                                         lb=
                                                                                         df.iloc[index - 1, :][
                                                                                             'feat_bound'],
                                                                                         ub=row['feat_bound'],
                                                                                         score=row['score'][class_nr]))
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        else:
            raise "not sure what to do"

    print('END')

def ebm_to_sql(model_name, df, classes, split=True):
    if len(classes) > 2:
        lookup_df_to_sql_multiclass(model_name, df['feature_single'], classes, split=True)
    else:
        lookup_df_to_sql(model_name, df, split)


def save_model_and_extras(ebm, model_name, split, logging):
    # extract lookup table from EBM
    lookup_df = extractLookupTable(ebm)

    # In case of regression
    if not hasattr(ebm, 'classes_'):
        ebm.classes_ = [0]
        lookup_df['intercept'] = [lookup_df['intercept']]

    # Write printed output to file
    with open('{model_name}/model/ebm_in_sql.sql'.format(model_name=model_name), 'w') as f:
        with redirect_stdout(f):
            model_name = model_name.split('/')[1]
            ebm_to_sql(model_name, lookup_df, ebm.classes_, split)
    print('SQL version of EBM saved')
    logging.info('SQL version of EBM saved')

if __name__ == '__main__':
    # Used for testing faster
    import joblib 

    # Variables
    split = False 
    model_name = 'easy_there'
    
    # Load model in joblibd format
    filename = f'../trained_models/20230705_test_upgrade_v5/model/ebm_classification.sav'

    ebm = joblib.load(open(filename, 'rb'))

    # extract lookup table from EBM
    lookup_df = extractLookupTable(ebm)

    # In case of regression
    if not hasattr(ebm, 'classes_'):
        ebm.classes_ = [0]
        lookup_df['intercept'] = [lookup_df['intercept']]

    # Print output
    ebm_to_sql(model_name, lookup_df, ebm.classes_, split=split)
    