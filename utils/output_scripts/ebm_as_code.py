import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from contextlib import redirect_stdout

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
            ebm.feature_groups_
    ):
        # below includes treatment for missing values
        model_graph = ebm.additive_terms_[feature_group_index]

        # NOTE: This uses stddev. for bounds, consider issue warnings.
        errors = ebm.term_standard_deviations_[feature_group_index]

        if len(feature_indexes) == 1:

            # hack. remove the 0th index which is for missing values
            model_graph = model_graph[1:]
            errors = errors[1:]

            bin_labels = ebm.preprocessor_._get_bin_labels(feature_indexes[0])

            scores = list(model_graph)
            upper_bounds = list(model_graph + errors)
            lower_bounds = list(model_graph - errors)

            # Check for non categorical variable (binary or strings)
            if (len(bin_labels) > 2) and not (all(isinstance(n, str) for n in bin_labels)):
                feat_bound = bin_labels[1:-1] + [np.PINF]
                feat_type = 'numeric'
            else:
                feat_bound = bin_labels
                feat_type = 'categorical'

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
            # hack. remove the 0th index which is for missing values
            model_graph = model_graph[1:, 1:]
            # errors = errors[1:, 1:]  # NOTE: This is commented as it's not used in this branch.

            score = model_graph

            bin_labels = [ebm.pair_preprocessor_._get_bin_labels(feature_indexes[0]),(ebm.pair_preprocessor_._get_bin_labels(feature_indexes[1]))]

            feat_bound = [[]] * len(bin_labels)
            feat_type = [[]] * len(bin_labels)

            # not implemented for interaction terms
            # score_lower_bound = [[]] * len(bin_labels)
            # score_upper_bound = [[]] * len(bin_labels)

            # Make loop over labels to update the score and features bounds
            # Check for non binary variable
            for b in range(len(bin_labels)):
                if (len(bin_labels[b]) > 2) and not (all(isinstance(n, str) for n in bin_labels[b])):
                    feat_bound[b] = bin_labels[b][1:-1] + [np.PINF]
                    feat_type[b] = 'numeric'
                else:
                    feat_bound[b] = bin_labels[b]
                    feat_type[b] = 'categorical'

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

    # All transformations below only applied on the single feature
    # we leave the interaction (double feature) df as it is and has to be treated
    # differently in the 2sql function
    df_feat_types = df_of_lists_single[['feature','feat_type']].drop_duplicates().reset_index(drop=True).copy()

    df_of_lists_single = df_of_lists_single.set_index(['feature'])
    lookup_df = df_of_lists_single[['feat_bound', 'score',
       'score_lower_bound', 'score_upper_bound']].apply(pd.Series.explode).reset_index()

    # add feature type back in
    lookup_df = lookup_df.merge(df_feat_types, on='feature', how='left')

    # sort_values but None will be in the bottom
    lookup_df.sort_values(['feature','feat_bound'], inplace=True, ascending=False)
    # So reverse the ordering
    lookup_df = lookup_df.iloc[::-1]

    lookup_df['nr_features'] = 1

    # being able for score arrays to be grouped
    lookup_df['score_hash'] = pd.util.hash_pandas_object(lookup_df['score'].astype(str), index=False)
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

    return {'intercept': ebm.intercept_, 'feature_single': lookup_df_grouped, 'feature_double': df_of_lists_double}

def lookup_df_to_sql(df_dict, split=True):
    # Create list for all feature score names
    feature_list = []

    # start with intercept term
    if not split:
        print('{intercept} -- intercept'.format(intercept=df_dict['intercept'][0]))
    elif split:
        print('{intercept} AS intercept'.format(intercept=df_dict['intercept'][0]))

    feature_list.append('intercept')

    # for single feature
    single_features = single_features_2_sql(df_dict['feature_single'], split)
    feature_list = feature_list + single_features

    if len(df_dict['feature_double']) > 0:
        double_features = double_features_2_sql(df_dict['feature_double'], split)
        feature_list = feature_list + double_features

    if not split:
        print('AS score')
    elif split:
        # sum up all separate scores
        print(', ', end='')
        print(*feature_list, sep=' + ')
        print(' AS score')

    # Applying softmax
    print(', EXP(score)/(EXP(score) + 1) AS probability')

def single_features_2_sql(df, split=True):
    feature_nr = 0
    feature_list = []
    for f in df['feature'].unique():
        feature_df = df[df['feature'] == f].reset_index(drop=True)

        if not split:
            print('+\nCASE')
        elif split:
            print(',\nCASE')

        single_feature_2_sql(feature_df, f)
        feature_nr += 1
        feature_list.append(f'{f}_score')

        if split:
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

def double_features_2_sql(df, split=True):

    feature_list = []

    for df_index, df_row in df.iterrows():

        feature_name = '_x_'.join(df_row['feature'])
        if not split:
            print('+\nCASE')
        elif split:
            print(',\nCASE')

        double_feature_2_sql_old(df_index, df_row)

        feature_list.append(f'{feature_name}_score')

        if split:
            print(f'AS {feature_name}_score')

    return feature_list

def double_feature_2_sql_old(df_index, df_row):
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


def lookup_df_to_sql_multiclass(df, classes):
    class_nr = 0
    for c in classes:
        feature_nr = 0
        for f in df['feature'].unique():
            feature_df = df[df['feature'] == f].reset_index(drop=True)

            if (feature_nr == 0) & (class_nr == 0):
                print('CASE')
            elif (feature_nr == 0) & (class_nr > 0):
                print(', CASE')
            else:
                print('+\nCASE')

            for index, row in feature_df.iterrows():
                # check for last numeric bound
                if row['feat_bound'] == np.PINF:
                    print(' WHEN {feature} >= {lb} THEN {score}'.format(feature=f,
                                                                            lb=feature_df.iloc[index - 1, :]['feat_bound'],
                                                                            score=row['score'][class_nr]))
                # check if string/category
                elif isinstance(row['feat_bound'], str):
                    print(" WHEN {feature} = '{lb}' THEN {score}".format(feature=f, lb=row['feat_bound'],
                                                                             score=row['score'][class_nr]))
                # check if boolean casted to int
                elif row['feat_bound'] == int(row['feat_bound']):
                    print(' WHEN {feature}::INT = {lb} THEN {score}'.format(feature=f, lb=row['feat_bound'],
                                                                                score=row['score'][class_nr]))
                # otherwise it should be a float
                elif isinstance(row['feat_bound'], float):
                    if index == 0:
                        # still use either first or second bound (based on if they are for made for null handling or not
                        if row['feat_bound'] > 0:
                            print(' WHEN {feature} <= {ub} THEN {score}'.format(feature=f,
                                                                                         ub=row['feat_bound'],
                                                                                         score=row['score'][class_nr]))
                    else:
                        # still use either first or second bound (based on if they are for made for null handling or not
                        if (index == 1) & (row['feat_bound'] > 0) & (feature_df.iloc[index-1, :]['feat_bound'] < 0):
                            print(' WHEN {feature} <= {ub} THEN {score}'.format(feature=f,
                                                                                         ub=row['feat_bound'],
                                                                                         score=row['score'][class_nr]))

                        print(' WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}'.format(feature=f,
                                                                                                     lb=feature_df.iloc[index-1, :]['feat_bound'],
                                                                                                     ub=row['feat_bound'],
                                                                                                     score=row['score'][class_nr]))
                else:
                    raise "not sure what to do"

            print('END')

            feature_nr += 1

        print('AS {classification}'.format(classification=c))

        class_nr += 1

    for c in classes:
        if c == classes[0]:
            print(', EXP({classification})'.format(classification=c), end='')
        else:
            print(' + EXP({classification})'.format(classification=c), end='')
    print(' AS total_score')

    for c in classes:
        print(', EXP({classification}) / (total_score) AS probability_{classification}'.format(classification=c), end='\n')

def ebm_to_sql(df, classes, split=True):
    if len(classes) > 2:
        lookup_df_to_sql_multiclass(df['feature_single'], classes)
    else:
        lookup_df_to_sql(df, split)


def save_model_and_extras(ebm, model_name, split, logging):
    # extract lookup table from EBM
    lookup_df = extractLookupTable(ebm)

    # Write printed output to file
    with open('{model_name}/model/ebm_in_sql.sql'.format(model_name=model_name), 'w') as f:
        with redirect_stdout(f):
            print(f"'{model_name.split('/')[1]}' AS model_name \n, ", end='')
            ebm_to_sql(lookup_df, ebm.classes_, split)
    print('SQL version of EBM saved')
    logging.info('SQL version of EBM saved')
