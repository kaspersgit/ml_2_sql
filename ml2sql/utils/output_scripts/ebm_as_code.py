import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import logging

logger = logging.getLogger(__name__)


def ReduceSingleFeature(df_):
    # For safety
    df = df_.copy()

    # drop any duplicates (which should not exist)
    df_feat_types = (
        df[["feature", "feat_type"]].drop_duplicates().reset_index(drop=True).copy()
    )

    # Set index
    df = df.set_index(["feature"])

    # Explode per column
    lookup_df = (
        df[["feat_bound", "score", "score_lower_bound", "score_upper_bound"]]
        .apply(lambda x: x.apply(pd.Series).stack())
        .reset_index()
        .drop("level_1", axis=1)
    )

    # add feature type back in
    lookup_df = lookup_df.merge(df_feat_types, on="feature", how="left")

    # sort_values but None will be in the bottom
    lookup_df.sort_values(["feature", "feat_bound"], inplace=True, ascending=False)

    # So reverse the ordering
    lookup_df = lookup_df.iloc[::-1]

    lookup_df["nr_features"] = 1

    # being able for score arrays to be grouped
    # for categorical features do not group by score hence we include value in score_hash
    lookup_df["pre_hash_column"] = np.where(
        lookup_df["feat_type"] == "categorical",
        lookup_df["score"].astype(str) + lookup_df["feat_bound"].astype(str),
        lookup_df["score"].astype(str),
    )
    lookup_df["score_hash"] = pd.util.hash_pandas_object(
        lookup_df["pre_hash_column"], index=False
    )
    hash_lookup_df = (
        lookup_df[["score_hash", "score"]]
        .drop_duplicates("score_hash")
        .reset_index(drop=True)
        .copy()
    )

    # to group by score but keep them in order (not grouping similar scores for different feature values)
    adj_check = (lookup_df.score_hash != lookup_df.score_hash.shift()).cumsum()

    lookup_df["adj_score"] = adj_check
    lookup_df_simple = (
        lookup_df.groupby(["feature", "feat_type", "score_hash", "adj_score"])
        .agg(
            {
                "feat_bound": "last",
                "score_lower_bound": "first",
                "score_upper_bound": "last",
            },
            sort=False,
        )
        .reset_index(drop=False)
    )

    lookup_df_grouped = lookup_df_simple.merge(
        hash_lookup_df, on="score_hash", how="left"
    )
    # ordering sets None in last place
    lookup_df_grouped.sort_values(
        ["feature", "feat_bound"], inplace=True, ascending=False
    )
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
        if (row["score"].shape[0] > row["score"].shape[1]) & (
            row["feat_type"][0] == "numeric"
        ):
            # loop over column needed to be transposed
            for c in ["feature", "feat_bound", "score", "feat_type"]:
                if isinstance(df.loc[index, c], list):
                    df.at[index, c] = [df.at[index, c][1], df.at[index, c][0]]
                elif isinstance(df.loc[index, c], np.ndarray):
                    df.at[index, c] = np.transpose(df.at[index, c])

    # Reduce

    # Split list valued columns into two columns
    df[["feat_1", "feat_2"]] = df["feature"].apply(pd.Series)
    df[["feat_bound_1", "feat_bound_2"]] = df["feat_bound"].apply(pd.Series)
    df[["feat_type_1", "feat_type_2"]] = df["feat_type"].apply(pd.Series)

    # Drop unused column
    df = df.drop(["feat_bound"], axis=1)

    # Set index
    df = df.set_index(["feature"])

    # first explode on score and feat_bound_1 (same dimensions)
    df_explode_1 = df.explode(["score", "feat_bound_1"])
    lookup_df = df_explode_1.explode(["score", "feat_bound_2"])

    # sort_values but None will be in the bottom
    lookup_df = lookup_df.sort_values(
        ["feat_1", "feat_2", "feat_bound_1", "feat_bound_2"], ascending=False
    )

    # So reverse the ordering
    lookup_df = lookup_df.iloc[::-1]

    lookup_df["nr_features"] = 2

    # being able for score arrays to be grouped
    # for categorical features do not group by score hence we include value in score_hash
    mask_category = ["categorical" in x for x in lookup_df["feat_type_2"]]
    mask_inf_bound = [np.inf == x for x in lookup_df["feat_bound_2"]]
    mask = [a or b for a, b in zip(mask_category, mask_inf_bound)]

    lookup_df["pre_hash_column"] = np.where(
        mask,
        lookup_df["score"].astype(str)
        + lookup_df["feat_bound_1"].astype(str)
        + lookup_df["feat_bound_2"].astype(str),
        lookup_df["score"].astype(str) + lookup_df["feat_bound_1"].astype(str),
    )
    lookup_df["score_hash"] = pd.util.hash_pandas_object(
        lookup_df["pre_hash_column"], index=False
    )
    hash_lookup_df = (
        lookup_df[["score_hash", "score"]]
        .drop_duplicates("score_hash")
        .reset_index(drop=True)
        .copy()
    )

    # to group by score but keep them in order (not grouping similar scores for different feature values)
    adj_check = (lookup_df.score_hash != lookup_df.score_hash.shift()).cumsum()

    lookup_df["adj_score"] = adj_check
    lookup_df_simple = (
        lookup_df.groupby(
            [
                "feat_1",
                "feat_2",
                "feat_type_1",
                "feat_type_2",
                "score_hash",
                "adj_score",
            ]
        )
        .agg(
            {
                "feat_bound_1": "first",
                "feat_bound_2": "last",  # last to have numeric bounds correctly (with the existence of inf)
                "score_lower_bound": "first",
                "score_upper_bound": "last",
            },
            sort=False,
        )
        .reset_index(drop=False)
    )

    lookup_df_grouped = lookup_df_simple.merge(
        hash_lookup_df, on="score_hash", how="left"
    )

    # ordering sets None in last place
    lookup_df_grouped.sort_values(
        ["feat_1", "feat_2", "feat_bound_1", "feat_bound_2"],
        inplace=True,
        ascending=False,
    )

    # So reverse the ordering
    lookup_df_grouped = lookup_df_grouped.iloc[::-1].reset_index(drop=True)

    # Group second feature to list again for easier SQL writing
    lookup_df_group_feat2 = (
        lookup_df_grouped.groupby(
            ["feat_1", "feat_2", "feat_type_1", "feat_type_2", "feat_bound_1"]
        )
        .agg({"feat_bound_2": list, "score": list})
        .reset_index(drop=False)
    )

    return lookup_df_group_feat2


def extractLookupTable(ebm, post_params):
    """Provides reformatted structure of EBM

    Args:
        ebm: A trained EBM

    Returns:
        A dict of dataframes including information to create SQL code
    """

    # Add per feature graph
    lookup_dicts = []

    for feature_group_index, feature_indexes in enumerate(ebm.term_features_):
        # below includes treatment for missing values
        model_graph = ebm.term_scores_[feature_group_index]

        # NOTE: This uses stddev. for bounds, consider issue warnings.
        errors = ebm.standard_deviations_[feature_group_index]

        if len(feature_indexes) == 1:
            # hack. remove the last index which is for unknown values
            model_graph = model_graph[0:-1]
            errors = errors[0:-1]

            feature_bins = ebm.bins_[feature_group_index][0]

            scores = list(model_graph)
            upper_bounds = list(model_graph + errors)
            lower_bounds = list(model_graph - errors)

            if isinstance(feature_bins, dict):
                # categorical
                feat_bound = list(feature_bins.keys())
                feat_bound.insert(0, None)
                feat_type = "categorical"

            else:
                feat_bound = np.append(feature_bins, np.inf)
                feat_bound = np.append(None, feat_bound)
                feat_type = "numeric"

            score = scores
            score_lower_bound = lower_bounds
            score_upper_bound = upper_bounds

            lookup_dict = {
                "nr_features": 1,
                "feature": ebm.feature_names[feature_indexes[0]],
                "feat_bound": feat_bound,
                "score": list(
                    np.round(score, post_params["sql_decimals"])
                ),  ## To avoid data type overflow
                "score_lower_bound": score_lower_bound,
                "score_upper_bound": score_upper_bound,
                "feat_type": feat_type,
            }

            lookup_dicts.append(lookup_dict)

        elif len(feature_indexes) == 2:
            # hack. remove the last index which is for unknown values and just zeros
            model_graph = model_graph[0:-1, 0:-1]

            score = model_graph

            feat_bound = [[]] * len(feature_indexes)
            feat_type = [[]] * len(feature_indexes)

            # Make loop over labels to update the score and features bounds
            # Check for non binary variable
            for b in range(len(feature_indexes)):
                bin_levels = ebm.bins_[feature_indexes[b]]
                feature_bins = bin_levels[
                    min(len(feature_indexes), len(bin_levels)) - 1
                ]

                if isinstance(feature_bins, dict):
                    feat_bound[b] = list(feature_bins.keys())
                    feat_bound[b].insert(0, None)
                    feat_type[b] = "categorical"
                else:
                    feat_bound[b] = np.append(feature_bins, np.inf)
                    feat_bound[b] = np.append(None, feat_bound[b])
                    feat_type[b] = "numeric"

            lookup_dict = {
                "nr_features": 2,
                "feature": [
                    ebm.feature_names[feature_indexes[0]],
                    ebm.feature_names[feature_indexes[1]],
                ],
                "feat_bound": feat_bound,
                "score": np.round(score, post_params["sql_decimals"]),
                "feat_type": feat_type,
                # not implemented for interaction terms
                # , 'score_lower_bound': score_lower_bound
                # , 'score_upper_bound': score_upper_bound
            }

            lookup_dicts.append(lookup_dict)

        else:  # pragma: no cover
            raise Exception("Interactions greater than 2 not supported.")

    df_of_lists = pd.DataFrame(lookup_dicts)

    # we treat single and double features differently
    df_of_lists_single = df_of_lists[df_of_lists["nr_features"] == 1].reset_index(
        drop=True
    )
    df_of_lists_double = df_of_lists[df_of_lists["nr_features"] == 2].reset_index(
        drop=True
    )

    # Reduce Single featuers lookup
    lookup_df_single = ReduceSingleFeature(df_of_lists_single)

    # Check if interactions exist
    if len(df_of_lists_double) > 0:
        # Restructure and reduce double feature df
        lookup_df_double = RestructureReduceInteractions(df_of_lists_double)
    else:
        lookup_df_double = None

    return {
        "intercept": ebm.intercept_,
        "feature_single": lookup_df_single,
        "feature_double": lookup_df_double,
    }


def lookup_df_to_sql(model_name, df_dict, model_type, split):
    # Create list for all feature score names
    feature_list = []

    # Start with intercept term
    intercept = df_dict["intercept"][0]
    if not split:
        print("SELECT \n")
        print(f"'{model_name}' AS model_name \n, ")
        print(f"{intercept} AS intercept")
    elif split:
        # Creating CTE to create table aliases
        print("WITH feature_scores AS (\nSELECT")
        print(f"'{model_name}' AS model_name \n, ")
        print(f"{intercept} AS intercept")

    feature_list.append("intercept")

    # for single feature
    single_features = single_feature_handling(df_dict["feature_single"])
    feature_list = feature_list + single_features

    if len(df_dict["feature_double"]) > 0:
        double_features = double_feature_sql_handling(df_dict["feature_double"])
        feature_list = feature_list + double_features

    if not split:
        # Sum up all separate scores
        print(", ", end="")
        print(*feature_list, sep=" + ")

        if model_type == "regression":
            print(" AS prediction")
        else:
            print(" AS score")
            # Applying softmax
            print(", EXP(score)/(EXP(score) + 1) AS probability")

        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")

    elif split:
        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")

        # Close CTE and create next SELECT statement
        print("), add_sum_scores AS (")
        print("SELECT *")

        # Sum up all separate scores
        print(", ", end="")
        print(*feature_list, sep=" + ")

        if model_type == "regression":
            print(" AS prediction")
        else:
            print(" AS score")

        print("FROM feature_scores")

        # Close CTE and make final Select statement
        print(")")
        print("SELECT *")

        if model_type != "regression":
            # Applying softmax
            print(", EXP(score)/(EXP(score) + 1) AS probability")

        print("FROM add_sum_scores")


def single_feature_handling(df):
    feature_nr = 0
    feature_list = []
    for f in df["feature"].unique():
        feature_df = df[df["feature"] == f].reset_index(drop=True)

        # Each feature score as seperate column
        print(",\nCASE")

        single_feature_2_sql(feature_df, f)
        feature_nr += 1
        feature_list.append(f"{f}_score")

        # feature score alias
        print(f"AS {f}_score")

    return feature_list


def single_feature_2_sql(df, feature):
    for index, row in df.iterrows():
        # Check if bound is for missing values (=None)
        if row["feat_bound"] is None:
            print(
                " WHEN {feature} IS NULL THEN {score}".format(
                    feature=feature,
                    score=row["score"],
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check if string/category
        elif row["feat_type"] == "categorical":
            # check for manual imputed null values
            print(
                " WHEN {feature} = '{lb}' THEN {score}".format(
                    feature=feature, lb=row["feat_bound"], score=row["score"]
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check for last numeric bound
        elif row["feat_bound"] == np.inf:
            print(
                " WHEN {feature} >= {lb} THEN {score}".format(
                    feature=feature,
                    lb=df.iloc[index - 1, :]["feat_bound"],
                    score=row["score"],
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # otherwise it should be a float
        elif isinstance(row["feat_bound"], float):
            # First bound (skipping the missing values)
            if index <= 1:
                print(
                    " WHEN {feature} <= {ub} THEN {score}".format(
                        feature=feature, ub=row["feat_bound"], score=row["score"]
                    )
                )

            else:
                print(
                    " WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}".format(
                        feature=feature,
                        lb=df.iloc[index - 1, :]["feat_bound"],
                        ub=row["feat_bound"],
                        score=row["score"],
                    )
                )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        else:
            raise "not sure what to do"

    print("END")


def double_feature_sql_handling(df):
    feature_nr = 0
    feature_list = []
    df["double_feature"] = df["feat_1"] + "_x_" + df["feat_2"]
    for f in df["double_feature"].unique():
        feature_df = df[df["double_feature"] == f].reset_index(drop=True)

        # Each double feature score as case statement
        print(",\nCASE")

        double_feature_2_sql(feature_df, f)
        feature_nr += 1
        feature_list.append(f"{f}_score")

        # feature score alias
        print(f"AS {f}_score")

    return feature_list


def double_feature_2_sql(df, double_feature):
    for index, row in df.iterrows():
        # Check if bound is for missing values (=None)
        if row["feat_bound_1"] is None:
            print(
                " WHEN {feature} IS NULL THEN \n      CASE".format(
                    feature=row["feat_1"]
                )
            )

        # check if string/category
        elif row["feat_type_1"] == "categorical":
            print(
                " WHEN {feature} = '{lb}' THEN \n     CASE".format(
                    feature=row["feat_1"], lb=row["feat_bound_1"]
                )
            )

        # check for last numeric bound
        elif row["feat_bound_1"] == np.inf:
            print(
                " WHEN {feature} >= {lb} THEN \n      CASE".format(
                    feature=row["feat_1"], lb=df.loc[index - 1, "feat_bound_1"]
                )
            )

        # otherwise it should be a float
        elif isinstance(row["feat_bound_1"], float):
            # First bound (skipping the missing values)
            if index <= 1:
                print(
                    " WHEN {feature} <= {ub} THEN \n      CASE".format(
                        feature=row["feat_1"], ub=row["feat_bound_1"]
                    )
                )

            else:
                print(
                    " WHEN {feature} > {lb} AND {feature} <= {ub} THEN \n     CASE".format(
                        feature=row["feat_1"],
                        lb=df.loc[index - 1, "feat_bound_1"],
                        ub=row["feat_bound_1"],
                    )
                )

        else:
            raise "not sure what to do"

        # Nr of bound values
        nr_bounds = len(row["feat_bound_2"])

        # Looping over the bound values for the second feature
        for sf_index in range(nr_bounds):
            # Check if bound is for missing values (=None)
            if row["feat_bound_2"][sf_index] is None:
                print(
                    "         WHEN {feature} IS NULL THEN {score}".format(
                        feature=row["feat_2"],
                        score=row["score"][sf_index],
                    )
                )

            # check if string/category
            elif row["feat_type_2"] == "categorical":
                # check for manual imputed null values
                print(
                    "         WHEN {feature} = '{lb}' THEN {score}".format(
                        feature=row["feat_2"],
                        lb=row["feat_bound_2"][sf_index - 1],
                        score=row["score"][sf_index],
                    )
                )

            # check for last numeric bound
            elif row["feat_bound_2"][sf_index] == np.inf:
                print(
                    "         WHEN {feature} >= {lb} THEN {score}".format(
                        feature=row["feat_2"],
                        lb=row["feat_bound_2"][sf_index - 1],
                        score=row["score"][sf_index],
                    )
                )

            # otherwise it should be a float
            elif isinstance(row["feat_bound_2"][sf_index], float):
                # First bound (skipping the missing values)
                if sf_index <= 1:
                    print(
                        "         WHEN {feature} <= {ub} THEN {score}".format(
                            feature=row["feat_2"],
                            ub=row["feat_bound_2"][sf_index],
                            score=row["score"][sf_index],
                        )
                    )

                else:
                    print(
                        "         WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}".format(
                            feature=row["feat_2"],
                            lb=row["feat_bound_2"][sf_index - 1],
                            ub=row["feat_bound_2"][sf_index],
                            score=row["score"][sf_index],
                        )
                    )

            else:
                raise "not sure what to do"

            # Add ELSE 0.0 as last entry
            if sf_index + 1 == nr_bounds:
                print("         ELSE 0.0")

        print("     END")
    # Catch anything else
    print(" ELSE 0.0")
    print("END")


def lookup_df_to_sql_multiclass(model_name, df, classes, split):
    df_single = df["feature_single"]
    intercepts = df["intercept"]

    if split:
        # Add starting cte
        print("WITH feature_scores AS (")

    print("SELECT")
    print(f"'{model_name}' AS model_name")
    # Add intercepts
    for i, c in enumerate(classes):
        print(f", {intercepts[i]} AS intercept_{c}")

    class_nr = 0
    feature_list = {}
    for c in classes:
        feature_nr = 0
        feature_list[c] = []

        ## Add intercept to feature list
        feature_list[c].append(f"intercept_{c}")

        for f in df_single["feature"].unique():
            feature_df = df_single[df_single["feature"] == f].reset_index(drop=True)

            single_feature_2_sql_multiclass(feature_df, f, class_nr)

            # Feature score as alias
            print(f"AS {f}_score_{c}")
            feature_list[c].append(f"{f}_score_{c}")

            feature_nr += 1
        class_nr += 1

    if split:
        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")
        # Close CTE and create next SELECT statement
        print("), add_sum_scores AS (")
        print("SELECT *")

    if not split:
        for c in classes:
            # Sum up all separate scores
            print(", ", end="")
            print(*feature_list[c], sep=" + ")
            print(f" AS score_{c}")

    elif split:
        for c in classes:
            # Create CTE with sums of all features scores per class
            print(", ", end="")
            print(*feature_list[c], sep=" + ")
            print(f" AS score_{c}")
        # From CTE
        print("FROM feature_scores")

    # Summing feature scores to total score per class
    if not split:
        for c in classes:
            if c == classes[0]:
                print(f", EXP(score_{c})", end="")
            else:
                print(f" + EXP(score_{c})", end="")
        print(" AS total_score")

    elif split:
        # Close CTE and create next SELECT statement
        print("), add_sum_all_scores AS (")
        print("SELECT *")

        for c in classes:
            if c == classes[0]:
                print(f", EXP(score_{c})", end="")
            else:
                print(f" + EXP(score_{c})", end="")
        print(" AS total_score")

        # Check class with highest score

        # Get max score
        print(f", GREATEST({', '.join([f'score_{i}' for i in classes])}) AS max_score")

        # Close CTE and create final SELECT statement
        print("FROM add_sum_scores")
        print(")")
        print("SELECT *")

    # Applying softmax
    if not split:
        for c in classes:
            print(f", EXP(score_{c}) / (total_score) AS probability_{c}", end="\n")

        print(
            f", GREATEST({', '.join([f'probability_{i}' for i in classes])}) AS max_probability"
        )

        print(", CASE")
        for c in classes:
            print(f"\t WHEN probability_{c} = max_probability THEN '{c}'")
        print("END AS prediction")

        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")
    elif split:
        for c in classes:
            print(f", EXP(score_{c}) / (total_score) AS probability_{c}", end="\n")

        print(", EXP(max_score) / (total_score) AS max_probability")

        print(", CASE")
        for c in classes:
            print(f"\t WHEN score_{c} = max_score THEN '{c}'")
        print("END AS prediction")

        print("FROM add_sum_all_scores")


def single_feature_2_sql_multiclass(df, feature, class_nr):
    # Start new case statement
    print(",\nCASE")

    for index, row in df.iterrows():
        # Check if bound is for missing values (=None)
        if row["feat_bound"] is None:
            print(
                " WHEN {feature} IS NULL THEN {score}".format(
                    feature=feature,
                    score=row["score"][class_nr],
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check if string/category
        elif row["feat_type"] == "categorical":
            # check for manual imputed null values
            print(
                " WHEN {feature} = '{lb}' THEN {score}".format(
                    feature=feature, lb=row["feat_bound"], score=row["score"][class_nr]
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # check for last numeric bound
        elif row["feat_bound"] == np.inf:
            print(
                " WHEN {feature} >= {lb} THEN {score}".format(
                    feature=feature,
                    lb=df.iloc[index - 1, :]["feat_bound"],
                    score=row["score"][class_nr],
                )
            )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        # otherwise it should be a float
        elif isinstance(row["feat_bound"], float):
            # First bound (skipping the missing values)
            if index <= 1:
                print(
                    " WHEN {feature} <= {ub} THEN {score}".format(
                        feature=feature,
                        ub=row["feat_bound"],
                        score=row["score"][class_nr],
                    )
                )

            else:
                print(
                    " WHEN {feature} > {lb} AND {feature} <= {ub} THEN {score}".format(
                        feature=feature,
                        lb=df.iloc[index - 1, :]["feat_bound"],
                        ub=row["feat_bound"],
                        score=row["score"][class_nr],
                    )
                )
            # Add ELSE 0.0 as last entry
            if index == df.index[-1]:
                print(" ELSE 0.0")

        else:
            raise "not sure what to do"

    print("END")


def ebm_to_sql(model_name, df, classes, split=True):
    if len(classes) > 2:
        lookup_df_to_sql_multiclass(model_name, df, classes, split)
    elif len(classes) == 1:
        model_type = "regression"
        lookup_df_to_sql(model_name, df, model_type, split)
    else:
        model_type = "binary"
        lookup_df_to_sql(model_name, df, model_type, split)


def save_model_and_extras(ebm, model_name, post_params):
    # extract lookup table from EBM
    lookup_df = extractLookupTable(ebm, post_params)

    # In case of regression
    if not hasattr(ebm, "classes_"):
        ebm.classes_ = [0]
        lookup_df["intercept"] = [lookup_df["intercept"]]

    # Write printed output to file
    with open(f"{model_name}/model/ebm_in_sql.sql", "w") as f:
        with redirect_stdout(f):
            model_name = model_name.split("/")[-1]
            ebm_to_sql(model_name, lookup_df, ebm.classes_, post_params["sql_split"])
    logger.info("SQL version of EBM saved")
