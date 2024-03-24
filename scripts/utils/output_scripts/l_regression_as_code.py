import numpy as np
from contextlib import redirect_stdout
import logging

logger = logging.getLogger(__name__)


def extract_parameters(model):
    """
    Extracts model_type, features, coefficients, and intercept from a trained logistic regression model.

    Parameters:
    - trained_model: The trained logistic regression model object.

    Returns:
    - model_type: String, either regression of classification
    - features: List of feature names.
    - coefficients: List of coefficients corresponding to each feature.
    - intercept: Intercept of the logistic regression model.
    """
    try:
        # Extract model type
        if model.__class__.__name__ == "LinearRegression":
            model_type = "regression"
            pclasses = None
        elif len(model.classes_) > 2:
            model_type = "multiclass"
            pclasses = model.classes_
        elif len(model.classes_) == 2:
            model_type = "binary"
            pclasses = model.classes_

        # Extract features
        features = model.feature_names_in_

        if model_type == "binary":
            coefficients = model.sk_model_.coef_[0]
        else:
            coefficients = model.sk_model_.coef_

        # Extract intercept
        if model_type == "binary":
            intercept = model.sk_model_.intercept_[0]
        else:
            intercept = model.sk_model_.intercept_

        return model_type, pclasses, features, coefficients, intercept

    except Exception as e:
        # Handle exceptions based on your specific model or library
        print(f"Error extracting parameters: {str(e)}")
        return None, None, None, None, None


def format_sql(
    model_name, model_type, pclasses, features, coefficients, intercept, post_params
):
    # round coefficients if needed
    coefficients = list(np.round(coefficients, post_params["sql_decimals"]))

    # List of column aliases of the scores of the different features
    if model_type == "multiclass":
        score_cols = {}
        for i, c in enumerate(pclasses):
            score_cols[c] = " + ".join([f"{feature}_score_{c}" for feature in features])
    else:
        score_cols = " + ".join([f"{feature}_score" for feature in features])

    # Include the individual score per feature in the SELECT clause
    if model_type == "multiclass":
        feature_scores = {}
        for i, c in enumerate(pclasses):
            feature_scores[c] = "\t, ".join(
                [
                    f"({coef} * {feature}) AS {feature}_score_{c}\n"
                    for coef, feature in zip(coefficients[i], features)
                ]
            )
    else:
        feature_scores = "\t, ".join(
            [
                f"({coef} * {feature}) AS {feature}_score\n"
                for coef, feature in zip(coefficients, features)
            ]
        )

    # Create the SQL query with the logistic regression formula and feature scores
    if not post_params["sql_split"]:
        print("SELECT")
        print(f"\t'{model_name}' AS model_name")

        if model_type == "regression":
            print(f"\t, {intercept} AS intercept")
            print(f"\t, {feature_scores}", end="")
            print(f"\t, {score_cols} + intercept AS prediction")
        elif model_type == "binary":
            print(f"\t, {intercept} AS intercept")
            print(f"\t, {feature_scores}", end="")
            print(f"\t, {score_cols} + {intercept} AS score")
            print("\t, 1 / (1 + EXP(-(score))) AS probability")
        elif model_type == "multiclass":
            for i, c in zip(intercept, pclasses):
                print(f"\t, {i} AS intercept_{c}")

            for c in pclasses:
                print(f"\t, {feature_scores[c]}", end="")

            for c in pclasses:
                print(f"\t, {score_cols[c]} + intercept_{c} AS score_{c}")

            print("\t, (EXP(", end="")
            class_score_list = [f"score_{c}" for c in pclasses]
            print(*class_score_list, sep=") + EXP(", end=")) AS total_score\n")

            for c in pclasses:
                print(f"\t, EXP(score_{c}) / total_score AS probability_{c}")

        print("FROM <source_table>;  -- TODO replace with correct table")

    elif post_params["sql_split"]:
        # Creating CTE to create table aliases
        print("WITH feature_scores AS (\nSELECT")
        print(f"\t'{model_name}' AS model_name")

        if model_type == "regression":
            print(f"\t, {intercept} AS intercept")
            print(f"\t, {feature_scores}", end="")
        elif model_type == "binary":
            print(f"\t, {intercept} AS intercept")
            print(f"\t, {feature_scores}", end="")
        elif model_type == "multiclass":
            for i, c in zip(intercept, pclasses):
                print(f"\t, {i} AS intercept_{c}")

            for c in pclasses:
                print(f"\t, {feature_scores[c]}", end="")

        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")

        # Close CTE and create next SELECT statement
        print("), add_sum_scores AS (")
        print("SELECT *")

        if model_type == "regression":
            print(f"\t, {score_cols} + intercept AS prediction")
        elif model_type == "binary":
            print(f"\t, {score_cols} + intercept AS score")
        elif model_type == "multiclass":
            # scores per class
            for c in pclasses:
                print(f"\t, {score_cols[c]} + intercept_{c} AS score_{c}")
            class_score_list = [f"score_{c}" for c in pclasses]
            print("\t, (EXP(", end="")
            print(*class_score_list, sep=") + EXP(", end=")) AS total_score\n")

        print("FROM feature_scores")

        # Close CTE and make final Select statement
        print(")")
        print("SELECT *")

        if model_type == "binary":
            # TODO adjust accordingly for multiclass
            # Applying softmax
            print(", 1 / (1 + EXP(-score)) AS probability")
        elif model_type == "multiclass":
            for c in pclasses:
                print(f"\t, EXP(score_{c})/total_score AS probability_{c}")

        print("FROM add_sum_scores")
    return


def save_model_and_extras(clf, model_name, post_params):
    model_type, pclasses, features, coefficients, intercept = extract_parameters(clf)
    # Write printed output to file
    with open(
        "{model_name}/model/lregression_in_sql.sql".format(model_name=model_name), "w"
    ) as f:
        with redirect_stdout(f):
            model_name = model_name.split("/")[-1]
            format_sql(
                model_name,
                model_type,
                pclasses,
                features,
                coefficients,
                intercept,
                post_params,
            )
    logger.info("SQL version of logistic/linear regression saved")
