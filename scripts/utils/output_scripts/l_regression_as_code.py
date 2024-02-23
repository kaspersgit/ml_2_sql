from contextlib import redirect_stdout


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
        model_type = (
            "regression"
            if model.__class__.__name__ == "LinearRegression"
            else "classification"
        )

        # Extract features
        features = model.feature_names_in_

        # Extract coefficients
        if model_type == "regression":
            coefficients = model.sk_model_.coef_
        else:
            coefficients = model.sk_model_.coef_[0]

        # Extract intercept
        if model_type == "regression":
            intercept = model.sk_model_.intercept_
        else:
            intercept = model.sk_model_.intercept_[0]

        return model_type, features, coefficients, intercept

    except Exception as e:
        # Handle exceptions based on your specific model or library
        print(f"Error extracting parameters: {str(e)}")
        return None, None, None, None


def format_sql(model_name, model_type, features, coefficients, intercept, split):
    # Calculate the linear combination of features and coefficients
    linear_combination = " + ".join(
        [f"({coef} * {feature})" for coef, feature in zip(coefficients, features)]
    )

    # Include the individual score per feature in the SELECT clause
    feature_scores = "\t, ".join(
        [
            f"({coef} * {feature}) AS {feature}_score\n"
            for coef, feature in zip(coefficients, features)
        ]
    )

    # Create the SQL query with the logistic regression formula and feature scores
    if not split:
        print("SELECT")
        print(f"\t'{model_name}' AS model_name")
        print(f"\t, {intercept} AS intercept")
        print(f"\t, {feature_scores}", end="")

        if model_type == "regression":
            print(f"\t, {linear_combination} + {intercept} AS prediction")
        else:
            print(
                f"\t, 1 / (1 + EXP(-({linear_combination} + {intercept}))) AS probability"
            )
        print("FROM <source_table>;  -- TODO replace with correct table")

    elif split:
        # Creating CTE to create table aliases
        print("WITH feature_scores AS (\nSELECT")
        print(f"\t'{model_name}' AS model_name")
        print(f"\t, {intercept} AS intercept")
        print(f"\t, {feature_scores}")
        # Add placeholder for source table
        print("FROM <source_table> -- TODO replace with correct table")

        # Close CTE and create next SELECT statement
        print("), add_sum_scores AS (")
        print("SELECT *")

        # Sum up all separate scores
        print(", ", end="")
        feature_scores_list = [f"{feature}_score" for feature in features]
        scores_list = feature_scores_list + [intercept]
        print(*scores_list, sep=" + ", end="")

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
            print(", 1 / (1 + EXP(-score)) AS probability")

        print("FROM add_sum_scores")
    return


def save_model_and_extras(clf, model_name, sql_split, logging):
    model_type, features, coefficients, intercept = extract_parameters(clf)
    # Write printed output to file
    with open(
        "{model_name}/model/lregression_in_sql.sql".format(model_name=model_name), "w"
    ) as f:
        with redirect_stdout(f):
            model_name = model_name.split("/")[-1]
            format_sql(
                model_name, model_type, features, coefficients, intercept, sql_split
            )
    logging.info("SQL version of logistic/linear regression saved")
