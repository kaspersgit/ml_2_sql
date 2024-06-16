from interpret.glassbox import LinearRegression, LogisticRegression
import logging

logger = logging.getLogger(__name__)


def trainModel(X_train, y_train, params, model_type):
    # if 'feature_names' not in params.keys():
    #     params['feature_names'] = X_train.columns
    if model_type == "regression":
        clf = LinearRegression(**params)
        clf_name = "Linear regression"
    elif model_type == "classification":
        clf = LogisticRegression(**params)
        # Hard code classes_
        clf.classes_ = list(set(y_train))
        clf_name = "Logistic regression"
    else:
        logging.warning("Only regression or classification available")

    clf.fit(X_train, y_train)
    logger.info(f"Model non default params:\n {clf.kwargs}")
    logger.info(f"Trained {clf_name.lower()}")

    return clf


def featureExplanationSave(clf, given_name, file_type):
    clf_global = clf.explain_global()

    # Save overall feature importance graph
    plotly_fig = clf_global.visualize()

    if file_type == "png":
        plotly_fig.write_image(
            "{given_name}/1_overall_feature_importance.png".format(
                given_name=given_name
            )
        )
    elif file_type == "html":
        plotly_fig.write_html(
            "{given_name}/1_overall_feature_importance.html".format(
                given_name=given_name
            )
        )

    # Save feature specific explanation graphs
    for index, value in enumerate(clf.feature_names_in_):
        plotly_fig = clf_global.visualize(index)

        # reformatting feature name
        feature_name = clf.feature_names_in_[index]
        chars = "\\`./ "
        for c in chars:
            if c in feature_name:
                feature_name = feature_name.replace(c, "_")

        if file_type == "png":
            plotly_fig.write_image(f"{given_name}/explain_{feature_name}.png")
        elif file_type == "html":
            # or as html file
            plotly_fig.write_html(f"{given_name}/explain_{feature_name}.html")

    logger.info(
        "Explanation plots of {n_features} features saved".format(n_features=index + 1)
    )


def postModelPlots(clf, given_name, file_type):
    featureExplanationSave(clf, given_name, file_type)
