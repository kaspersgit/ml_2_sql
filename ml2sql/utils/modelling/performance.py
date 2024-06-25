import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_squared_log_error,
)

# The actual algorithms (grey as we refer to them dynamically)
from ml2sql.utils.modelling.models import ebm  # noqa: F401
from ml2sql.utils.modelling.models import decision_tree  # noqa: F401
from ml2sql.utils.modelling.models import l_regression  # noqa: F401

import logging

logger = logging.getLogger(__name__)


def plotConfusionMatrixStatic(given_name, y_true, y_pred, data_type):
    """
    Plot and save a confusion matrix for given predictions.

    Parameters
    ----------
    given_name : str
        The name of the experiment or model.
    y_true : array-like of shape (n_samples,)
        True labels of the samples.
    y_pred : array-like of shape (n_samples,)
        Predicted labels of the samples.
    data_type : str
        The type of the data (train, val, test, etc.)

    Returns
    -------
    None

    Notes
    -----
    Saves the confusion matrix in {given_name}/performance/{data_type}_confusion_matrix.png

    """

    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)  # noqa: F841
    plt.xticks(rotation=90)
    plt.savefig(
        "{given_name}/performance/{data_type}_confusion_matrix.png".format(
            given_name=given_name, data_type=data_type
        ),
        bbox_inches="tight",
    )

    logger.info(f"Created and saved confusion matrix for {data_type} data")


# Mainly meant for binary classification
def plotConfusionMatrixSlider(given_name, y_true, y_prob, data_type):
    conf_matrices = []
    steps = []

    threshold_list = np.arange(0.0, 1.05, 0.05)

    for threshold in threshold_list:
        y_pred = [1 if x > threshold else 0 for x in y_prob]

        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        # confusion matrix
        cm = []
        for p in [0, 1]:
            new_row = []
            for a in [1, 0]:
                new_row.append(
                    len(df[(df["y_pred"] == p) & (df["y_true"] == a)]) / len(y_pred)
                )
            cm.append(new_row)

        # Add to list of matrices
        conf_matrices.append(cm)

        # Create sklearn summary report
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Create a list of lists for the matrix
        metrics_matrix = []

        # Iterate over the keys and values in the summary report dictionary
        for key, value in report.items():
            if key == "accuracy":
                metrics_matrix.append(["", "", "", "", ""])
                metrics_matrix.append(["Accuracy", "", "", value, ""])
            elif key.startswith(("macro", "weighted")):
                metrics_matrix.append(
                    [
                        key.capitalize(),
                        value["precision"],
                        value["recall"],
                        value["f1-score"],
                        value["support"],
                    ]
                )
            else:
                row = [
                    key,
                    value["precision"],
                    value["recall"],
                    value["f1-score"],
                    value["support"],
                ]
                metrics_matrix.append(row)

        # Formatting options
        precision_format = "{:.3f}"
        integer_format = "{:d}"
        empty_string_format = ""

        # Format the matrix elements
        formatted_matrix = []
        for row in metrics_matrix:
            formatted_row = [
                precision_format.format(cell)
                if isinstance(cell, float)
                else integer_format.format(cell)
                if isinstance(cell, int)
                else empty_string_format
                if cell == ""
                else cell
                for cell in row
            ]
            formatted_matrix.append(formatted_row)

        # transpose matrix
        metrics = list(zip(*formatted_matrix))

        # Add values to the different steps
        steps.append(
            dict(
                method="restyle",
                args=[
                    {
                        "z": [cm],  # in the initial fig update z and text
                        "text": [cm],
                        "cells.values": [metrics],
                    }
                ],
                label=round(threshold, 2),
            )
        )

        if threshold == 0.5:
            conf_matrix = cm
            table_metrics = metrics

    # Manually add labels
    labels = ["1", "0"]
    labels_r = labels.copy()
    labels_r.reverse()

    # Make subplots top table bottom heatmap
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "table"}, {"type": "heatmap"}]]
    )

    # Create heatmap
    heatmap = go.Heatmap(
        x=labels,
        y=labels_r,
        z=conf_matrix,
        text=conf_matrix,
        texttemplate="%{text:.3f}",
        hovertemplate="<br>".join(
            [
                "Predicted: %{y}",
                "Actual: %{x}",
                "Share of total cases: %{z}",
                "<extra></extra>",
            ]
        ),
    )

    # Create table
    metrics_table = go.Table(
        header=dict(
            values=["          ", "Precision", "Recall", "F1-Score", "Support"]
        ),
        cells=dict(values=table_metrics),
    )

    # Add table and heatmap to figure
    fig.add_trace(metrics_table, row=1, col=1)
    fig.add_trace(heatmap, row=1, col=2)

    fig.update_layout(
        width=1000,
        height=600,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        xaxis_type="category",
        yaxis_type="category",
        xaxis_side="top",
        title_text="Normalized confusion matrix",
        title_x=0.5,
    )

    sliders = [
        dict(
            active=10,
            currentvalue={"prefix": "Threshold: "},
            pad={"t": 50},  # sort it out later
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)

    fig.write_html(
        f"{given_name}/performance/{data_type}_confusion_matrix.html", auto_play=False
    )

    logger.info(f"Created and saved confusion matrix for {data_type} data")


def plotConfusionMatrix(given_name, y_true, y_prob, y_pred, file_type, data_type):
    # If html is wanted and binary classification
    # Make confusion matrix plot with slider
    if (file_type == "html") & (len(set(y_true)) == 2):
        plotConfusionMatrixSlider(given_name, y_true, y_prob, data_type)

    # Otherwise make 'simple' static confusion matrix plot
    else:
        plotConfusionMatrixStatic(given_name, y_true, y_pred, data_type)


def plotClassificationCurve(given_name, y_true, y_prob, curve_type, data_type):
    """
    Plots the ROC or Precision-Recall curve for a binary classification model, and saves the plot image.

    Parameters:
    -----------
    given_name : str
        Path where the plot image should be saved.
    y_true : array-like of shape (n_samples,)
        True binary labels for the samples.
    y_prob : array-like of shape (n_samples,)
        Estimated probabilities of the positive class for the samples.
    curve_type : str
        Type of curve to plot. Either 'ROC' for Receiver Operating Characteristic curve, or 'PR' for Precision-Recall curve.
    data_type : str
        Type of data being plotted. This string is included in the plot title.

    Returns:
    --------
    mean_auc : float
        Mean area under the curve (AUC) value for all folds or for the fitted model.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    curve_type = curve_type.lower()

    if curve_type == "roc":
        title = "ROC curve - {data_type} data".format(data_type=data_type)
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate (Recall)"
        diagCor1 = [0, 1]
        diagCor2 = [0, 1]

        curve_ = "roc"

    elif ("recall" in curve_type and "precision" in curve_type) or curve_type == "pr":
        title = "Precision-Recall curve - {data_type}".format(data_type=data_type)
        xlabel = "True Positive Rate (Recall)"
        ylabel = "Precision"

        # add the random line (= share of positives in sample)
        if isinstance(y_true, list):
            pos_share = sum([sum(el) for el in y_true]) / sum(
                [len(el) for el in y_true]
            )
        else:
            pos_share = sum(y_true) / len(y_true)

        diagCor1 = [0, 1]
        diagCor2 = [pos_share, pos_share]

        curve_ = "pr"

    # Create plot
    fig = go.Figure()

    # set axis range to 0 - 1
    fig.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=title,
        width=1000,
        height=800,
    )

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=diagCor1[0],
        x1=diagCor1[1],
        y0=diagCor2[0],
        y1=diagCor2[1],
        line=dict(
            color="black",
            width=2,
            dash="dot",
        ),
    )

    # Save the auc(s)
    auc_list = list()

    if isinstance(y_prob, list):
        for fold_id in range(len(y_prob)):
            if curve_ == "roc":
                xVar, yVar, thresholds = roc_curve(y_true[fold_id], y_prob[fold_id])
            elif curve_ == "pr":
                yVar, xVar, thresholds = precision_recall_curve(
                    y_true[fold_id], y_prob[fold_id]
                )

            # Calculate area under curve (AUC)
            auc_list.append(auc(xVar, yVar))
            fig.add_trace(
                go.Scatter(x=xVar, y=yVar, mode="lines", name=f"Fold {fold_id}")
            )
    else:
        if curve_ == "roc":
            xVar, yVar, thresholds = roc_curve(y_true, y_prob)
        elif curve_ == "pr":
            yVar, xVar, thresholds = precision_recall_curve(y_true, y_prob)

        # Calculate area under curve (AUC)
        auc_list.append(auc(xVar, yVar))
        fig.add_trace(go.Scatter(x=xVar, y=yVar, mode="lines", name="Fitted model"))

    # add (average) auc in image
    fig.add_annotation(
        x=0.5, y=0, text=f"Mean AUC: {np.mean(auc_list)}", showarrow=False, yshift=10
    )

    fig.write_image(f"{given_name}/performance/{data_type}_{curve_type}_plot.png")

    logger.info(f"Created and saved {curve_type} plot for {data_type} data")

    return np.mean(auc_list)


def plotCalibrationCurve(given_name, y_true, y_prob, data_type):
    """
    Plot the calibration curve for a set of true and predicted values.

    Parameters
    ----------
    given_name : str
        The name of the given data.
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_prob : array-like of shape (n_samples,)
        The predicted probabilities of target.
    data_type : str
        The type of the given data (train or test).

    Returns
    -------
    None

    Notes
    -----
    This function creates and saves a calibration plot that shows the relationship
    between the true target values and the predicted probabilities of target. The plot
    is saved in a subdirectory named `performance` inside the directory specified by
    `given_name`.

    The function uses the `calibration_curve` function from scikit-learn to calculate
    the fraction of positives and the mean predicted value for a range of predicted
    probabilities. The number of points in the plot is determined by the `n_bins`
    parameter of `calibration_curve`.

    The function also calculates the Brier score loss for each fold of cross-validation
    and adds the mean Brier score loss to the plot.
    """

    # Create plot
    fig = go.Figure()

    # set axis range to 0 - 1
    fig.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        xaxis_title="Predicted probability",
        yaxis_title="Share being positive",
        title=f"Calibration plot (reliability curve) - {data_type} data",
        width=1000,
        height=800,
    )

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        line=dict(
            color="black",
            width=2,
            dash="dot",
        ),
    )

    # Save the brier score loss
    bsl_list = list()

    if isinstance(y_prob, list):
        for fold_id in range(len(y_prob)):
            # summaries actuals and predicted probs to (bins) number of points
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true[fold_id], y_prob[fold_id], n_bins=10, strategy="quantile"
            )

            # Calculate area under curve (AUC)
            bsl_list.append(brier_score_loss(y_true[fold_id], y_prob[fold_id]))
            fig.add_trace(
                go.Scatter(
                    x=mean_predicted_value,
                    y=fraction_of_positives,
                    mode="markers+lines",
                    name=f"Fold {fold_id}",
                )
            )
    else:
        # summaries actuals and predicted probs to (bins) number of points
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )

        # Calculate area under curve (AUC)
        bsl_list.append(brier_score_loss(y_true, y_prob))
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode="markers+lines",
                name="Fitted model",
            )
        )

    # add (average) auc in image
    fig.add_annotation(
        x=0.5,
        y=0,
        text=f"Mean Brier Score Loss: {np.mean(bsl_list)}",
        showarrow=False,
        yshift=10,
    )

    fig.write_image(f"{given_name}/performance/{data_type}_calibration_plot.png")

    logger.info(f"Created and saved calibration plot for {data_type} data")

    return


# for multiclass classification WIP
def multiClassPlotCalibrationCurvePlotly(given_name, actuals, probs, title, bins=10):
    """
    Plot the calibration curve for a set of true and predicted values

    :param actuals: true target value
    :param probs: predicted probabilities per class as pandas dataframe
    :param bins: how many bins to divide data in for plotting
    :param savefig: boolean if plot should be saved
    :param saveFileName: str path to which to save the plot
    :return: calibration plot
    """
    calibration_df = pd.DataFrame(
        columns=["classification", "fraction_of_positives", "mean_predicted_value"]
    )

    # loop through different classes
    for cl in probs.columns:
        y_true = np.where(actuals == cl, 1, 0)
        y_prob = probs[cl]

        # summarise actuals and predicted probs to (bins) number of points
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=bins, strategy="quantile"
        )

        calibration_df = calibration_df.append(
            pd.DataFrame(
                {
                    "classification": cl,
                    "fraction_of_positives": fraction_of_positives,
                    "mean_predicted_value": mean_predicted_value,
                }
            ),
            ignore_index=True,
        )

    # Create scatter plot
    fig = px.scatter(
        calibration_df,
        x="mean_predicted_value",
        y="fraction_of_positives",
        title="Calibration plot (reliability curve)",
        color="classification",
        width=1200,
        height=800,
    )

    # Make trace be line plus dots
    for ld in range(len(fig.data)):
        fig.data[ld].update(mode="markers+lines")

    # set axis range to 0 - 1
    # fig.update_layout(xaxis_range=[-0.1,1.1], yaxis_range=[-0.1,1.1], xaxis_title='Predicted probability', yaxis_title='Fraction of positives')

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        line=dict(
            color="black",
            width=2,
            dash="dot",
        ),
    )

    fig.show(renderer="browser")


def plotProbabilityDistribution(given_name, y_true, y_prob, data_type):
    """
    Plot the probability distribution of true and predicted values.

    Parameters
    ----------
    given_name : str
        Given name to create and save plot.
    y_true : array-like of shape (n_samples,)
        True target value.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities.
    data_type : str
        Type of data, such as "train", "test", or "validation".

    Returns
    -------
    None
    """
    import plotly.figure_factory as ff

    df = pd.DataFrame({"actuals": y_true, "prob": y_prob})
    do = df[df["actuals"] == 1]["prob"]
    dont = df[df["actuals"] == 0]["prob"]

    # Catching any kind of exception
    try:
        # Create distplot with custom bin_size
        fig = ff.create_distplot(
            [do, dont], ["1", "0"], colors=["green", "red"], bin_size=0.01
        )
    except Exception as e:
        logger.info(f"Could not create distribution plot because of \n{e}")
        return

    # Update size of figure
    fig.update_layout(
        xaxis_title="Predicted probability",
        yaxis_title="Frequency",
        title=f"Distribution plot - {data_type} data",
        width=1000,
        height=800,
    )

    # fig.show(renderer='browser')

    fig.write_image(f"{given_name}/performance/{data_type}_distribution_plot.png")

    logger.info("Created and saved probability distribution plot")

    return


def plotDistribution(given_name, groups, values, data_type):
    """
    Create a plotly histogram figure displaying the distribution of values for each group.

    Parameters
    ----------
    given_name: str
    A string representing the name of the plot to be created.
    groups: array-like of shape (n_samples,)
    An array-like object containing the group assignments of each sample.
    values: array-like of shape (n_samples,)
    An array-like object containing the values to be plotted.
    data_type: str
    A string representing the type of data being plotted (e.g., training, validation, test).

    Returns
    -------
    None

    Raises
    ------
    Exception: If the histogram cannot be created.
    """
    df = pd.DataFrame({"groups": groups, "values": values})

    min_value = min(values)
    max_value = max(values)
    bin_size = (max_value - min_value) / 50

    colors = ["red", "green", "blue", "purple", "orange"]

    # Catching any kind of exception
    try:
        # Create distplot
        fig = go.Figure()
        for g in range(len(set(groups))):
            X = df[df["groups"] == g]["values"]
            fig.add_trace(
                go.Histogram(
                    x=X,
                    histnorm="probability density",
                    name=str(g),  # name used in legend and hover labels
                    xbins=dict(  # bins used for histogram
                        start=min_value, end=max_value, size=bin_size
                    ),
                    marker_color=colors[g],
                    opacity=0.75,
                )
            )
    except Exception as e:
        logger.info(f"Could not create distribution plot because of \n{e}")
        return

    fig.update_layout(
        xaxis_title="Predicted probability",
        yaxis_title="Probability density",
        title=f"Distribution plot - {data_type} data",
        width=1000,
        height=800,
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
    )

    fig.write_image(f"{given_name}/performance/{data_type}_distribution_plot.png")

    logger.info("Created and saved probability distribution plot")

    return


"""
Metrics and plot which are related to regression
"""


def plotYhatVsYSave(given_name, y_true, y_pred, data_type):
    """
    Plots a scatter plot of predicted values (y_pred) against true values (y_true)
    and saves the plot in a specified directory.

    Parameters:
    -----------
    given_name : str
        Name of the directory where the plot will be saved.
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    data_type : str
        Type of data, e.g. train, validation, or test.

    Returns:
    --------
    None
    """

    plot_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    fig = px.scatter(plot_df, "y_true", "y_pred")
    fig.write_image(
        "{given_name}/performance/{data_type}_scatter_yhat_vs_y.png".format(
            given_name=given_name, data_type=data_type
        )
    )

    logger.info(f"Scatter plot of yhat vs y saved for {data_type}")


def plotQuantileError(given_name, y_true, y_pred, data_type):
    # Add prediction error
    plot_df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "y_diff": y_pred - y_true}
    )

    # Make 20 quantiles
    plot_df["quantile"] = pd.qcut(plot_df["y_true"], 20, duplicates="drop")

    # Sort by quantile and make quantile colunn a string for plotting
    plot_df = plot_df.iloc[plot_df["quantile"].cat.codes.argsort()]
    plot_df["quantile"] = plot_df["quantile"].astype(str)

    fig = px.box(plot_df, "quantile", "y_diff")
    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Prediction - actual",
        title=f"Quantile error plot - {data_type} data",
        width=1000,
        height=800,
    )

    fig.write_image(
        "{given_name}/performance/{data_type}_quantile_error_plot.png".format(
            given_name=given_name, data_type=data_type
        )
    )

    logger.info(f"Quantile error plot saved for {data_type}")


def regressionMetricsTable(given_name, y_true, y_pred, X_all, data_type):
    y_true_pos = np.clip(y_true, 0, None)
    y_pred_pos = np.clip(y_pred, 0, None)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_all.shape[1] - 1)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true_pos, y_pred_pos)
    rmsle_value = msle**0.5

    header = ["Metric"]

    # Define the rows of the table
    rows = [
        ["Mean Absolute Error (MAE)"],
        ["Mean Squared Error (MSE)"],
        ["Root Mean Squared Error (RMSE)"],
        ["R-squared (R2)"],
        ["Adjusted R-squared (Adj R2)"],
        ["Mean Absolute Percentage Error (MAPE)"],
        ["Explained Variance Score (EVS)"],
        ["Max Error"],
        ["Median Absolute Error (MedAE)"],
        ["Mean Squared Log Error (MSLE)"],
        ["Root Mean Squared Log Error (RMSLE)"],
    ]

    # Create the table
    fig = go.Figure(
        data=[go.Table(header=dict(values=header), cells=dict(values=rows))]
    )

    # Add metrics data
    metric_values = [
        [mae],
        [mse],
        [rmse],
        [r2],
        [adj_r2],
        [mape],
        [evs],
        [me],
        [medae],
        [msle],
        [rmsle_value],
    ]

    metric_names = [row for row in rows]

    # Add metrics data as a new column
    cells_values = [metric_names, metric_values]
    column_names = ["Metric", "Value"]

    fig.add_trace(
        go.Table(
            header=dict(values=column_names),
            cells=dict(values=cells_values, format=["", ".3"]),
        )
    )

    # Update table layout
    fig.update_layout(
        title="Regression Performance Metrics",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Save table
    fig.write_image(f"{given_name}/performance/{data_type}_regression_metrics.png")

    logger.info("Created and saved regression metrics table")

    return


# To be used for feature exploration (TODO add more colors)
def plotDistributionViolin(given_name, feature_name, groups, values, data_type):
    """
    Plot violin plot of distribution of values across different groups.

    Parameters
    ----------
    given_name : str
    Name of the figure to be saved.

    groups : array_like
    Group labels of the data.

    values : array_like
    Values of the data.

    data_type : str
    Type of the data.

    Returns
    -------
    None
    Returns nothing, but saves the violin plot figure as a png file.

    Notes
    -----
    This function uses the Plotly library to create the violin plot. It first creates a pandas DataFrame from the input data, and then uses a for loop to create a violin plot for each group. The resulting figure is saved as a png file in the directory specified by given_name.
    """
    df = pd.DataFrame({"groups": groups, "values": values})

    colors = ["red", "green", "blue", "purple", "orange"]

    # Catching any kind of exception
    try:
        # Create Violin plot
        fig = go.Figure()
        for g in range(len(set(groups))):
            X = df[df["groups"] == g]["values"]
            fig.add_trace(
                go.Violin(
                    x=X,
                    name=str(g),  # name used in legend and hover labels
                    marker_color=colors[g],
                    opacity=0.75,
                )
            )
    except Exception as e:
        logger.info(f"Could not create distribution plot because of \n{e}")
        return

    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Group",
        title=f"Distribution plot - {data_type} data",
        width=1000,
        height=800,
    )

    fig.write_image(f"{given_name}/feature_info/{feature_name}_distributions.png")

    logger.info("Created and saved feature distribution plot")

    return


def modelPerformancePlots(
    clf, model_name, model_type, given_name, data_type, post_datasets, post_params
):
    # Performance and other post modeling plots
    # unpack dict
    X_all = post_datasets["X_all"]
    y_all, y_all_pred = post_datasets["y_all"], post_datasets["y_all_pred"]
    y_test_concat, y_test_pred = (
        post_datasets["y_test_concat"],
        post_datasets["y_test_pred"],
    )
    y_test_list = post_datasets["y_test_list"]

    if model_type == "classification":
        # probabilities only for classifications
        y_test_prob, y_all_prob, y_test_prob_list = (
            post_datasets["y_test_prob"],
            post_datasets["y_all_prob"],
            post_datasets["y_test_prob_list"],
        )

        # Threshold dependant
        if data_type == "test":
            plotConfusionMatrix(
                given_name,
                y_test_concat,
                y_test_prob,
                y_test_pred,
                post_params["file_type"],
                data_type=data_type,
            )

        elif data_type == "train":
            plotConfusionMatrix(
                given_name,
                y_all,
                y_all_prob,
                y_all_pred,
                post_params["file_type"],
                data_type=data_type,
            )

        if len(clf["final"].classes_) == 2:
            # Also create pr curve for class 0
            y_all_neg = np.array([1 - j for j in list(y_all)])
            y_all_prob_neg = np.array([1 - j for j in list(y_all_prob)])

            y_test_list_neg = [[1 - j for j in i] for i in y_test_list]
            y_test_prob_list_neg = [[1 - j for j in i] for i in y_test_prob_list]

            # Threshold independent
            if data_type == "test":
                plotClassificationCurve(
                    given_name,
                    y_test_list,
                    y_test_prob_list,
                    curve_type="roc",
                    data_type=data_type,
                )

                plotClassificationCurve(
                    given_name,
                    y_test_list,
                    y_test_prob_list,
                    curve_type="pr",
                    data_type=f"{data_type}_class_1",
                )
                plotClassificationCurve(
                    given_name,
                    y_test_list_neg,
                    y_test_prob_list_neg,
                    curve_type="pr",
                    data_type=f"{data_type}_class_0",
                )

                plotCalibrationCurve(
                    given_name,
                    y_test_list,
                    y_test_prob_list,
                    data_type=data_type,
                )

                plotProbabilityDistribution(
                    given_name,
                    y_test_concat,
                    y_test_prob,
                    data_type=data_type,
                )

            elif data_type == "train":
                plotClassificationCurve(
                    given_name,
                    y_all,
                    y_all_prob,
                    curve_type="roc",
                    data_type=data_type,
                )

                plotClassificationCurve(
                    given_name,
                    y_all,
                    y_all_prob,
                    curve_type="pr",
                    data_type=f"{data_type}_class_1",
                )

                plotClassificationCurve(
                    given_name,
                    y_all_neg,
                    y_all_prob_neg,
                    curve_type="pr",
                    data_type=f"{data_type}_class_0",
                )

                plotCalibrationCurve(given_name, y_all, y_all_prob, data_type=data_type)

                plotProbabilityDistribution(
                    given_name, y_all, y_all_prob, data_type=data_type
                )

        # If multiclass classification
        elif len(clf["final"].classes_) > 2:
            # loop through classes
            for c in clf["final"].classes_:
                # stating the class
                logger.info(f"\nFor class {c}:")

                # creating a list of all the classes except the current class
                other_class = [x for x in clf["final"].classes_ if x != c]

                # Get index of selected class in clf['final'].classes_
                class_index = list(clf["final"].classes_).index(c)

                # marking the current class as 1 and all other classes as 0
                y_test_list_ova = [
                    [0 if x in other_class else 1 for x in fold_]
                    for fold_ in y_test_list
                ]
                y_test_prob_list_ova = [
                    [x[class_index] for x in fold_] for fold_ in y_test_prob_list
                ]

                # concatonate probs together to one list for distribution plot
                y_test_ova = np.concatenate(y_test_list_ova, axis=0)
                y_test_prob_ova = np.concatenate(y_test_prob_list_ova, axis=0)

                # y_all_ova = [0 if x in other_class else 1 for x in y_all]
                # y_all_prob_ova = [x[class_index] for x in y_all_prob]

                # Threshold independent
                if data_type == "test":
                    # plotClassificationCurve(given_name, y_all_ova, y_all_prob_ova, curve_type='roc', data_type=f'train_class_{c}')
                    plotClassificationCurve(
                        given_name,
                        y_test_list_ova,
                        y_test_prob_list_ova,
                        curve_type="roc",
                        data_type=f"{data_type}_class_{c}",
                    )

                    # plotClassificationCurve(given_name, y_all_ova, y_all_prob_ova, curve_type='pr', data_type='train_class1')
                    plotClassificationCurve(
                        given_name,
                        y_test_list_ova,
                        y_test_prob_list_ova,
                        curve_type="pr",
                        data_type=f"{data_type}_class_{c}",
                    )

                    # multiClassPlotCalibrationCurvePlotly(given_name, y_all, pd.DataFrame(y_all_prob, columns=clf['final'].classes_), title='fun')
                    plotCalibrationCurve(
                        given_name,
                        y_test_list_ova,
                        y_test_prob_list_ova,
                        data_type=f"{data_type}_class_{c}",
                    )

                    # plotProbabilityDistribution(given_name, y_all_ova, y_all_prob_ova, data_type='train')
                    plotProbabilityDistribution(
                        given_name,
                        y_test_ova,
                        y_test_prob_ova,
                        data_type=f"{data_type}_class_{c}",
                    )

    # if regression
    elif model_type == "regression":
        if data_type == "test":
            plotYhatVsYSave(given_name, y_test_concat, y_test_pred, data_type=data_type)
            plotQuantileError(
                given_name, y_test_concat, y_test_pred, data_type=data_type
            )
            regressionMetricsTable(
                given_name,
                y_test_concat,
                y_test_pred,
                X_all,
                data_type=data_type,
            )
        elif data_type == "train":
            plotYhatVsYSave(given_name, y_all, y_all_pred, data_type=data_type)
            plotQuantileError(given_name, y_all, y_all_pred, data_type=data_type)
            regressionMetricsTable(
                given_name,
                y_all,
                y_all_pred,
                X_all,
                data_type=data_type,
            )


def postModellingPlots(
    clf, model_name, model_type, given_name, post_datasets, post_params
):
    # Create model performance plots for train and test data
    for data_type in ["train", "test"]:
        modelPerformancePlots(
            clf,
            model_name,
            model_type,
            given_name,
            data_type,
            post_datasets,
            post_params,
        )

    # Post modeling plots, specific per model but includes feature importance among others
    globals()[model_name].postModelPlots(
        clf["final"],
        given_name + "/feature_importance",
        post_params["file_type"],
    )
