import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def plotConfusionMatrixSave(given_name, y_true, y_pred, data_type, logging):
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
    logging : logging object
        The object to log the status and messages.

    Returns
    -------
    None

    Notes
    -----
    Saves the confusion matrix in {given_name}/performance/{data_type}_confusion_matrix.png

    """

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    plt.xticks(rotation=90)
    plt.savefig('{given_name}/performance/{data_type}_confusion_matrix.png'.format(given_name=given_name, data_type=data_type), bbox_inches='tight')

    print('Confusion matrix saved')
    logging.info('Confusion matrix saved')

def classificationReportSave(given_name, y_true, y_pred, data_type, logging):
    """
    Save the classification report as a CSV file.

    Parameters
    ----------
    given_name : str
        The name of the project or experiment.
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    data_type : str
        The type of data being evaluated (e.g. 'train', 'test').
    logging : logging.Logger
        An instance of the logging.Logger class to record the event.

    Returns
    -------
    None
        This function does not return any value but saves the classification report as a CSV file.

    """

    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('{given_name}/performance/{data_type}_classification_report.csv'.format(given_name=given_name, data_type=data_type))

    print('Classification report saved')
    logging.info('Classification report saved')

def plotYhatVsYSave(given_name, y_true, y_pred, data_type, logging):
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
    logging : logging.Logger
        Logging instance to log information about the execution.

    Returns:
    --------
    None
    """

    plot_df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
    fig = px.scatter(plot_df, 'y_true', 'y_pred')
    fig.write_image('{given_name}/performance/{data_type}_scatter_yhat_vs_y.png'.format(given_name=given_name, data_type=data_type))

    print(f'Scatter plot of yhat vs y saved for {data_type}')
    logging.info(f'Scatter plot of yhat vs y saved for {data_type}')

def plotClassificationCurve(given_name, y_true, y_prob, curve_type, data_type, logging):
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
    logging : logging object
        Logging object for recording progress and error messages.

    Returns:
    --------
    mean_auc : float
        Mean area under the curve (AUC) value for all folds or for the fitted model.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    curve_type = curve_type.lower()

    if curve_type == 'roc':
        title = 'ROC curve - {data_type} data'.format(data_type=data_type)
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate (Recall)'
        diagCor1 = [0, 1]
        diagCor2 = [0, 1]

        curve_ = 'roc'

    elif ('recall' in curve_type and 'precision' in curve_type) or curve_type == 'pr':
        title = 'Precision-Recall curve - {data_type}'.format(data_type=data_type)
        xlabel = 'True Positive Rate (Recall)'
        ylabel = 'Precision'

        # add the random line (= share of positives in sample)
        if isinstance(y_true, list):
            pos_share = sum([sum(el) for el in y_true]) / sum([len(el) for el in y_true])
        else:
            pos_share = sum(y_true)/len(y_true)

        diagCor1 = [0, 1]
        diagCor2 = [pos_share, pos_share]

        curve_ = 'pr'

    # Create plot
    fig = go.Figure()

    # set axis range to 0 - 1
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1],
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      title=title,
                      width=1000,
                      height=800)

    # Add diagonal reference line
    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=diagCor1[0], x1=diagCor1[1], y0=diagCor2[0], y1=diagCor2[1],
                  line=dict(
                      color="black",
                      width=2,
                      dash="dot",
                  )
    )

    # Save the auc(s)
    auc_list = list()

    if isinstance(y_prob, list):
        for fold_id in range(len(y_prob)):

            if curve_ == 'roc':
                xVar, yVar, thresholds = roc_curve(y_true[fold_id], y_prob[fold_id])
            elif curve_ == 'pr':
                yVar, xVar, thresholds = precision_recall_curve(y_true[fold_id], y_prob[fold_id])

            # Calculate area under curve (AUC)
            auc_list.append(auc(xVar, yVar))
            fig.add_trace(go.Scatter(x=xVar, y=yVar, mode='lines', name=f'Fold {fold_id}'))
    else:

        if curve_ == 'roc':
            xVar, yVar, thresholds = roc_curve(y_true, y_prob)
        elif curve_ == 'pr':
            yVar, xVar, thresholds = precision_recall_curve(y_true, y_prob)

        # Calculate area under curve (AUC)
        auc_list.append(auc(xVar, yVar))
        fig.add_trace(go.Scatter(x=xVar, y=yVar, mode='lines', name=f'Fitted model'))

    # add (average) auc in image
    fig.add_annotation(x=0.5, y=0,
                       text=f"Mean AUC: {np.mean(auc_list)}",
                       showarrow=False,
                       yshift=10)

    fig.write_image(f'{given_name}/performance/{data_type}_{curve_type}_plot.png')

    print(f'Created and saved {curve_type} plot for {data_type} data')
    logging.info(f'Created and saved {curve_type} plot for {data_type} data')

    return np.mean(auc_list)

def plotCalibrationCurve(given_name, y_true, y_prob, data_type, logging):
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
    logging : logging.Logger
        An instance of the `logging.Logger` class.

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

    Examples
    --------
    >>> y_true = [0, 1, 1, 0, 1, 0]
    >>> y_prob = [0.2, 0.6, 0.7, 0.4, 0.8, 0.1]
    >>> plotCalibrationCurve("data", y_true, y_prob, "test", logging.getLogger())
    """

    # Create plot
    fig = go.Figure()

    # set axis range to 0 - 1
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1],
                      xaxis_title='Predicted probability', yaxis_title='Share being positive',
                      title=f'Calibration plot (reliability curve) - {data_type} data',
                      width=1000,
                      height=800)

    # Add diagonal reference line
    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=0, x1=1, y0=0, y1=1,
                  line=dict(
                      color="black",
                      width=2,
                      dash="dot",
                  )
    )

    # Save the brier score loss
    bsl_list = list()

    if isinstance(y_prob, list):
        for fold_id in range(len(y_prob)):
            # summaries actuals and predicted probs to (bins) number of points
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_true[fold_id], y_prob[fold_id], n_bins=10, strategy='quantile')

            # Calculate area under curve (AUC)
            bsl_list.append(brier_score_loss(y_true[fold_id], y_prob[fold_id]))
            fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='markers+lines', name=f'Fold {fold_id}'))
    else:
        # summaries actuals and predicted probs to (bins) number of points
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

        # Calculate area under curve (AUC)
        bsl_list.append(brier_score_loss(y_true, y_prob))
        fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='markers+lines', name=f'Fitted model'))

    # add (average) auc in image
    fig.add_annotation(x=0.5, y=0,
                       text=f"Mean Brier Score Loss: {np.mean(bsl_list)}",
                       showarrow=False,
                       yshift=10)

    fig.write_image(f'{given_name}/performance/{data_type}_calibration_plot.png')

    print(f'Created and saved calibration plot')
    logging.info(f'Created and saved calibration plot')

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
    calibration_df = pd.DataFrame(columns=['classification','fraction_of_positives','mean_predicted_value'])

    # loop through different classes
    for cl in probs.columns:
        y_true = np.where(actuals == cl, 1, 0)
        y_prob = probs[cl]

        # summarise actuals and predicted probs to (bins) number of points
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=bins, strategy='quantile')

        calibration_df = calibration_df.append(pd.DataFrame({'classification':cl, 'fraction_of_positives':fraction_of_positives, 'mean_predicted_value':mean_predicted_value}), ignore_index=True)

    # Create scatter plot
    fig = px.scatter(calibration_df, x='mean_predicted_value', y='fraction_of_positives', title='Calibration plot (reliability curve)', color='classification',width=1200, height=800)


    # Make trace be line plus dots
    for l in range(len(fig.data)):
        fig.data[l].update(mode='markers+lines')

    # set axis range to 0 - 1
    # fig.update_layout(xaxis_range=[-0.1,1.1], yaxis_range=[-0.1,1.1], xaxis_title='Predicted probability', yaxis_title='Fraction of positives')

    # Add diagonal reference line
    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=0, x1=1, y0=0, y1=1,
                  line=dict(
                      color="black",
                      width=2,
                      dash="dot",
                  )
    )

    fig.show(renderer='browser')

def plotProbabilityDistribution(given_name, y_true, y_prob, data_type, logging):
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
    logging : logger object
        Logging object for printing errors and warnings.

    Returns
    -------
    None
    """
    import plotly.figure_factory as ff
    df = pd.DataFrame({'actuals': y_true, 'prob': y_prob})
    do = df[df['actuals']==1]['prob']
    dont = df[df['actuals']==0]['prob']

    # Catching any kind of exception
    try:
        # Create distplot with custom bin_size
        fig = ff.create_distplot([do,dont], ['1','0'], colors=['green','red'], bin_size=.01)
    except Exception as e:
        print(f'Could not create distribution plot because of \n{e}')
        logging.info(f'Could not create distribution plot because of \n{e}')
        return

    # Update size of figure
    fig.update_layout(xaxis_title='Predicted probability', yaxis_title='Frequency',
                      title=f'Distribution plot - {data_type} data',
                      width=1000,
                      height=800)

    # fig.show(renderer='browser')

    fig.write_image(f'{given_name}/performance/{data_type}_distribution_plot.png')

    print(f'Created and saved probability distribution plot')
    logging.info(f'Created and saved probability distribution plot')

    return

def plotDistribution(given_name, groups, values, data_type, logging):
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
    logging: logger object
    A logger object that can be used to record messages.

    Returns
    -------
    None

    Raises
    ------
    Exception: If the histogram cannot be created.

    Example
    -------
    >>> groups = [0, 0, 1, 1, 2, 2]
    >>> values = [1, 2, 2, 3, 3, 3]
    >>> plotDistribution("myplot", groups, values, "training", logging.getLogger())
    """
    df = pd.DataFrame({'groups': groups, 'values': values})
    
    min_value = min(values)
    max_value = max(values)
    bin_size = (max_value - min_value) / 50
    
    colors = ['red','green','blue','purple','orange']

    # Catching any kind of exception
    try:
        # Create distplot 
        fig = go.Figure()
        for g in range(len(set(groups))):
            X = df[df['groups']==g]['values']
            fig.add_trace(go.Histogram(
                x=X,
                histnorm='probability density',
                name=str(g), # name used in legend and hover labels
                xbins=dict( # bins used for histogram
                    start=min_value,
                    end=max_value,
                    size=bin_size
                ),
                marker_color=colors[g],
                opacity=0.75
            ))
    except Exception as e:
        print(f'Could not create distribution plot because of \n{e}')
        logging.info(f'Could not create distribution plot because of \n{e}')
        return

    fig.update_layout(
        xaxis_title='Predicted probability',
        yaxis_title='Probability density',
        title=f'Distribution plot - {data_type} data',
        width=1000,
        height=800,
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )

    fig.write_image(f'{given_name}/performance/{data_type}_distribution_plot.png')

    print(f'Created and saved probability distribution plot')
    logging.info(f'Created and saved probability distribution plot')

    return

# To be used for feature exploration (TODO add more colors)
def plotDistributionViolin(given_name, groups, values, data_type, logging):
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

    logging : logging.Logger
    Logger object for logging messages.

    Returns
    -------
    None
    Returns nothing, but saves the violin plot figure as a png file.

    Notes
    -----
    This function uses the Plotly library to create the violin plot. It first creates a pandas DataFrame from the input data, and then uses a for loop to create a violin plot for each group. The resulting figure is saved as a png file in the directory specified by given_name.

    Examples
    --------
    >>> plotDistributionViolin('my_figure', [0, 1, 0, 1], [2.5, 3.7, 4.2, 5.3], 'continuous', logging)
    """
    df = pd.DataFrame({'groups': groups, 'values': values})
    
    colors = ['red','green','blue','purple','orange']

    # Catching any kind of exception
    try:
        # Create Violin plot 
        fig = go.Figure()
        for g in range(len(set(groups))):
            X = df[df['groups']==g]['values']
            fig.add_trace(go.Violin(
                x=X,
                name=str(g), # name used in legend and hover labels
                marker_color=colors[g],
                opacity=0.75
            ))
    except Exception as e:
        print(f'Could not create distribution plot because of \n{e}')
        logging.info(f'Could not create distribution plot because of \n{e}')
        return

    fig.update_layout(
        xaxis_title='Value',
        yaxis_title='Group',
        title=f'Distribution plot - {data_type} data',
        width=1000,
        height=800
    )

    fig.write_image(f'{given_name}/feature_info/{feature_name}_class_distributions.png')

    print(f'Created and saved feature distribution plot')
    logging.info(f'Created and saved feature distribution plot')

    return
