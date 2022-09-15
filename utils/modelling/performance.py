import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def plotConfusionMatrixSave(given_name, y_true, y_pred, data_type, logging):

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    plt.xticks(rotation=90)
    plt.savefig('{given_name}/performance/{data_type}_confusion_matrix.png'.format(given_name=given_name, data_type=data_type), bbox_inches='tight')

    print('Confusion matrix saved')
    logging.info('Confusion matrix saved')

def classificationReportSave(given_name, y_true, y_pred, data_type, logging):

    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('{given_name}/performance/{data_type}_classification_report.csv'.format(given_name=given_name, data_type=data_type))

    print('Classification report saved')
    logging.info('Classification report saved')

def plotYhatVsYSave(given_name, y_true, y_pred, data_type, logging):

    plot_df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
    fig = px.scatter(plot_df, 'y_true', 'y_pred')
    fig.write_image('{given_name}/performance/{data_type}_scatter_yhat_vs_y.png'.format(given_name=given_name, data_type=data_type))

    print(f'Scatter plot of yhat vs y saved for {data_type}')
    logging.info(f'Scatter plot of yhat vs y saved for {data_type}')

def plotClassificationCurve(given_name, y_true, y_prob, curve_type, data_type, logging):
    """
    :param given_name: string path where to save
    :param probs: float models predicted probability
    :param y_true: int 1 or 0 for the y_true outcome
    :param title: str title of the plot
    :param curve_type: str either ROC or PR
    :param savefig: boolean if plot should be saved
    :param saveFileName: str where to save the plot to
    :return:
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
    Plot the calibration curve for a set of true and predicted values

    :param actuals: true target value
    :param probs: predicted probability of target
    :param bins: how many bins to divide data in for plotting
    :param savefig: boolean if plot should be saved
    :param saveFileName: str path to which to save the plot
    :return: calibration plot
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
                calibration_curve(y_true[fold_id], y_prob[fold_id], n_bins=10)

            # Calculate area under curve (AUC)
            bsl_list.append(brier_score_loss(y_true[fold_id], y_prob[fold_id]))
            fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='markers+lines', name=f'Fold {fold_id}'))
    else:
        # summaries actuals and predicted probs to (bins) number of points
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=10)

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

# for multiclass classification WIP
def multiClassPlotCalibrationCurvePlotly(actuals, probs, title, bins=10):
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
            calibration_curve(y_true, y_prob, n_bins=bins)

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

    fig.show()

def plotProbabilityDistribution(given_name, y_true, y_prob, data_type, logging):
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
