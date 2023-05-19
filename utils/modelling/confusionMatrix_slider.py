##############
## new attempt ## 
import numpy as np 
nsize = 10000
y_prob = np.random.rand(nsize)
y_true = np.random.choice([0,1], size=nsize)


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd 

conf_matrices = []
steps=[]

threshold_list = np.arange(0.0, 1.05, 0.05)

for threshold in threshold_list:
    y_pred = [1 if x > threshold else 0 for x in y_prob]

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # confusion matrix
    cm = []
    for p in {0,1}:
        cm.append([])
        for a in {0,1}:
            cm[p].append(len(df[(df['y_pred'] == p) & (df['y_true'] == a)])/len(y_pred))

    trfa = len(df[(df['y_pred'] == 1) & (df['y_true'] == 0)])/len(y_pred)
    trtr = len(df[(df['y_pred'] == 1) & (df['y_true'] == 1)])/len(y_pred)
    fafa = len(df[(df['y_pred'] == 0) & (df['y_true'] == 0)])/len(y_pred)
    fatr = len(df[(df['y_pred'] == 0) & (df['y_true'] == 1)])/len(y_pred)

    # Calculate metrics
    # ignore div by 0 or 0/0 warning and just state nan
    cm_metrics = {"0":{}, "1": {}}
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_metrics["0"]["precision"] = np.float64(cm[1][1]) / (cm[1][0] + cm[1][1]))
        cm_metrics["0"]["recall"] = np.float64(cm[1][1]) / (cm[1][1] + cm[0][1]))
        cm_metrics["0"]["f1"] = np.float64(2 * cm_metrics["0"]["precision"] * cm_metrics["0"]["recall"]) / (cm_metrics["0"]["precision"] + cm_metrics["0"]["recall"])
        cm_metrics["0"]["accuracy"] = np.float64(cm[1][1] + cm[0][0]) / sum(cm[0] + cm[1]))

        precision = np.float64(trtr) / (trtr + trfa)
        recall = np.float64(trtr) / (trtr + fatr)
        f1 = np.float64(2 * precision * recall) / (precision + recall)
        accuracy = np.float64(trtr + fafa) / (trtr + trfa + fatr + fafa)

    # force 3 decimal places
    metrics = [['F1-score','Accuracy','Recall','Precision'],
                [f1, accuracy, recall, precision]]

    conf_matrices.append(cm)

    steps.append(dict(method = "restyle",
                args = [{'z': [ cm ], #in the initial fig update z and text
                    'text': [cm],
                    'cells.values':[metrics]}],
                    label=round(threshold,2),
                    ))
    
    if threshold == 0.5:
        conf_matrix = cm
        table_metrics = metrics


labels = ["Positive", "Negative"]
labels_r = labels.copy()
labels_r.reverse()

# Make subplots top table bottom heatmap
fig = make_subplots(rows=1, cols=2,
                    specs=[[{'type': 'table'},
                           {'type': 'heatmap'}]])

# Create heatmap 
heatmap = go.Heatmap(x=labels, y=labels_r,
                    z=conf_matrix,
                    text=conf_matrix,
                    texttemplate="%{text:.3f}",
                    hovertemplate="<br>".join([
                        "Predicted: %{x}",
                        "Actual: %{y}",
                        "# cases: %{z}",
                        "<extra></extra>"
                    ])
)


# Create table
metrics_table = go.Table(header=dict(values=['Metric', 'Value']),
                 cells=dict(values=table_metrics, format=["", ".3f"])
                         )

# Add table and heatmap to figure 
fig.add_trace(metrics_table, row=1, col=1)
fig.add_trace(heatmap, row=1, col=2)



fig.update_layout(width=1000, height=600,
                  xaxis_title="Actual",
                  yaxis_title="Predicted",
                  xaxis_type="category", 
                  yaxis_type="category", 
                  xaxis_side="top",
                  title_text= "Normalized confusion matrix", 
                  title_x=0.5,
                  )

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Threshold: "},
    pad={"t": 50}, #sort it out later
    steps=steps
)]

fig.update_layout(sliders=sliders)
fig.show(renderer='browser')





### original
##############
    import xarray as xr

    cms = []

    threshold_list = np.arange(0.0, 1.05, 0.05)

    for threshold in threshold_list:
        y_pred = [1 if x > threshold else 0 for x in y_prob]

        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

        # predicted / actual 
        trfa = len(df[(df['y_pred'] == 1) & (df['y_true'] == 0)])
        trtr = len(df[(df['y_pred'] == 1) & (df['y_true'] == 1)])
        fafa = len(df[(df['y_pred'] == 0) & (df['y_true'] == 0)])
        fatr = len(df[(df['y_pred'] == 0) & (df['y_true'] == 1)])

        z = [[trtr, trfa],
                [fatr, fafa]]  
        
        cms.append(z)

    # Round to 2 decimals 
    threshold_list = [round(i,2) for i in threshold_list]

    # convert to xarray
    da = xr.DataArray(cms, coords=[threshold_list, ['1', '0'], ['1', '0']], dims=['Threshold', 'Predicted', 'Actual'])

    # Create figure
    fig = px.imshow(da, 
                    title='Confusion Matrix',
                    animation_frame='Threshold',
                    text_auto=True,
                    width=750, height=750, 
                    labels=dict(animation_frame="Threshold"))

    # Add metrics per frame
    for frame in fig.frames:
        
        # Get confusion matrix values of frame
        trtr = frame.data[0].z[0,0]
        trfa = frame.data[0].z[0,1]
        fatr = frame.data[0].z[1,0]
        fafa = frame.data[0].z[1,1]

        # Calculate metrics
        # ignore div by 0 or 0/0 warning and just state nan
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.float64(trtr) / (trtr + trfa)
            recall = np.float64(trtr) / (trtr + fatr)
            f1 = np.float64(2 * precision * recall) / (precision + recall)
            accuracy = np.float64(trtr + fafa) / (trtr + trfa + fatr + fafa)

        frame.layout['title'] = f'Confusion Matrix<br><sup>Precision: {precision:.2f}\tRecall: {recall:.2f}\tF1-Score: {f1:.2f}\tAccuracy: {accuracy:.2f}</sup>'

    # set default slider value to 0.5 and update layout and trace accordingly
    fig.layout.sliders[0]['active'] = 10  
    fig.update_layout(fig.frames[10].layout)
    fig.update_traces(z=fig.frames[10].data[0].z)

    fig.update_layout(
        xaxis = dict(
            title='Actual',
            dtick=1,
        ),
        yaxis = dict(
            title='Predicted',
            dtick=1,
        )
    )

    fig["layout"].pop("updatemenus")

    fig.write_html(f'{given_name}/performance/{data_type}_confusion_matrix.html', auto_play=False)

    print(f'Created and saved confusion matrix for {data_type} data')
    logging.info(f'Created and saved confusion matrix for {data_type} data')
#####################3

import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,
                    specs=[[{'rowspan': 2}, {}], [None, {}]],
                    subplot_titles=('First Subplot', 'Second Subplot', 'Third Subplot'))

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=1)
fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=2, col=2)

fig.update_layout(title='Unequal Subplots')
fig.show(renderer='browser')