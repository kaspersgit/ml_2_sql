##############
## new attempt ## 
import numpy as np 
nsize = 10000
y_prob = np.random.rand(nsize)
y_true = np.random.choice([0,1], size=nsize)


import plotly.graph_objects as go
from plotly.tools import make_subplots
import numpy as np
import pandas as pd 

conf_matrices = []
steps=[]

threshold_list = np.arange(0.0, 1.05, 0.05)

for threshold in threshold_list:
    y_pred = [1 if x > threshold else 0 for x in y_prob]

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # predicted / actual 
    trfa = len(df[(df['y_pred'] == 1) & (df['y_true'] == 0)])/len(y_pred)
    trtr = len(df[(df['y_pred'] == 1) & (df['y_true'] == 1)])/len(y_pred)
    fafa = len(df[(df['y_pred'] == 0) & (df['y_true'] == 0)])/len(y_pred)
    fatr = len(df[(df['y_pred'] == 0) & (df['y_true'] == 1)])/len(y_pred)

    # Calculate metrics
    # ignore div by 0 or 0/0 warning and just state nan
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.float64(trtr) / (trtr + trfa)
        recall = np.float64(trtr) / (trtr + fatr)
        f1 = np.float64(2 * precision * recall) / (precision + recall)
        accuracy = np.float64(trtr + fafa) / (trtr + trfa + fatr + fafa)

    metrics = [['F1-score','Accuracy','Recall','Precision'],
                [round(f1,3), round(accuracy,3), round(recall,3), round(precision,3)]]

    z = [[trtr, trfa],
            [fatr, fafa]]  
    
    # round to 3 decimals
    z = np.round(z,3)

    conf_matrices.append(z)

    steps.append(dict(method = "restyle",
                args = [{'z': [ z ], #in the initial fig update z and text
                    'text': [z],
                    'cells.values':[metrics]}],
                    label=round(threshold,2),
                    ))
    
    if threshold == 0.5:
        conf_matrix = z
        table_metrics = metrics


labels = ["Positive", "Negative"]
labels_r = labels.copy()
labels_r.reverse()

# Make subplots top table bottom heatmap
fig = make_subplots(rows=2, cols=1,
                    specs=[[{'type': 'table'}],
                           [{'type': 'heatmap'}]])

# Create heatmap 
heatmap = go.Heatmap(x=labels, y=labels_r,
                    z=conf_matrix,
                    text=conf_matrix,
                    texttemplate="%{text}",
                    hovertemplate="<br>".join([
                        "Predicted: %{x}",
                        "Actual: %{y}",
                        "# cases: %{z}",
                        "<extra></extra>"
                    ])
)


# Create heatmap 
metrics_table = go.Table(header=dict(values=['Metric', 'Value']),
                 cells=dict(values=table_metrics))

# Add table and heatmap to figure 
fig.add_trace(metrics_table, row=1, col=1)
fig.add_trace(heatmap, row=2, col=1)



fig.update_layout(width=600, height=800,
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
fig.show()





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