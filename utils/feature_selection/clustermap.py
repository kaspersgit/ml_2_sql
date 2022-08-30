import plotly.graph_objects as go
import plotly.figure_factory as ff

import numpy as np
from scipy.spatial.distance import pdist, squareform


def plotClustermap(df, given_name, file_type, logging):
    # only numerical features can be used atm

    # data_array = data.view((np.float, len(data.dtype.names)))
    data_ = df.select_dtypes(exclude=['object', 'bool']).copy()
    data_corr = data_.corr()
    data_corr.fillna(0, inplace=True)
    data_corr = np.clip(data_corr, -1 , 1)
    # data_array = data_corr.transpose()
    labels = data_corr.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(data_corr, orientation='bottom', labels=labels)
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(data_corr, orientation='right', labels=labels)
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']

    heat_data = data_corr.loc[dendro_leaves,:]
    heat_data = heat_data.loc[:,dendro_leaves]

    x = dendro_leaves
    y = dendro_leaves
    z = heat_data

    # Edit hovertext
    hovertext = list()
    for yi, yy in enumerate(y):
        hovertext.append(list())
        for xi, xx in enumerate(x):
            hovertext[-1].append('x: {}<br />y: {}<br />corr.: {}'.format(xx, yy, z[yy][xx]))

    heatmap = [
        go.Heatmap(
            x = x,
            y = y,
            z = z,
            colorscale = 'Blues',
            hoverinfo = 'text',
            hovertext = hovertext
        )
    ]

    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout({'width':1200, 'height':1200,
                             'showlegend':False,
                            'hovermode': 'closest',
                             })
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks':""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""
                            })
    # Edit yaxis2
    fig.update_layout(yaxis2={'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""})

    if file_type == 'png':
        fig.write_image(f'{given_name}/feature_info/clustermap.png')
    elif file_type == 'html':
        # or as html file
        fig.write_html(f'{given_name}/feature_info/clustermap.html')
    print(f'Created and saved clustermap')
    logging.info(f'Created and saved clustermap')

