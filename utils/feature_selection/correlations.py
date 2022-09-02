import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import scipy.stats as ss


def plotClustermap(dfc, matrix_type, given_name, file_type, logging):

    # using correlation matrix/dataframe as input
    labels = dfc.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(dfc, orientation='bottom', labels=labels)
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(dfc, orientation='right', labels=labels)
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']

    heat_data = dfc.loc[dendro_leaves,:]
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
        fig.write_image(f'{given_name}/feature_info/{matrix_type}_clustermap.png')
    elif file_type == 'html':
        # or as html file
        fig.write_html(f'{given_name}/feature_info/{matrix_type}_clustermap.html')

    print(f'Created and saved {matrix_type} clustermap')
    logging.info(f'Created and saved {matrix_type} clustermap')

def plotPearsonCorrelation(df, given_name, file_type, logging):
    # Numerical values
    # Creating pearson correlation matrix
    print(f'Creating Pearson correlation matrix')
    logging.info(f'Creating Pearson correlation matrix')

    data_ = df.select_dtypes(exclude=['object', 'category', 'bool']).copy()

    # if above dataframe is empty then skip function
    if data_.empty:
        print(f'No numerical variables found')
        logging.info(f'No numerical variables found')
        return

    data_corr = data_.corr()
    data_corr.fillna(0, inplace=True)
    data_corr = np.clip(data_corr, -1 , 1)

    matrix_type = 'numeric'

    # Plot matrix
    plotClustermap(data_corr, matrix_type, given_name, file_type, logging)


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def plotCramervCorrelation(df, given_name, file_type, logging):
    # Nominal Categorical values
    # Creating CramerV correlation matrix
    print(f'Creating CramerV correlation matrix')
    logging.info(f'Creating CramerV correlation matrix')

    data_ = df.select_dtypes(include=['category', 'object', 'bool']).copy()

    # if above dataframe is empty then skip function
    if data_.empty:
        print(f'No categorical variables found')
        logging.info(f'No categorical variables found')
        return

    cols = data_.columns
    df_cramerv = pd.DataFrame(columns=cols, index=cols)
    for i in cols:
        for j in cols:
            if i == j:
                cramervcorr = 1
            else:
                cm = pd.crosstab(data_[i], data_[j])
                cramervcorr = cramers_corrected_stat(cm)

            df_cramerv.at[i,j] = cramervcorr

    # Clean up values
    df_cramerv.fillna(0, inplace=True)
    df_cramerv = np.clip(df_cramerv, 0, 1)

    # set matrix data type
    matrix_type = 'categorical'

    # Plot matrix
    plotClustermap(df_cramerv, matrix_type, given_name, file_type, logging)

