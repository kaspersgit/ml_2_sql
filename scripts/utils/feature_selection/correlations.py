import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import scipy.stats as ss


def plotClustermap(dfc, matrix_type, given_name, file_type, logging):
    """
    Plot a clustermap of a given correlation matrix/dataframe and save it as either a png or an html file.

    Args:
        dfc (pandas DataFrame): A correlation matrix/dataframe.
        matrix_type (str): A string representing the type of matrix being plotted (e.g., 'correlation', 'covariance').
        given_name (str): A string representing the name of the project being worked on.
        file_type (str): A string representing the type of file to save the plot as (either 'png' or 'html').
        logging (logging.Logger): A logging object used for logging events.

    Returns:
        None
    """

    # using correlation matrix/dataframe as input
    labels = dfc.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(dfc, orientation="bottom", labels=labels)
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(dfc, orientation="right", labels=labels)
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]

    heat_data = dfc.loc[dendro_leaves, :]
    heat_data = heat_data.loc[:, dendro_leaves]

    x = dendro_leaves
    y = dendro_leaves
    z = heat_data

    # Edit hovertext
    hovertext = list()
    for yi, yy in enumerate(y):
        hovertext.append(list())
        for xi, xx in enumerate(x):
            hovertext[-1].append(
                "x: {}<br />y: {}<br />corr.: {}".format(xx, yy, z[yy][xx])
            )

    heatmap = [
        go.Heatmap(
            x=x, y=y, z=z, colorscale="Blues", hoverinfo="text", hovertext=hovertext
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout(
        {
            "width": 1200,
            "height": 1200,
            "showlegend": False,
            "hovermode": "closest",
        }
    )
    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
        }
    )
    # Edit xaxis2
    fig.update_layout(
        xaxis2={
            "domain": [0, 0.15],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )
    # Edit yaxis2
    fig.update_layout(
        yaxis2={
            "domain": [0.825, 0.975],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    if file_type == "png":
        fig.write_image(f"{given_name}/feature_info/{matrix_type}_clustermap.png")
    elif file_type == "html":
        # or as html file
        fig.write_html(f"{given_name}/feature_info/{matrix_type}_clustermap.html")


def plotPearsonCorrelation(df, given_name, file_type, logging):
    """
    Creates a Pearson correlation matrix using numerical columns of a Pandas DataFrame,
    and generates a clustermap plot of the matrix using the plotClustermap function.

    Args:
        df (Pandas DataFrame): The DataFrame containing the data to analyze.
        given_name (str): The name of the project, used in the output file name.
        file_type (str): The file type to save the output as ('png' or 'html').
        logging (logging.Logger): The logger object to log progress and errors.

    Returns:
        None

    Raises:
        None

    """
    # Numerical values
    # Creating pearson correlation matrix
    data_ = df.select_dtypes(exclude=["object", "category", "bool"]).copy()

    # if above dataframe is empty then skip function
    if data_.empty:
        logging.info("No numerical variables found")
        return
    elif data_.shape[1] == 1:
        logging.info(
            "Skip Pearson correlation matrix due to fewer than 2 numerical variables found"
        )
        return

    data_corr = data_.corr()
    data_corr.fillna(0, inplace=True)
    data_corr = np.clip(data_corr, -1, 1)

    matrix_type = "numeric"

    # Plot matrix
    plotClustermap(data_corr, matrix_type, given_name, file_type, logging)

    # success message
    logging.info("Created Pearson correlation matrix (for numerical features)")


def cramers_corrected_stat(confusion_matrix):
    """
    Calculates the corrected Cramer's V statistic for categorical-categorical association.

    Parameters:
    -----------
    confusion_matrix: array-like, shape (n_categories, n_categories)
        A confusion matrix representing the association between two categorical variables.

    Returns:
    --------
    cramers_v: float
        The corrected Cramer's V statistic for the given confusion matrix.

    References:
    -----------
    1. Bergsma, Wicher, and Marcel A. Croon. "Marginal models for dependent, clustered, and longitudinal categorical data." Springer Science & Business Media, 2009.
    2. Bergsma, Wicher, and Marcel A. Croon. "Reply to 'A note on the gamma statistic for measuring nominal association'." Journal of the Korean Statistical Society 42.3 (2013): 323-328.
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plotCramervCorrelation(df, given_name, file_type, logging):
    """
    Plot CramerV correlation matrix for nominal categorical values.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing categorical variables for which correlation is to be computed
    given_name : str
        Given name for the output file containing the correlation matrix plot
    file_type : str
        Type of the output file, e.g., png, pdf, etc.
    logging : logging.Logger
        Logging object for recording the progress and errors during runtime.

    Returns:
    --------
    None

    Raises:
    -------
    None

    Notes:
    ------
    This function computes the Cramer's V correlation matrix for the categorical variables in the input dataframe.
    If the dataframe does not have at least two categorical variables, the function will skip the computation and return.
    The correlation matrix is plotted using the seaborn clustermap function, which creates a heatmap of the correlation
    matrix, and performs hierarchical clustering of the variables.

    References:
    -----------
    - Bergsma, W., & Wicher, M. (2013).
      Journal of the Korean Statistical Society, 42(3), 323-328.
    """
    # Nominal Categorical values
    # Creating CramerV correlation matrix
    data_ = df.select_dtypes(include=["category", "object", "bool"]).copy()

    # if above dataframe is empty then skip function
    if data_.shape[1] < 2:
        logging.info(
            "Skip CramerV correlation matrix due to fewer than 2 categorical variables found"
        )
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

            df_cramerv.at[i, j] = cramervcorr

    # Clean up values
    df_cramerv.fillna(0, inplace=True)
    df_cramerv = np.clip(df_cramerv, 0, 1)

    # cast matrix content to float
    df_cramerv = df_cramerv.astype(float)

    # set matrix data type
    matrix_type = "categorical"

    # Plot matrix
    plotClustermap(df_cramerv, matrix_type, given_name, file_type, logging)

    # Success message
    logging.info("Created CramerV correlation matrix (for categorical values)")
