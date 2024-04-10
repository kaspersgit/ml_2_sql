import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import scipy.stats as ss
import logging

logger = logging.getLogger(__name__)


def plotClustermap(dfc, matrix_type, given_name, file_type):
    """
    Plot a clustermap of a given correlation matrix/dataframe and save it as either a png or an html file.

    Args:
        dfc (pandas DataFrame): A correlation matrix/dataframe.
        matrix_type (str): A string representing the type of matrix being plotted (e.g., 'correlation', 'covariance').
        given_name (str): A string representing the name of the project being worked on.
        file_type (str): A string representing the type of file to save the plot as (either 'png' or 'html').

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
                "x: {}<br />y: {}<br />corr.: {}".format(xx, yy, z[xx][yy])
            )

    # Figure out where correlation of 0 falls
    # Different color scale based on correlation coefficient used (some range from -1 to 1 and some from 0 to 1)
    if "pearson" in matrix_type:
        # Create custom colorscale
        colorscale = [
            [0, "rgba(214, 39, 40, 0.85)"],
            [0.5, "rgba(255, 255, 255, 0.85)"],
            [1, "rgba(6,54,21, 0.85)"],
        ]
        min_value = -1
        max_value = 1
    else:
        # Create custom colorscale
        colorscale = [[0, "rgba(255, 255, 255, 0.85)"], [1, "rgba(6,54,21, 0.85)"]]
        min_value = 0
        max_value = 1

    heatmap = [
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
            zmin=min_value,  # Set the minimum value for the colorscale
            zmax=max_value,  # Set the maximum value for the colorscale
            hoverinfo="text",
            hovertext=hovertext,
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
            "domain": [0.2, 1],
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
            "domain": [0, 0.2],
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
            "domain": [0, 0.8],
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
            "domain": [0.8, 0.95],
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


def plotPearsonCorrelation(df, given_name, file_type):
    """
    Creates a Pearson correlation matrix using numerical columns of a Pandas DataFrame,
    and generates a clustermap plot of the matrix using the plotClustermap function.

    Args:
        df (Pandas DataFrame): The DataFrame containing the data to analyze.
        given_name (str): The name of the project, used in the output file name.
        file_type (str): The file type to save the output as ('png' or 'html').

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
        logger.info("No numerical variables found")
        return
    elif data_.shape[1] == 1:
        logger.info(
            "Skip Pearson correlation matrix due to fewer than 2 numerical variables found"
        )
        return

    data_corr = data_.corr()
    data_corr.fillna(0, inplace=True)
    data_corr = np.clip(data_corr, -1, 1)

    matrix_type = "pearson_numeric"

    # Plot matrix
    plotClustermap(data_corr, matrix_type, given_name, file_type)

    # success message
    logger.info("Created Pearson correlation matrix (for numerical features)")


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


def plotCramervCorrelation(df, given_name, file_type):
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
        logger.info(
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
    matrix_type = "cramerv_categorical"

    # Plot matrix
    plotClustermap(df_cramerv, matrix_type, given_name, file_type)

    # Success message
    logger.info("Created CramerV correlation matrix (for categorical values)")


# Xi correlation (paper: https://arxiv.org/pdf/1909.10140.pdf)
def xicor_original(X, Y, ties=True):
    np.random.seed(42)

    # Convert to array if list
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)

    n = len(X)
    order = np.array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        L = np.array([sum(y >= Y[order]) for y in Y[order]])
        r = L.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = np.array([r[j] == r[i] for i in range(n)])
                r[tie_index] = np.random.choice(
                    r[tie_index] - np.arange(0, sum([r[j] == r[i] for i in range(n)])),
                    sum(tie_index),
                    replace=False,
                )
        return 1 - n * sum(abs(r[1:] - r[: n - 1])) / (2 * sum(L * (n - L)))
    else:
        r = np.array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum(abs(r[1:] - r[: n - 1])) / (n**2 - 1)


# Faster implementation 99.9% similar to above
def xicor(X, Y, ties=True):
    np.random.seed(42)
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)
    order = X.argsort()
    Y_sorted = Y[order]
    if ties:
        L = np.searchsorted(Y_sorted, Y_sorted, side="right")
        r = L.copy()
        for j in range(n):
            tie_index = r == r[j]
            tie_count = tie_index.sum()
            if tie_count > 1:
                tie_adjustment = np.arange(tie_count)
                np.random.shuffle(tie_adjustment)
                r[tie_index] = r[tie_index] - tie_adjustment
        return 1 - n * sum(abs(r[1:] - r[: n - 1])) / (2 * sum(L * (n - L)))
    else:
        r = np.searchsorted(Y_sorted, Y_sorted, side="right")
        return 1 - 3 * sum(abs(r[1:] - r[: n - 1])) / (n**2 - 1)


def plotXiCorrelation(df, given_name, file_type):
    """
    Creates a Xi correlation matrix using numerical columns of a Pandas DataFrame,
    and generates a clustermap plot of the matrix using the plotClustermap function.

    Args:
        df (Pandas DataFrame): The DataFrame containing the data to analyze.
        given_name (str): The name of the project, used in the output file name.
        file_type (str): The file type to save the output as ('png' or 'html').

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
        logger.info("No numerical variables found")
        return
    elif data_.shape[1] == 1:
        logger.info(
            "Skip Xi correlation matrix due to fewer than 2 numerical variables found"
        )
        return

    cols = data_.columns
    data_corr = pd.DataFrame(columns=cols, index=cols)
    for i in cols:
        for j in cols:
            if i == j:
                xicorr = 1
            else:
                xicorr = xicor(data_[i], data_[j])

            data_corr.at[i, j] = xicorr

    data_corr.fillna(0, inplace=True)
    data_corr = np.clip(data_corr, 0, 1)

    # cast matrix content to float
    data_corr = data_corr.astype(float)

    matrix_type = "xi_numeric"

    # Plot matrix
    plotClustermap(data_corr, matrix_type, given_name, file_type)

    # success message
    logger.info("Created Xi correlation matrix (for numerical features)")
