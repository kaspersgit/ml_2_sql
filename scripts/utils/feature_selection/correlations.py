import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats as ss

logger = logging.getLogger(__name__)

# Define constants
PEARSON_COLORSCALE = [
    [0, "rgba(214, 39, 40, 0.85)"],
    [0.5, "rgba(255, 255, 255, 0.85)"],
    [1, "rgba(6,54,21, 0.85)"],
]

OTHER_COLORSCALE = [[0, "rgba(255, 255, 255, 0.85)"], [1, "rgba(6,54,21, 0.85)"]]


def cramers_corrected_stat(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculates the corrected Cramer's V statistic for categorical-categorical association.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        A confusion matrix representing the association between two categorical variables.

    Returns
    -------
    float
        The corrected Cramer's V statistic for the given confusion matrix.

    References
    ----------
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


def xicor(X: np.ndarray, Y: np.ndarray, ties: bool = True) -> float:
    """
    Calculates the Xi correlation coefficient between two numerical variables.

    Parameters
    ----------
    X : np.ndarray
        The first numerical variable.
    Y : np.ndarray
        The second numerical variable.
    ties : bool, optional
        Whether to handle ties in the data, by default True.

    Returns
    -------
    float
        The Xi correlation coefficient between X and Y.

    References
    ----------
    https://arxiv.org/pdf/1909.10140.pdf
    """
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


def create_correlation_matrix(
    df: pd.DataFrame, corr_type: str
) -> Optional[pd.DataFrame]:
    """
    Creates a correlation matrix based on the specified correlation type.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    corr_type : str
        The type of correlation to calculate ('pearson', 'cramerv', or 'xi').

    Returns
    -------
    Optional[pd.DataFrame]
        The correlation matrix as a DataFrame, or None if there are insufficient variables.

    Raises
    ------
    ValueError
        If an invalid correlation type is provided.
    """
    if corr_type == "pearson":
        data = df.select_dtypes(exclude=["object", "category", "bool"])
        if data.empty:
            logger.info("No numerical variables found")
            return None
        elif data.shape[1] < 2:
            logger.info(
                "Skip Pearson correlation matrix due to fewer than 2 numerical variables found"
            )
            return None
        corr_matrix = data.corr()
    elif corr_type == "cramerv":
        data = df.select_dtypes(include=["category", "object", "bool"])
        if data.shape[1] < 2:
            logger.info(
                "Skip CramerV correlation matrix due to fewer than 2 categorical variables found"
            )
            return None
        cols = data.columns
        corr_matrix = pd.DataFrame(columns=cols, index=cols)
        for i in cols:
            for j in cols:
                if i == j:
                    corr_matrix.at[i, j] = 1
                else:
                    cm = pd.crosstab(data[i], data[j])
                    corr_matrix.at[i, j] = cramers_corrected_stat(cm)
    elif corr_type == "xi":
        data = df.select_dtypes(exclude=["object", "category", "bool"])
        if data.empty:
            logger.info("No numerical variables found")
            return None
        elif data.shape[1] < 2:
            logger.info(
                "Skip Xi correlation matrix due to fewer than 2 numerical variables found"
            )
            return None
        cols = data.columns
        corr_matrix = pd.DataFrame(columns=cols, index=cols)
        for i in cols:
            for j in cols:
                if i == j:
                    corr_matrix.at[i, j] = 1
                else:
                    corr_matrix.at[i, j] = xicor(data[i], data[j])
    else:
        raise ValueError(f"Invalid correlation type: {corr_type}")

    corr_matrix.fillna(0, inplace=True)
    if corr_type in ["pearson"]:
        corr_matrix = np.clip(corr_matrix, -1, 1)
    else:
        corr_matrix = np.clip(corr_matrix, 0, 1)

    corr_matrix = corr_matrix.astype(float)
    return corr_matrix


def plot_clustermap(
    corr_matrix: pd.DataFrame,
    matrix_type: str,
    project_name: str,
    file_type: str,
) -> None:
    """
    Plot a clustermap of a given correlation matrix and save it as either a png or an html file.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        A correlation matrix.
    matrix_type : str
        A string representing the type of matrix being plotted (e.g., 'correlation', 'covariance').
    project_name : str
        A string representing the name of the project being worked on.
    file_type : str
        A string representing the type of file to save the plot as (either 'png' or 'html').

    Returns
    -------
    None
    """
    labels = corr_matrix.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(corr_matrix, orientation="bottom", labels=labels)
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(corr_matrix, orientation="right", labels=labels)
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]

    heat_data = corr_matrix.loc[dendro_leaves, :]
    heat_data = heat_data.loc[:, dendro_leaves]

    x = dendro_leaves
    y = dendro_leaves
    z = heat_data

    # Edit hovertext
    hovertext = [
        ["x: {}<br />y: {}<br />corr.: {}".format(xx, yy, z[xx][yy]) for xx in x]
        for yy in y
    ]

    # Figure out color scale
    if "pearson" in matrix_type:
        colorscale = PEARSON_COLORSCALE
        min_value = -1
        max_value = 1
    else:
        colorscale = OTHER_COLORSCALE
        min_value = 0
        max_value = 1

    heatmap = [
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
            zmin=min_value,
            zmax=max_value,
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
        width=1200,
        height=1200,
        showlegend=False,
        hovermode="closest",
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

    output_path = f"{project_name}/feature_info/{matrix_type}_clustermap.{file_type}"
    if file_type == "png":
        fig.write_image(output_path)
    elif file_type == "html":
        fig.write_html(output_path)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    logger.info(f"Created {matrix_type} correlation matrix clustermap")


def plot_correlations(df: pd.DataFrame, project_name: str, file_type: str) -> None:
    """
    Plot correlation matrices for the given DataFrame and save them as png or html files.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    project_name : str
        A string representing the name of the project being worked on.
    file_type : str
        A string representing the type of file to save the plots as (either 'png' or 'html').

    Returns
    -------
    None
    """
    for corr_type in ["pearson", "cramerv", "xi"]:
        corr_matrix = create_correlation_matrix(df, corr_type)
        if corr_matrix is not None:
            matrix_type = f"{corr_type}_{'numeric' if corr_type in ['pearson', 'xi'] else 'categorical'}"
            plot_clustermap(corr_matrix, matrix_type, project_name, file_type)


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("path/to/your/data.csv")
    plot_correlations(data, "my_project", "png")
