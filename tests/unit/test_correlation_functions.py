import pandas as pd
import numpy as np
import pytest

from ml2sql.utils.feature_selection.correlations import (
    cramers_corrected_stat,
    xicor,
    create_correlation_matrix,
)


def test_cramers_corrected_stat():
    # Mock chi2_contingency function from scipy.stats
    confusion_matrix = pd.DataFrame([[10, 5, 5], [2, 10, 10]])
    result = cramers_corrected_stat(confusion_matrix)

    assert np.isclose(result, 0.4, atol=1e-1)  # Test correlation with tolerance


def test_xicor():
    # Sample data arrays
    X = np.arange(3, 100, 2)
    Y = X + 10 * np.sin(X)

    result = xicor(X, Y)
    assert np.isclose(result, 0.7, atol=1e-1)  # Test correlation with tolerance


def test_create_correlation_matrix_pearson():
    # Mock corr function to return a sample correlation matrix
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    corr_matrix = create_correlation_matrix(data, "pearson")

    assert isinstance(corr_matrix, pd.DataFrame)
    assert np.all((corr_matrix >= -1) & (corr_matrix <= 1))
    assert corr_matrix.shape == (2, 2)  # Check dimensions


def test_create_correlation_matrix_cramerv():
    # Mock crosstab function to return a sample contingency table
    data = pd.DataFrame({"col1": ["A", "A", "B", "B"], "col2": ["C", "D", "C", "D"]})
    corr_matrix = create_correlation_matrix(data, "cramerv")

    assert isinstance(corr_matrix, pd.DataFrame)
    assert np.all((corr_matrix >= -1) & (corr_matrix <= 1))
    assert corr_matrix.shape == (2, 2)  # Check dimensions


def test_create_correlation_matrix_invalid_type():
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})

    with pytest.raises(ValueError):
        create_correlation_matrix(data, "invalid_type")
