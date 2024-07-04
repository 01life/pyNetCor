import pytest
import numpy as np
import scipy.stats
from pynetcor import cor


def test_pvalue_calculation():
    np.random.seed(42)
    XX = np.random.random((10, 5))

    correlations = []
    pvalues = []
    for i in range(XX.shape[1]):
        for j in range(XX.shape[1]):
            r, p = scipy.stats.pearsonr(XX[:, i], XX[:, j])
            correlations.append(r)
            pvalues.append(p)
    correlations = np.asarray(correlations)
    pvalues = np.asarray(pvalues)

    derived_pvalues = cor.pvalue_student_t(
        correlations, XX.shape[0] - 2, approx=False, threads=8
    )

    assert np.all(np.isclose(pvalues, derived_pvalues))


def test_pvalue_approximation():
    np.random.seed(42)
    XX = np.random.random((10, 5))

    correlations = []
    pvalues = []
    for i in range(XX.shape[1]):
        for j in range(XX.shape[1]):
            r, p = scipy.stats.pearsonr(XX[:, i], XX[:, j])
            correlations.append(r)
            pvalues.append(p)
    correlations = np.asarray(correlations)
    pvalues = np.asarray(pvalues)

    derived_pvalues_approx = cor.pvalue_student_t(
        correlations, XX.shape[0] - 2, approx=True, threads=8
    )

    assert np.all(np.isclose(pvalues, derived_pvalues_approx, atol=1e-2))


if __name__ == "__main__":
    pytest.main()
