import pytest
import numpy as np
from pynetcor import cor


def test_full_correlation_matrix():
    np.random.seed(42)
    XX = np.random.random((5, 10))

    cor_default_values = np.corrcoef(XX)
    cor_test_values = cor.corrcoef(XX)

    assert np.allclose(cor_default_values, cor_test_values)


if __name__ == "__main__":
    pytest.main()
