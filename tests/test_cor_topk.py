import pytest
import numpy as np
from pynetcor import cor


def test_topk_correlation():
    np.random.seed(42)
    k = 15
    XX = np.random.random((10, 5))

    cor_ref = np.corrcoef(XX).flatten()
    topk_order = np.argsort(-np.abs(cor_ref))[:k]
    topk_values = cor_ref[topk_order]
    val_ref = topk_values

    val = cor.cor_topk(XX, XX, k=k)[:, 2]

    assert np.allclose(val, val_ref)


if __name__ == "__main__":
    pytest.main()
