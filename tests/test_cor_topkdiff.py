import pytest
import numpy as np
from pynetcor import cor


def test_topk_differential_correlation():
    np.random.seed(42)
    m, n = 20, 10
    k = 15

    m1 = (np.random.random((m, n)) - 0.5) * 10
    m2 = (np.random.random((m, n)) - 0.5) * 10

    cm1 = np.corrcoef(m1)
    cm2 = np.corrcoef(m2)
    m_diff = cm1 - cm2
    v_diff = m_diff.flatten()
    order = np.argsort(-np.abs(v_diff))

    topkdiff = cor.cor_topkdiff(
        x1=m1, y1=m2, x2=m1, y2=m2, k=k, method="pearson", threads=8
    )[:, 2]

    assert np.allclose(np.abs(v_diff[order])[:k], np.abs(topkdiff))


if __name__ == "__main__":
    pytest.main()
