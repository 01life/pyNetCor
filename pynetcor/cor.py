from typing import Any, Iterator

import numpy as np
from numpy import ndarray

from ._core import *

__all__ = [
    "corrcoef",
    "cortest",
    "cor_topk",
    "cor_topkdiff",
    "chunked_corrcoef",
    "chunked_cortest",
    "pvalue_student_t",
]


def corrcoef(
    x, y=None, method: str = "pearson", nan_action: str = "auto", threads: int = 1
) -> ndarray:
    """
    Calculate the correlation coefficient between each row of two arrays.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array.
    y : array_like, optional
        A 1-D or 2-D array. `y` has the same column length as `x`. If not provided, the correlation coefficient will be
        calculated between `x` and itself.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    nan_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.

    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    ndarray
        A 2D array representing the matrix of correlation coefficients.

    Examples
    --------
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> corrcoef(x, y)
    array([[1.        , 1.        ],
            [1.        , 1.        ]])

    """
    if nan_action == "auto":
        if method == "pearson" or method == "spearman":
            nan_action = "ignore"
        else:
            nan_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    return corrCoef(x, y, method, nan_action, threads)


def pvalue_student_t(x, df: int, approx: bool = True, threads: int = 1) -> ndarray:
    """
    Calculate the p-value for correlations(pearson or spearman) using the Student's t-distribution.

    Parameters
    ----------
    x : array_like
        Array of correlation coefficients.
    df : int
        The degrees of freedom.
    approx : bool, default True
        Whether to use the approximation method of p-value calculation.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    ndarray
        Array has the same shape as `x`.

    """
    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    return pvalueStudentT(x, df, approx, threads)


def cortest(
    x,
    y=None,
    method: str = "pearson",
    na_action="auto",
    approx_pvalue: bool = True,
    adjust_pvalue: bool = False,
    adjust_method: str = "BH",
    approx_adjust_pvalue: bool = False,
    threads: int = 1,
) -> ndarray:
    """
    Testing for correlation between each row of two arrays, using one of pearson, spearman or kendall.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array.
    y : array_like, optional
        A 1-D or 2-D array. `y` has the same column length as `x`. If not provided, the correlation coefficient will be
        calculated between `x` and itself.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    na_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.

    approx_pvalue : bool, default True
        Whether to use the approximation method of p-value calculation.
    adjust_pvalue : bool, default False
        Set this parameter to adjust p-value for multiple hypothesis testing.
    adjust_method : {'holm', 'hochberg', 'bonferroni', 'BH', 'BY'}, default 'BH'
        The method used to adjust p-value for multiple hypothesis testing.
    approx_adjust_pvalue : bool, default False
        Whether to use the approximation method of p-value adjustment.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    ndarray
        A 2D array with 4 columns: [index1, index2, r, p],
        or 5 columns if adjust_pvalue is True: [index1, index2, r, p, p_adjusted]

    Examples
    --------
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> cortest(x, y)
    array([[0.        , 0.        , 1.        , 2.2e-16.        ],
    """
    if na_action == "auto":
        if method == "pearson" or method == "spearman":
            na_action = "ignore"
        else:
            na_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    return corTest(
        x,
        y,
        method,
        na_action,
        approx_pvalue,
        adjust_pvalue,
        adjust_method,
        approx_adjust_pvalue,
        threads,
    )


class CorrcoefIterator:
    def __init__(self, iter):
        self._cpp_object = iter

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        """
        Get the next item from the iterator.

        Returns:
            Any: The next item in the iteration.

        Raises:
            StopIteration: When there are no more items to return.
        """
        try:
            return next(self._cpp_object)
        except Exception as e:
            # Assuming the C++ code raises an exception to indicate the end of iteration
            raise StopIteration from e


class CortestIterator:
    def __init__(self, iter):
        self._cpp_object = iter

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        """
        Get the next item from the iterator.

        Returns:
            Any: The next item in the iteration.

        Raises:
            StopIteration: When there are no more items to return.
        """
        try:
            return next(self._cpp_object)
        except Exception as e:
            # Assuming the C++ code raises an exception to indicate the end of iteration
            raise StopIteration from e


def chunked_corrcoef(
    x,
    y=None,
    method: str = "pearson",
    nan_action: str = "auto",
    chunk_size: int = 1024,
    threads: int = 1,
) -> CorrcoefIterator:
    """
    Iterating for correlation between each row of two arrays into chunks.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array.
    y : array_like, optional
        A 1-D or 2-D array. `y` has the same column length as `x`. If not provided, the correlation coefficient will be
        calculated between `x` and itself.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    nan_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.

    chunk_size : int, default 1024
        Rows number of correlation matrix to be calculated per chunk.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    CorrcoefIter
        Iterator over the computation of correlation matrix, utilizing a lazy evaluation approach that processes
        the data chunk by chunk.

    """
    if nan_action == "auto":
        if method == "pearson" or method == "spearman":
            nan_action = "ignore"
        else:
            nan_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    if chunk_size < 1:
        raise ValueError("The chunk size must be greater than 0.")

    return CorrcoefIterator(
        chunkedCorrcoef(x, y, method, nan_action, chunk_size, threads)
    )


def chunked_cortest(
    x,
    y=None,
    correlation_method: str = "pearson",
    na_action: str = "auto",
    approx_pvalue: bool = True,
    adjust_pvalue: bool = False,
    adjust_method: str = "BH",
    chunk_size: int = 1024,
    threads: int = 1,
) -> CortestIterator:
    """
    Iterating for testing the correlation between each row of two arrays into chunks, using one of pearson, spearman
    or kendall.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array.
    y : array_like, optional
        A 1-D or 2-D array. `y` has the same column length as `x`. If not provided, the correlation coefficient will be
        calculated between `x` and itself.
    correlation_method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    na_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.

    approx_pvalue : bool, default True
        Whether to use the approximation method of p-value calculation.
    adjust_pvalue : bool, default False
        Set this parameter to approximate adjusted P-value for multiple hypothesis testing.
        NOTE: `chunked` function only supports approximate adjusted P-value.
    adjust_method : {'holm', 'hochberg', 'bonferroni', 'BH', 'BY'}, default 'BH'
        The method used to adjust p-value for multiple hypothesis testing.
    chunk_size : int, default 1024
        `chunk_size` * columns number of `x` to be calculated per chunk.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    CortestIter
        Iterator over the computation of testing correlations, utilizing a lazy evaluation approach that processes
        the data chunk by chunk.

    """
    if na_action == "auto":
        if correlation_method == "pearson" or correlation_method == "spearman":
            na_action = "ignore"
        else:
            na_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    if chunk_size < 1:
        raise ValueError("The chunk size must be greater than 0.")

    return CortestIterator(
        chunkedCortest(
            x,
            y,
            correlation_method,
            na_action,
            approx_pvalue,
            adjust_pvalue,
            adjust_method,
            chunk_size,
            threads,
        )
    )


def cor_topk(
    x,
    y=None,
    method: str = "pearson",
    k: float = 0.01,
    na_action: str = "auto",
    correlation_mode: str = "both",
    compute_pvalue: bool = True,
    approx_pvalue: bool = True,
    chunk_size: int = 1024,
    threads: int = 1,
) -> ndarray:
    """
    Searching the global top k correlations between each row of two arrays, using one of pearson, spearman or kendall.

    Parameters
    ----------
    x : array_like
        A 2-D array.
    y : array_like, optional
        A 2-D array. `y` has the same column length as `x`. If not provided, the correlation coefficient will be
        calculated between `x` and itself.
    k : float, default 0.01
       The top k percentage of correlations, where k ranges from 0 to 1, or the top k number of correlations if k
       exceeds 1.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    na_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.
    correlation_mode : {'positive', 'negative', 'both'}, default 'both'
        The mode for comparing topk correlations.
    compute_pvalue : bool, default True
        Whether to calculate the p-value for each correlation.
    approx_pvalue : bool, default True
        Whether to use the approximation method of p-value calculation.
    chunk_size : int, default 1024
        `chunk_size` * columns number of `x` to be calculated per chunk.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    ndarray
        A 2D array with 4 columns: [index1, index2, r, p] or 3 columns: [index1, index2, r] if `compute_pvalue` is False.

    """
    if k == 0:
        return np.array([[]])
    elif k < 0:
        raise ValueError("The top k must be greater than 0.")

    if na_action == "auto":
        if method == "pearson" or method == "spearman":
            na_action = "ignore"
        else:
            na_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    if chunk_size < 1:
        raise ValueError("The chunk size must be greater than 0.")

    return corTopk(
        x,
        y,
        method,
        k,
        na_action,
        correlation_mode,
        compute_pvalue,
        approx_pvalue,
        chunk_size,
        threads,
    )


def cor_topkdiff(
    x1,
    y1,
    x2=None,
    y2=None,
    method: str = "pearson",
    k: float = 0.01,
    na_action: str = "auto",
    chunk_size: int = 1024,
    threads: int = 1,
) -> ndarray:
    """
    Searching the global top k differences in correlation between pairs of features across two states or timepoints,
    using one of pearson, spearman or kendall.

    Parameters
    ----------
    x1 : array_like
        A 2-D array.
    y1 : array_like
        A 2-D array. `y1` has the same feature number as `x1`.
    x2 : array_like
        A 2-D array. `x2` has the same column length as `x1`. If not provided, the correlation coefficient will be
        calculated between `x1` and itself.
    y2 : array_like
        A 2-D array. `y2` has the same column length as `y1`. If not provided, the correlation coefficient will be
        calculated between `x2` and itself.
    k : float, default 0.01
       The top k percentage of correlations, where k ranges from 0 to 1, or the top k number of correlations if k
       exceeds 1.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The method used to calculate the correlation coefficient.
    na_action : {'auto', 'ignore', 'fillMean', 'fillMedian'}, default 'auto'
        The action to take when encountering NaN values.
        Pearson and Spearman recommend using the ignore method and Kendall recommends using the fillMedian method.
        The ignore method cannot be used for Kendall.

        * 'ignore': the calculation ignores pairs of elements that contain NaN.
        * 'fillMean': fills the NaN values in each row with the mean of non-NaN values.
        * 'fillMedian': fills the NaN values in each row with the median of non-NaN values.
    chunk_size : int, default 1024
        `chunk_size` * columns number of `x` to be calculated per chunk.
    threads : int, default 1
        The number of threads to use.

    Returns
    -------
    ndarray
        A 2D array with 5 columns: [index1, index2, diffCor, cor1, cor2].

    """
    if k == 0:
        return np.array([[]])
    elif k < 0:
        raise ValueError("The top k must be greater than 0.")

    if na_action == "auto":
        if method == "pearson" or method == "spearman":
            na_action = "ignore"
        else:
            na_action = "fillMedian"

    if threads < 1:
        raise ValueError("The number of threads must be greater than 0.")

    if chunk_size < 1:
        raise ValueError("The chunk size must be greater than 0.")

    return corTopkDiff(
        x1,
        y1,
        x2,
        y2,
        method,
        k,
        na_action,
        chunk_size,
        threads,
    )
