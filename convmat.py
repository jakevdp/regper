import numpy as np
from scipy.linalg import toeplitz


def convolution_matrix(x, N=None, mode='full'):
    """Compute the Convolution Matrix

    This function computes a convolution matrix to perform a
    computation equivalent to numpy.convolve(x, y)

    Parameters
    ----------
    x :

    N :

    mode :


    Returns
    -------
    C :

    Example
    -------
    >>> x = np.random.rand(10)
    >>> y = np.random.rand(20)
    >>> xy = np.convolve(x, y, mode='full')
    >>> Cfull = convolution_matrix(x, len(y), mode='full')
    >>> np.allclose(xy, np.dot(Cfull, y))
    True
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x should be 1-dimensional")

    M = len(x)
    N = M if N is None else N

    if mode == 'full':
        Nout = M + N - 1
        offset = 0
    elif mode == 'valid':
        Nout = max(M, N) - min(M, N) + 1
        offset = min(M, N) - 1
    elif mode == 'same':
        Nout = max(N, M)
        offset = (min(N, M) - 1) // 2
    else:
        raise ValueError("mode='{0}' not recognized".format(mode))

    xpad = np.hstack([x, np.zeros(Nout)])
    n = np.arange(Nout)[:, np.newaxis]
    m = np.arange(N)
    return xpad[n - m + offset]


#------------------------------------------------------------
# Tests:
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize('M', [10, 15, 20, 25])
@pytest.mark.parametrize('N', [10, 15, 20, 25])
@pytest.mark.parametrize('dtype', ['float', 'complex'])
@pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
def test_convolution_matrix(M, N, dtype, mode):
    rand = np.random.RandomState(42)
    x = rand.rand(M)
    y = rand.rand(N)

    if dtype == 'complex':
        x = x * np.exp(2j * np.pi * rand.rand(M))
        y = y * np.exp(2j * np.pi * rand.rand(N))

    result1 = np.dot(convolution_matrix(x, len(y), mode), y)
    result2 = np.convolve(x, y, mode)
    assert_allclose(result1, result2)
