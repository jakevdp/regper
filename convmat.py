import numpy as np


def convolution_matrix(x, N=None, mode='full'):
    """Compute the Convolution Matrix

    This function computes a convolution matrix that encodes
    the computation equivalent to ``numpy.convolve(x, y, mode)``

    Parameters
    ----------
    x : array_like
        One-dimensional input array
    N : integer (optional)
        Size of the array to be convolved. Default is len(x)
    mode : {'full', 'valid', 'same'}, optional
        The type of convolution to perform.
        See ``np.convolve`` for details.

    Returns
    -------
    C : ndarray
        Matrix operator encoding the convolution. The matrix is of shape
        Nout x N, where Nout depends on ``mode`` and the size of ``x``.

    Example
    -------
    >>> x = np.random.rand(10)
    >>> y = np.random.rand(20)
    >>> xy = np.convolve(x, y, mode='full')
    >>> C = convolution_matrix(x, len(y), mode='full')
    >>> np.allclose(xy, np.dot(C, y))
    True

    See Also
    --------
    numpy.convolve : direct convolution operation
    scipy.signal.fftconvolve : direct convolution via the
                               fast Fourier transform
    scipy.linalg.toeplitz : construct the Toeplitz matrix
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
