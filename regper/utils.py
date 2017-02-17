from __future__ import division
import numpy as np


def convolution_output_size(N, M, mode):
    """Return the size of a convolution output"""
    if mode == 'full':
        return M + N - 1
    elif mode == 'valid':
        return max(M, N) - min(M, N) + 1
    elif mode == 'same':
        return max(N, M)
    else:
        raise ValueError("mode='{0}' not recognized".format(mode))


def convolution_offset(N, M, mode):
    """Return the size of the offset in the Toeplitz matrix"""
    if mode == 'full':
        return 0
    elif mode == 'valid':
        return min(M, N) - 1
    elif mode == 'same':
        return (min(N, M) - 1) // 2
    else:
        raise ValueError("mode='{0}' not recognized".format(mode))


def convolution_matrix(x, N=None, mode='full'):
    """Compute the Convolution Matrix
    This function computes a convolution matrix that encodes
    the computation equivalent to ``numpy.convolve(x, y, mode)``

    Parameters
    ----------
    x : array_like
        One-dimensional input array
    N : integer (optional)
        Size of the array to be convolved. Default is len(x).
    mode : {'full', 'valid', 'same'}, optional
        The type of convolution to perform. Default is 'full'.
        See ``np.convolve`` documentation for details.

    Returns
    -------
    C : ndarray
        Matrix operator encoding the convolution. The matrix is of shape
        [Nout x N], where Nout depends on ``mode`` and the size of ``x``.

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
    Nout = convolution_output_size(M, N, mode)
    offset = convolution_offset(M, N, mode)

    x_padded = np.hstack([x, np.zeros(Nout)])
    n = np.arange(Nout)[:, np.newaxis]
    m = np.arange(N)
    return x_padded[offset + n - m]


def least_squares_cost(A, x, y, gamma_L2=0, gamma_L1=0):
    """Cost function for least squares

    returns ||A*x - y||^2 + gamma_L2 ||x||_2^2 + gamma_L1 ||x||_1

    Parameters
    ----------
    A : array_like
        [N x M] projection matrix
    x : array_like
        length-M vector or [M x K] matrix
    y : array_like
        length-N vector or [N x K] matrix
    gamma_L2 : float (optional)
        L2 regularization strength. Default=0
    gamma_L1 : float (optional)
        L1 regularization strength. Default=0

    Returns
    -------
    cost : float or length-K array
        The value of the (regularized) cost function
    """
    A, x, y = map(np.asarray, (A, x, y))
    return ((abs(np.dot(A, x) - y) ** 2).sum(0)
            + gamma_L2 * (abs(x) ** 2).sum(0)
            + gamma_L1 * abs(x).sum(0))
