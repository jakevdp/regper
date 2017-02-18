"""Direct deconvolution solvers"""

import warnings

import numpy as np
from .utils import convolution_matrix


def least_squares(A, y, gamma_L2=0.0):
    """Direct solution of a least squares problem

    Returns argmin_x ||A*x - y||^2 + gamma_L2 ||x||^2

    Parameters
    ----------
    A : array_like
        [N x M] projection matrix
    y : array_like
        length-M vector or [M x K] matrix
    gamma_L2 : float (optional)
        L2 regularization strength. Default=0

    Returns
    -------
    x : ndarray
        length-N vector or [N x K] matrix that minimizes the
        cost function.
    """
    A = np.asarray(A)
    y = np.asarray(y)
    N, M = A.shape

    AH = A.conj().transpose()
    I = np.eye(M)
    return np.linalg.solve(np.dot(AH, A) + gamma_L2 * I,
                           np.dot(AH, y))


def deconvolution(w, y, gamma_L2=0.0, Nx=None, mode='full'):
    """Direct deconvolution using a closed-form least squares

    Returns argmin_x ||conv(w, x) - y||^2 + gamma_L2 ||x||^2

    Parameters
    ----------
    w : array_like
        length-N array representing the convolution
    y : array_like
        length-M array or [M x K] matrix. Note that M must match the
        output of of np.convolve(w, x, mode).
    gamma_L2 : float, optional
        L2 regularization strength. Default=0
    Nx : int, optional
        The number of elements in the x array. Default = N
    mode : {'full', 'valid', 'same'}, optional
        The convolution mode (see ``np.convolve`` dostring for details)

    Returns
    -------
    x : ndarray
        the length-Nx or [Nx x K] matrix representing the deconvolution
    """
    if len(y) < Nx and gamma_L2 == 0:
        warnings.warn("Ill-posed deconvolution: len(y)={0}, len(x)={1}. "
                      "Try adding regularization or using a different "
                      "mode of convolution".format(len(y), Nx))
    C = convolution_matrix(w, Nx, mode=mode)
    return least_squares(C, y, gamma_L2)
