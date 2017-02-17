"""Direct deconvolution solvers"""

import numpy as np


def least_squares_cost(A, x, y, gamma_L2=0):
    """Cost function for least squares

    returns ||A*x - y||^2 + gamma_L2 ||x||^2

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

    Returns
    -------
    cost : float or length-K array
        The value of the (regularized) cost function
    """
    A, x, y = map(np.asarray, (A, x, y))
    return ((abs(np.dot(A, x) - y) ** 2).sum(0)
            + gamma_L2 * (abs(x) ** 2).sum(0))


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
