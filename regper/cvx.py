"""Deconvolution using CVXPY"""
import warnings

import numpy as np
from scipy.sparse import issparse
import cvxpy as cvx

from .utils import convolution_output_size, convolution_matrix


def least_squares(A, y, gamma_L2=0, gamma_L1=0):
    """Iterative solution to a least squares problem using CVXPY

    Returns argmin_x ||A*x - y||^2 + gamma_L2 ||x||^2 + gamma_L1 ||x||_1

    Parameters
    ----------
    A : array_like or sparse
        [N x M] projection matrix or operator
    y : array_like
        length-M vector
    gamma_L2 : float (optional)
        L2 regularization strength. Default=0
    gamma_L1 : float (optional)
        L1 regularization strength. Default=0

    Returns
    -------
    x : ndarray
        length-N vector that minimizes the cost function.
    """
    if not issparse(A):
        A = np.asarray(A)
    y = np.asarray(y)

    N, M = A.shape

    cplx = np.issubdtype(A.dtype, complex) or np.issubdtype(y.dtype, complex)

    # objective is ||Ax - y||^2 + gamma_L1 |x|_1 + gamma_L2 ||x||^2

    if cplx:
        # CVXPY does not handle complex, so we split the problem
        Ar, Ai = A.real, A.imag
        yr, yi = y.real, y.imag

        xr = cvx.Variable(M)
        xi = cvx.Variable(M)

        error = (cvx.sum_squares(Ar*xr - Ai*xi - yr) +
                 cvx.sum_squares(Ai*xr + Ar*xi - yi))
        u = cvx.norm(cvx.hstack(xr, xi), 2, axis=1)
    else:
        x = cvx.Variable(A.shape[1])
        error = cvx.sum_squares(A*x - y)
        u = x

    cost = error

    if gamma_L1 != 0:
        gamma_L1 = cvx.Parameter(value=gamma_L1, sign="positive")
        cost = cost + gamma_L1 * cvx.norm(u, 1)
    if gamma_L2 != 0:
        gamma_L2 = cvx.Parameter(value=gamma_L2, sign='positive')
        cost = cost + gamma_L2 * cvx.sum_squares(u)

    objective = cvx.Minimize(cost)
    prob = cvx.Problem(objective)
    prob.solve()
    print("Problem Status: {0}".format(prob.status))

    if cplx:
        result = np.array(xr.value).ravel() + 1j * np.array(xi.value).ravel()
    else:
        result = np.asarray(x.value).ravel()

    return result


def _direct_deconvolution(w, y, Nx, gamma_L2=0, gamma_L1=0):
    w = np.asarray(w)
    y = np.asarray(y)
    cplx = np.issubdtype(w.dtype, complex) or np.issubdtype(y.dtype, complex)

    if cplx:
        raise NotImplementedError('complex inputs')
    else:
        x = cvx.Variable(Nx)
        error = cvx.sum_squares(cvx.conv(w, x) - y)
        cost = error
        if gamma_L1 != 0:
            gamma_L1 = cvx.Parameter(value=gamma_L1, sign="positive")
            cost = cost + gamma_L1 * cvx.norm(x, 1)
        if gamma_L2 != 0:
            gamma_L2 = cvx.Parameter(value=gamma_L2, sign="positive")
            cost = cost + gamma_L2 * cvx.sum_squares(x)
        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective)

    prob.solve()
    print("Problem Status: {0}".format(prob.status))

    return np.asarray(x.value).ravel()


def deconvolution(w, y, gamma_L2=0, gamma_L1=0, Nx=None,
                  conv_method='direct', mode='full'):
    """Iterative deconvolution using least squares via CVXPY

    Returns argmin_x ||conv(w, x) - y||^2 + gamma_L2 ||x||^2 + gamma_L1 ||x||_1

    Parameters
    ----------
    w : array_like
        length-N array representing the convolution
    y : array_like
        length-M array or [M x K] matrix. Note that M must match the
        output of of np.convolve(w, x, mode).
    gamma_L2 : float, optional
        L2 regularization strength. Default=0
    gamma_L1 : float (optional)
        L1 regularization strength. Default=0
    Nx : int, optional
        The number of elements in the x array. Default = N
    conv_method : {'direct', 'matrix'}, optional
        Method to use for convolution. Default='fft'
    mode : {'full', 'valid', 'same'}, optional
        The convolution mode (see ``np.convolve`` dostring for details)

    Returns
    -------
    x : ndarray
        the length-Nx or [Nx x K] matrix representing the deconvolution
    """
    w = np.asarray(w)
    Nx = len(w) if Nx is None else Nx
    Ny = convolution_output_size(len(w), Nx, mode=mode)

    if len(y) != Ny:
        raise ValueError("Array sizes do not match convolution mode")

    if Ny < Nx and gamma_L2 == 0 and gamma_L1 == 0:
        warnings.warn("Ill-posed deconvolution: len(y)={0}, len(x)={1}. "
                      "Try adding regularization or using a different "
                      "mode of convolution".format(Ny, Nx))

    if conv_method == 'matrix':
        C = convolution_matrix(w, Nx, mode=mode)
        return least_squares(C, y, gamma_L2=gamma_L2, gamma_L1=gamma_L1)
    elif conv_method == 'direct':
        if mode != 'full':
            raise ValueError("Only mode='full' supported for direct method "
                             "of CVX deconvolution.")
        return _direct_deconvolution(w, y, Nx,
                                     gamma_L2=gamma_L2,
                                     gamma_L1=gamma_L1)
    else:
        raise ValueError("conv_method must be in {'matrix', 'direct', 'fft'}")
