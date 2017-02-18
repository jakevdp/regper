"""Iterative deconvolution solvers"""

import warnings

import numpy as np
from scipy import optimize, signal
from scipy.sparse.linalg import LinearOperator


from .utils import (convolution_matrix, convolution_output_size,
                    least_squares_cost, asanyoperator)


def least_squares(A, y, gamma_L2=0, gamma_L1=0, method=None, method_kwds=None):
    """Iterative solution to a least squares problem

    Returns argmin_x ||A*x - y||^2 + gamma_L2 ||x||^2 + gamma_L1 ||x||_1

    Parameters
    ----------
    A : array_like, sparse, or linear operator
        [N x M] projection matrix or operator
    y : array_like
        length-M vector
    gamma_L2 : float (optional)
        L2 regularization strength. Default=0
    gamma_L1 : float (optional)
        L1 regularization strength. Default=0
    method : string (optional)
        method to use (passed to ``scipy.optimize.minimize``)
    method_kwds : dict (optional)
        additional keywords passed to ``scipy.optimize.minimize``

    Returns
    -------
    x : ndarray
        length-N vector that minimizes the cost function.
    """
    A = asanyoperator(A)
    y = np.asarray(y)
    M, N = A.shape

    cplx = np.issubdtype(A.dtype, complex) or np.issubdtype(y.dtype, complex)

    if cplx:
        make_x = lambda x: x[:N] + 1j * x[N:]
        x0 = np.zeros(2 * N, dtype=float)
    else:
        make_x = lambda x: x
        x0 = np.zeros(N, dtype=float)
    func = lambda x: least_squares_cost(A, make_x(x), y,
                                        gamma_L2=gamma_L2,
                                        gamma_L1=gamma_L1)
    method_kwds = method_kwds or {}
    res = optimize.minimize(func, x0=x0, method=method, **method_kwds)

    if not res.success:
        warnings.warn(res.message)

    return make_x(res.x)


def deconvolution(w, y, gamma_L2=0, gamma_L1=0, Nx=None,
                  conv_method='matrix', mode='full',
                  method=None, method_kwds=None):
    """Iterative deconvolution using least squares

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
    gamma_L1 : float (optional)
        L1 regularization strength. Default=0
    Nx : int, optional
        The number of elements in the x array. Default = N
    conv_method : {'fft', 'direct', 'matrix'}, optional
        Method to use for convolution. Default='fft'
    mode : {'full', 'valid', 'same'}, optional
        The convolution mode (see ``np.convolve`` dostring for details)
    method : string (optional)
        method to use (passed to ``scipy.optimize.minimize``)
    method_kwds : dict (optional)
        additional keywords passed to ``scipy.optimize.minimize``

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
    elif conv_method == 'direct':
        C = LinearOperator((Ny, Nx), dtype=w.dtype,
                           matvec=lambda x: np.convolve(w, x, mode=mode))
    elif conv_method == 'fft':
        # Note: numpy.convolve does this switch internally
        #       scipy fftconvolve raises an error in some cases
        #       when the first argument is shorter than the second
        if len(w) < Nx:
            matvec = lambda x: signal.fftconvolve(x, w, mode=mode)
        else:
            matvec = lambda x: signal.fftconvolve(w, x, mode=mode)
        C = LinearOperator((Ny, Nx), dtype=w.dtype, matvec=matvec)
    else:
        raise ValueError("conv_method must be in {'matrix', 'direct', 'fft'}")

    return least_squares(C, y, gamma_L2=gamma_L2, gamma_L1=gamma_L1,
                         method=method, method_kwds=method_kwds)
