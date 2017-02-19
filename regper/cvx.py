"""Deconvolution using CVXPY"""
import numpy as np
from scipy.sparse import issparse
import cvxpy as cvx


def least_squares(A, y, gamma_L2=0, gamma_L1=0):
    """Iterative solution to a least squares problem

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
