import numpy as np
from ..direct import least_squares, least_squares_cost

import pytest
from numpy.testing import assert_allclose


def data(N=15, M=10, K=None, sigma=0.1, complex=False, rseed=53489):
    rand = np.random.RandomState(rseed)
    A = rand.randn(N, M)
    if K is None:
        x = rand.randn(M)
    else:
        x = rand.randn(M, K)
    if complex:
        A = A + 1j * rand.randn(*A.shape)
        x = x + 1j * rand.rand(*x.shape)
    y = np.dot(A, x)
    y += sigma * rand.randn(*y.shape)
    if complex:
        y += 1j * sigma * rand.randn(*y.shape)
    return A, x, y


@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('sigma', [0, 0.001, 0.1])
@pytest.mark.parametrize('K', [None, 2])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [15])
def test_direct_unregularized(N, M, K, sigma, complex):
    A, x, y = data(N, M, K, complex=complex, sigma=sigma)
    xfit = least_squares(A, y)
    assert_allclose(x, xfit, atol=5 * sigma)


@pytest.mark.parametrize('gamma_L2', [0.1, 1.0])
@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('sigma', [1E-1, 1E-2, 1E-3])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [15])
def test_direct_cost(N, M, sigma, complex, gamma_L2):
    rand = np.random.RandomState(543543)

    A, x, y = data(N, M, complex=complex, sigma=sigma)
    xfit = least_squares(A, y, gamma_L2=gamma_L2)
    cost_0 = least_squares_cost(A, xfit, y, gamma_L2)

    # Make sure we've actually found a minimum
    # By sampling 10 points near the fit value
    x_perturbed = xfit[:, None] + 0.01 * rand.randn(xfit.shape[0], 10)
    cost_perturbed = least_squares_cost(A, x_perturbed, y[:, None], gamma_L2)

    assert np.all(cost_0 <= cost_perturbed)
