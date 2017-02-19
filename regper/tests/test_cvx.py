import numpy as np
from ..utils import least_squares_cost
from ..cvx import least_squares, deconvolution

import pytest
from numpy.testing import assert_allclose

from scipy.sparse import csr_matrix


OPTYPES = {'array': np.asarray,
           'sparse': csr_matrix}


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


def conv_data(N=15, M=10, mode='full', sigma=0.1, complex=False, rseed=53489):
    rand = np.random.RandomState(rseed)
    w = rand.rand(N)
    x = rand.randn(M)
    if complex:
        w = w + 1j * rand.randn(N)
        x = x + 1j * rand.rand(M)
    y = np.convolve(w, x, mode=mode)
    y += sigma * rand.randn(*y.shape)
    if complex:
        y += 1j * sigma * rand.randn(*y.shape)
    return w, x, y


@pytest.mark.parametrize('optype', ['array', 'sparse'])
@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('sigma', [0, 0.001, 0.1])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [15])
def test_cvx_unregularized(N, M, sigma, complex, optype):
    A, x, y = data(N, M, complex=complex, sigma=sigma)
    A = OPTYPES[optype](A)
    xfit = least_squares(A, y)
    assert_allclose(x, xfit, atol=5 * sigma)


@pytest.mark.parametrize('gamma_L2', [0, 1.0])
@pytest.mark.parametrize('gamma_L1', [0, 1.0])
@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('sigma', [0, 0.001, 0.1])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [15])
def test_cvx_cost(N, M, sigma, complex, gamma_L1, gamma_L2):
    A, x, y = data(N, M, complex=complex, sigma=sigma)
    xfit = least_squares(A, y, gamma_L1=gamma_L1, gamma_L2=gamma_L2)
    cost_0 = least_squares_cost(A, xfit, y, gamma_L1=gamma_L1,
                                gamma_L2=gamma_L2)

    # Make sure we've actually found a minimum
    # By sampling 10 points near the fit value
    rand = np.random.RandomState(543543)
    x_perturbed = xfit[:, None] + 0.01 * rand.randn(xfit.shape[0], 10)
    cost_perturbed = least_squares_cost(A, x_perturbed, y[:, None],
                                        gamma_L1=gamma_L1,
                                        gamma_L2=gamma_L2)

    assert np.all(cost_0 <= cost_perturbed)


@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('conv_method', ['direct', 'matrix'])
@pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
@pytest.mark.parametrize('sigma', [0])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [10, 15])
def test_cvx_deconvolution(N, M, sigma, mode, conv_method, complex):
    w, x, y = conv_data(N, M, mode=mode, complex=complex, sigma=sigma)

    args = (w, y)
    kwargs = dict(Nx=len(x), mode=mode, conv_method=conv_method)

    if conv_method == 'direct' and mode != 'full':
        with pytest.raises(ValueError) as e_info:
            xfit = deconvolution(*args, **kwargs)
        assert str(e_info.value).startswith("Only mode='full'")
    elif len(y) < len(x):
        xfit = deconvolution(*args, **kwargs)
        #with pytest.warns(UserWarning) as warning:
        #    xfit = deconvolution(*args, **kwargs)
        #assert len(warning) == 1
        #assert warning[0].message.args[0].startswith("Ill-posed deconvolution")
    else:
        xfit = deconvolution(*args, **kwargs)
        assert_allclose(x, xfit)
