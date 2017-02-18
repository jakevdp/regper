import numpy as np
from .. import direct, iterative

import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator


OPTYPES = {'array': np.asarray,
           'sparse': csr_matrix,
           'operator': aslinearoperator}


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


@pytest.mark.parametrize('optype', ['array', 'sparse', 'operator'])
@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('sigma', [0, 0.001, 0.1])
@pytest.mark.parametrize('K', [None, 2])
@pytest.mark.parametrize('M', [4, 9])
@pytest.mark.parametrize('N', [11])
def test_iterative_unregularized(N, M, K, sigma, complex, optype):
    A, x, y = data(N, M, complex=complex, sigma=sigma)
    A = OPTYPES[optype](A)
    xfit = iterative.least_squares(A, y)
    assert_allclose(x, xfit, atol=max(1E-5, 5 * sigma))


@pytest.mark.parametrize('conv_method', ['fft', 'matrix', 'direct'])
@pytest.mark.parametrize('complex', [True, False])
@pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
@pytest.mark.parametrize('sigma', [0])
@pytest.mark.parametrize('M', [10, 15])
@pytest.mark.parametrize('N', [10, 15])
def test_iterative_deconvolution(N, M, sigma, mode, complex, conv_method):
    w, x, y = conv_data(N, M, mode=mode, complex=complex, sigma=sigma)
    if len(y) < len(x):
        with pytest.warns(UserWarning) as warning:
            xfit = iterative.deconvolution(w, y, Nx=len(x), mode=mode,
                                           conv_method=conv_method)
        assert len(warning) == 1
        assert warning[0].message.args[0].startswith("Ill-posed deconvolution")
    else:
        xfit = iterative.deconvolution(w, y, Nx=len(x), mode=mode,
                                       conv_method=conv_method)
        assert_allclose(x, xfit, atol=1E-3)
