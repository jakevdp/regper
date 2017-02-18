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


# TODO: test deconvolutions
