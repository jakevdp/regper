import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

import pytest
from numpy.testing import assert_allclose

from ..regper import L1_least_squares, L2_least_squares


LS_DICT = {'L1': L1_least_squares,
           'L2': L2_least_squares}

MAT_TYPE = {'operator': aslinearoperator,
            'sparse': csr_matrix,
            'array': np.asarray}


@pytest.fixture
def data(N=10, rseed=54325):
    rand = np.random.RandomState(rseed)
    A = rand.rand(N, N)
    x = np.zeros(N)
    x[rand.randint(0, N, 2)] = 1
    y = np.dot(A, x) + 1E-4 * rand.randn(N)
    return A, x, y


@pytest.fixture
def complex_data(N=10, rseed=54325):
    rand = np.random.RandomState(rseed)
    A = rand.rand(N, N) + 1j * rand.rand(N, N)
    x = np.zeros(N, dtype=complex)
    x[rand.randint(0, N, 2)] = 1
    x[rand.randint(0, N, 2)] = 1j
    y = np.dot(A, x) + 1E-4 * rand.randn(N) + 1E-4 * 1j * rand.randn(N)
    return A, x, y


@pytest.mark.parametrize('regularization', ['L1', 'L2'])
def test_unregularized(data, regularization):
    least_squares = LS_DICT[regularization]
    A, x, y = data
    x_ls = least_squares(A, y, lam=1E-8)
    assert_allclose(x_ls, x, atol=1E-2)


@pytest.mark.parametrize('regularization', ['L1', 'L2'])
@pytest.mark.parametrize('mat_type', ['operator', 'sparse'])
def test_input_types(data, regularization, mat_type):
    A, x, y = data
    least_squares = LS_DICT[regularization]
    Amat, AconjTmat = map(MAT_TYPE[mat_type], (A, A.conj().T))
    x_ls = least_squares(A, y, AconjT=A.T, lam=0.1)
    x_ls_mat = least_squares(Amat, y, AconjT=AconjTmat, lam=0.1)
    assert_allclose(x_ls, x_ls_mat, rtol=1E-3)
