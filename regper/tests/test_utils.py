import pytest
from numpy.testing import assert_allclose
import numpy as np
from ..utils import convolution_matrix


@pytest.mark.parametrize('M', [10, 11, 12, 13])
@pytest.mark.parametrize('N', [10, 11, 12, 13])
@pytest.mark.parametrize('dtype', ['float', 'complex'])
@pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
def test_convolution_matrix(M, N, dtype, mode):
    rand = np.random.RandomState(42)
    x = rand.rand(M)
    y = rand.rand(N)

    if dtype == 'complex':
        x = x + 1j * rand.randn(M)
        y = y + 1j * rand.randn(N)

    result1 = np.dot(convolution_matrix(x, len(y), mode), y)
    result2 = np.convolve(x, y, mode)
    assert_allclose(result1, result2)
