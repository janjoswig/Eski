import numpy as np
import pytest

from eski import metrics


@pytest.mark.parametrize(
    "p1,p2,distance",
    [
        ([0], [2], 2),
        ([1., 1.], [2., 1.], 1.),
        ([0, 0, 0], [0, 0, 0], 0),
        ([1, 0, 0], [0, 0, 0], 1),
        ([0, 0.5, 0], [0, -0.25, 0], 0.75)
    ]
)
def test_euclidean_distance(p1, p2, distance):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    assert distance == metrics.euclidean_distance(p1, p2, norm=True)
    assert np.allclose(p2 - p1, metrics.euclidean_distance(p1, p2))


def test_random_uniform():
    numbers = np.array([metrics.random_uniform() for _ in range(1000)])
    assert np.all((0 <= numbers) & (numbers <= 1))


def test_random_gaussian():
    numbers = np.array([metrics.random_gaussian() for _ in range(10000)])
    nbins = 7
    nleft_bins = nbins // 2
    h, _ = np.histogram(numbers, range=(-1.5, 1.5), bins=nbins)
    np.testing.assert_allclose(h[0:nleft_bins], h[-1:nleft_bins:-1], rtol=0.1)

    more_numbers = np.array([metrics.random_gaussian() for _ in range(10000)])
    assert not np.array_equal(numbers, more_numbers)


@pytest.mark.parametrize(
    "a,expected",
    [
        ([1, 2, 3, 4, 5], 5),
        ([-1.3, 0.4, 1.76], 1.76)
    ]
)
def test_get_max(a, expected):
    assert metrics.get_max(np.array(a, dtype=np.float64)) == expected
