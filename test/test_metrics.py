import numpy as np
import pytest

from eski import metrics


@pytest.mark.parametrize(
    "p1,p2,distance",
    [
        ([0, 0, 0], [0, 0, 0], 0),
        ([1, 0, 0], [0, 0, 0], 1),
        ([0, 0.5, 0], [0, -0.25, 0], 0.75)
    ]
)
def test_euclidean_distance(p1, p2, distance):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    assert distance == metrics.euclidean_distance(p1, p2)
