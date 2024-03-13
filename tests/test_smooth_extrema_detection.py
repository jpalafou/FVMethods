import numpy as np
import pytest
from finite_volume.sed import detect_smooth_extrema

num_trials = 5
h = 0.05


@pytest.mark.parametrize("trial", range(num_trials))
def test_reflection_invariance(trial):
    data = np.random.rand(10)
    assert np.all(
        detect_smooth_extrema(data, h=h, axis=0)
        == detect_smooth_extrema(-data, h=h, axis=0)
    )
