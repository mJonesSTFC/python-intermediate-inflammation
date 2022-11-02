"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test,expected",
    [
        ([[0, 0],[0, 0],[0, 0]], [0,0]),
        ([[1, 2],[3, 4],[5, 6]], [3,4]),
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test,expected",
    [
        ([[0, 0],[0, 0],[0, 0]], [0,0]),
        ([[1, 2],[3, 4],[5, 6]], [5,6]),
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for an array of zeros"""
    from inflammation.models import daily_max
    #use npt to compare array
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test,expected",
    [
        ([[0, 0],[0, 0],[0, 0]], [0,0]),
        ([[1, 2],[3, 4],[5, 6]], [1,2]),
    ]
)
def test_daily_min(test, expected):
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Test that type errors are handled correctly"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['abd',';laskdjf'], ['asd', 'khjlj']])

@pytest.mark.parametrize(
    "test,expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError)
    ]
)
def test_patient_normalise(test,expected, expect_raises):
    """test that normalisation fucntion works"""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)