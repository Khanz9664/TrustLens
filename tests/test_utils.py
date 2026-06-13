import numpy as np
import pytest

from trustlens.utils import (
    check_consistent_length,
    describe_array,
    flatten_dict,
    safe_divide,
    validate_array,
)


def test_validate_array_none():
    with pytest.raises(ValueError, match="'test' cannot be None"):
        validate_array(None, name="test")


def test_validate_array_empty():
    with pytest.raises(ValueError, match="'test' cannot be empty"):
        validate_array([], name="test")


def test_validate_array_success():
    arr = validate_array([1, 2, 3], name="test", ndim=1)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)


def test_validate_array_ndim_mismatch():
    with pytest.raises(ValueError, match="Expected 'test' to have 2 dimensions, got 1"):
        validate_array([1, 2, 3], name="test", ndim=2)


def test_check_consistent_length_empty():
    with pytest.raises(ValueError, match="At least one array must be provided"):
        check_consistent_length()


def test_check_consistent_length_none():
    with pytest.raises(ValueError, match="Array at index 1 is None"):
        check_consistent_length(np.array([1]), None)


def test_check_consistent_length_mismatch():
    with pytest.raises(ValueError, match="Inconsistent array lengths"):
        check_consistent_length(np.array([1, 2]), np.array([1]))


def test_safe_divide_type_error():
    with pytest.raises(TypeError, match="must be numeric"):
        safe_divide("1", 2)


def test_safe_divide_success():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(1, 0) == 0.0
    assert safe_divide(np.float64(10.0), 2) == 5.0


def test_flatten_dict_type_error():
    with pytest.raises(TypeError, match="must be a dictionary"):
        flatten_dict([1, 2, 3])


def test_flatten_dict_success():
    d = {"a": {"b": 1}, "c": 2}
    flat = flatten_dict(d)
    assert flat == {"a.b": 1, "c": 2}


def test_describe_array_empty():
    msg = describe_array(np.array([]), name="empty_test")
    assert "empty array" in msg
    assert "empty_test" in msg


def test_describe_array_success():
    msg = describe_array(np.array([1, 2, 3]))
    assert "shape=(3,)" in msg
    assert "min=1" in msg
    assert "max=3" in msg
