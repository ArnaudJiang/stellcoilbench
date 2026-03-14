"""Tests for NumpyJSONEncoder."""

import json

import numpy as np
import pytest

from stellcoilbench.cli import NumpyJSONEncoder


def test_numpy_json_encoder_handles_numpy_types():
    payload = {
        "i": np.int64(3),
        "f": np.float64(1.25),
        "b": np.bool_(True),
        "a": np.array([1, 2, 3]),
    }
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["i"] == 3
    assert loaded["f"] == 1.25
    assert loaded["b"] is True
    assert loaded["a"] == [1, 2, 3]


def test_numpy_json_encoder_handles_array_protocol():
    class _ArrayLike:
        def __array__(self):
            return np.array([4, 5, 6])

    payload = {"arr": _ArrayLike()}
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["arr"] == [4, 5, 6]


def test_numpy_json_encoder_array_protocol_error():
    class _BadArray:
        def __array__(self):
            raise TypeError("bad array")

    with pytest.raises(TypeError):
        json.dumps({"arr": _BadArray()}, cls=NumpyJSONEncoder)


def test_numpy_json_encoder_simsopt_object():
    """NumpyJSONEncoder converts simsopt objects to string."""

    class _FakeSimsoptObj:
        __module__ = "simsopt.geo.surface"

        def __str__(self):
            return "FakeSurface()"

    payload = {"obj": _FakeSimsoptObj()}
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["obj"] == "FakeSurface()"


def test_numpy_json_encoder_unknown_type_raises():
    """NumpyJSONEncoder raises TypeError for unrecognized types."""

    class _Unknown:
        pass

    with pytest.raises(TypeError):
        json.dumps({"obj": _Unknown()}, cls=NumpyJSONEncoder)


class TestNumpyJSONEncoderComprehensive:
    """Granular tests for NumpyJSONEncoder (merged from test_cli_comprehensive)."""

    @pytest.mark.parametrize(
        "np_val,expected",
        [
            (np.int32(42), 42),
            (np.int64(100), 100),
            (np.float32(3.14), pytest.approx(3.14, rel=1e-6)),
            (np.float64(2.71), 2.71),
        ],
        ids=["int32", "int64", "float32", "float64"],
    )
    def test_encode_numpy_numeric(self, np_val, expected):
        """NumpyJSONEncoder encodes numpy int/float to Python scalar."""
        encoder = NumpyJSONEncoder()
        assert encoder.default(np_val) == expected

    def test_encode_numpy_array_2d(self):
        encoder = NumpyJSONEncoder()
        arr = np.array([[1, 2], [3, 4]])
        result = encoder.default(arr)
        assert result == [[1, 2], [3, 4]]

    def test_encode_array_like(self):
        class ArrayLike:
            def __array__(self):
                return np.array([1, 2, 3])

        encoder = NumpyJSONEncoder()
        result = encoder.default(ArrayLike())
        assert result == [1, 2, 3]

    def test_encode_fallback_raises(self):
        encoder = NumpyJSONEncoder()
        with pytest.raises(TypeError):
            encoder.default("not numpy")

    def test_json_dumps_with_encoder(self):
        data = {
            "int_val": np.int32(42),
            "float_val": np.float64(3.14),
            "array_val": np.array([1, 2, 3]),
            "bool_val": np.bool_(True),
        }
        json_str = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(json_str)
        assert parsed["int_val"] == 42
        assert parsed["float_val"] == 3.14
        assert parsed["array_val"] == [1, 2, 3]
        assert parsed["bool_val"] is True
