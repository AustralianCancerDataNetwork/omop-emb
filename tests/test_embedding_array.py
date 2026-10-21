"""as_embedding_array: the one place embedding arrays are coerced and checked."""

from __future__ import annotations

import numpy as np
import pytest

from omop_emb.utils.embedding_utils import as_embedding_array


def test_float32_array_is_returned_as_is():
    array = np.ones((2, 3), dtype=np.float32)
    assert as_embedding_array(array, 3) is array


@pytest.mark.parametrize("source", [np.ones((2, 3), dtype=np.float64), [[1, 2, 3], [4, 5, 6]]])
def test_other_inputs_become_float32(source):
    result = as_embedding_array(source, 3)
    assert result.dtype == np.float32
    assert result.shape == (2, 3)


def test_dimensions_are_optional():
    assert as_embedding_array(np.ones((2, 5))).shape == (2, 5)


@pytest.mark.parametrize(
    "array, message",
    [
        (np.ones(3), "2-D"),
        (np.ones((1, 2, 3)), "2-D"),
        (np.ones((2, 4)), "dimensionality"),
        (np.array([[1.0, np.nan, 3.0]]), "NaN or infinite"),
        (np.array([[1.0, np.inf, 3.0]]), "NaN or infinite"),
        (np.array([[1e300, 1.0, 1.0]]), "NaN or infinite"),
    ],
    ids=["1-D", "3-D", "wrong dimensionality", "NaN", "infinite", "overflows float32"],
)
def test_malformed_arrays_raise(array, message):
    with pytest.raises(ValueError, match=message):
        as_embedding_array(array, 3)
