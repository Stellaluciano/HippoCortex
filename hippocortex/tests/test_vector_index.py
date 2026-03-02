import pytest

from hippocortex.cortex.vector_index import SimpleVectorIndex


def test_vector_index_search():
    idx = SimpleVectorIndex(dimension=3)
    idx.upsert("a", [1, 0, 0], {"agent_id": "x"})
    idx.upsert("b", [0, 1, 0], {"agent_id": "x"})

    hits = idx.search([1, 0, 0], k=1, filters={"agent_id": "x"})
    assert hits[0][0] == "a"


def test_vector_index_dimension_guard():
    idx = SimpleVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        idx.upsert("bad", [1, 2, 3], {})
