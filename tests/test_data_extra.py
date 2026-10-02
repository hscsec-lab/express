from unittest.mock import patch

import pytest
from pydantic import BaseModel

from express.base.data import _as_text, assert_serializable, from_torrent, get_torrent, search_exact


class Demo(BaseModel):
    name: str


def test_assert_serializable_failure():
    class Bad(BaseModel):
        name: str = "x"

    bad = Bad()
    with patch.object(Bad, "model_dump_json", side_effect=TypeError("nope")):
        with pytest.raises(AssertionError):
            assert_serializable(bad)


def test_torrent_type_roundtrip():
    obj = Demo(name="x")
    torrent = get_torrent(obj)
    assert from_torrent(torrent, Demo).name == "x"


def test_as_text_none_and_list():
    assert _as_text(None) == ""
    assert _as_text([1, "a"]) == "1 a"


def test_search_exact_case_insensitive():
    items = [Demo(name="AbC")]
    assert search_exact(items, "name", "abc", case_sensitive=False)
