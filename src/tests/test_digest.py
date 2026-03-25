import pytest
from pydantic import ValidationError

from base.model import Metadata


def test_metadata_valid_semver():
    data = {"name": "test", "version": "1.2.3", "tags": ["ml"], "authors": "Alice"}
    meta = Metadata(**data)
    assert meta.version == "1.2.3"
    assert meta.name == "test"


def test_metadata_invalid_semver():
    with pytest.raises(ValidationError, match="not a valid SemVer"):
        Metadata(name="test", version="1.2")


def test_metadata_optional_fields():
    meta = Metadata(name="minimal")
    assert meta.version is None
    assert meta.authors is None
    assert meta.tags is None