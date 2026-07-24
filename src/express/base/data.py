import zlib
from typing import TypeVar, Type, List, Any, Iterable

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class Torrent(str):
    ...

def assert_serializable(obj: BaseModel) -> bool:
    try:
        obj.model_dump_json()  # Pydantic v2
        return True
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Object is not JSON-serializable: {e}")
def get_torrent(obj: BaseModel) -> Torrent:
    """
    Serialize a reversible torrent from the given object.
    :param obj:
    :return:
    """
    return Torrent(zlib.compress(obj.model_dump_json().encode()).hex())
def from_torrent(torrent: Torrent, expect_type: Type[T]) -> T:
    """
    Deserialize a torrent back into the expected Pydantic type.
    :param torrent: Torrent
    :param expect_type: The expected Pydantic model type
    :return:
    """
    compressed = bytes.fromhex(torrent)
    json_bytes = zlib.decompress(compressed)
    json_str = json_bytes.decode()
    return expect_type.model_validate_json(json_str)


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def search_exact(
        items: List[T],
        field: str,
        value,
        case_sensitive: bool = True
) -> List[T]:
    """
    Exact-match search on a list of Pydantic model objects by field value.
    """
    def eq(a, b):
        if isinstance(a, str) and not case_sensitive:
            a, b = a.lower(), str(b).lower()
        return a == b
    return [item for item in items if eq(getattr(item, field), value)]


def search_models(
        items: Iterable[T],
        *,
        query: str | None = None,
        name: str | None = None,
        tag: str | None = None,
        author: str | None = None,
        version: str | None = None,
        emails: str | None = None,
) -> List[T]:
    """
    Case-insensitive substring search across common metadata fields.

    Free-text `query` words are AND-matched against name/authors/emails/version/tags.
    Optional field filters further narrow results.
    """
    words = [part.lower() for part in (query or "").split() if part.strip()]

    def match(item: T) -> bool:
        item_name = _field_text(getattr(item, "name", None))
        item_authors = _field_text(getattr(item, "authors", None))
        item_emails = _field_text(getattr(item, "emails", None))
        item_version = _field_text(getattr(item, "version", None))
        item_tags = getattr(item, "tags", None) or []
        tags_text = _field_text(item_tags)

        if name and name.lower() not in item_name.lower():
            return False
        if author and author.lower() not in item_authors.lower():
            return False
        if emails and emails.lower() not in item_emails.lower():
            return False
        if version and version.lower() not in item_version.lower():
            return False
        if tag:
            needle = tag.lower()
            if not any(needle in str(t).lower() for t in item_tags):
                return False
        if words:
            haystack = " ".join(
                [item_name, item_authors, item_emails, item_version, tags_text]
            ).lower()
            if not all(word in haystack for word in words):
                return False
        return True

    return [item for item in items if match(item)]
