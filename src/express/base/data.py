import zlib
from typing import TypeVar, Type, List

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
def search_exact(
        items: List[T],
        field: str,
        value,
        case_sensitive: bool = True
) -> List[T]:
    """
    Exact-match search on a list of Pydantic model objects by field value.
    :param items:
    :param field:
    :param value:
    :param case_sensitive:
    :return:
    """
    def eq(a, b):
        if isinstance(a, str) and not case_sensitive:
            a, b = a.lower(), str(b).lower()
        return a == b
    return [item for item in items if eq(getattr(item, field), value)]
