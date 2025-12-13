import zlib
from typing import TypeVar, Type

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
    从obj 序列化出可逆的torrent
    :param obj:
    :return:
    """
    return Torrent(zlib.compress(obj.model_dump_json().encode()).hex())
def from_torrent(torrent: Torrent, expect_type: Type[T]) -> T:
    """
    从torrent获取任意的expect_type
    :param torrent: Torrent
    :param expect_type: 期待的pydantic类型
    :return:
    """
    compressed = bytes.fromhex(torrent)
    json_bytes = zlib.decompress(compressed)
    json_str = json_bytes.decode()
    return expect_type.model_validate_json(json_str)