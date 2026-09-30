from express.base.file import FileMetadata
from express.base.storage.registry import content_keys_for_index


def test_content_keys_for_index_merges():
    a = FileMetadata(file_name="a", file_checksum_sha256="h1", file_relative_path="a")
    b = FileMetadata(
        file_name="b.safetensors",
        file_checksum_sha256="manifest",
        file_relative_path="b.safetensors",
        storage_unit="safetensors_v1",
        unit_manifest={"parts": [{"content_hash": "t1"}, {"content_hash": "t2"}]},
    )
    keys = content_keys_for_index([a, b])
    assert keys == {"h1", "t1", "t2"}
