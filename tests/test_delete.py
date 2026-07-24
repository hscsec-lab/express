from express.base.file import FolderIndex, FileMetadata, fast_checksum
from express.base.model import MODEL_INDEX_FILE_NAME
from express.base.remote import _content_keys_from_index, plan_chunk_deletion


def test_content_keys_skip_index_file(tmp_path):
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"abc")
    index = FolderIndex(folder_index=[
        FileMetadata(
            file_name="weights.bin",
            file_checksum_sha256=fast_checksum(payload),
            file_relative_path=payload.name,
        ),
        FileMetadata(
            file_name=MODEL_INDEX_FILE_NAME,
            file_checksum_sha256="index-hash",
            file_relative_path=MODEL_INDEX_FILE_NAME,
        ),
    ])
    keys = _content_keys_from_index(index)
    assert fast_checksum(payload) in keys
    assert "index-hash" not in keys


def test_plan_chunk_deletion_keeps_shared_objects():
    owned = {"a", "b", "c"}
    referenced = {"b", "d"}
    exclusive, shared = plan_chunk_deletion(owned, referenced)
    assert exclusive == {"a", "c"}
    assert shared == {"b"}


def test_fast_checksum_stable_for_small_file(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello-express")
    assert fast_checksum(path) == fast_checksum(path)
