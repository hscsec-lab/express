import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

import express.base.cleanup as cleanup_module
from express.base.cleanup import CleanupScope, managed_transfer, _try_delete_remote


@pytest.fixture(autouse=True)
def _managed_transfer_exits_via_raise(monkeypatch):
    """Tests must not terminate the runner via os._exit from managed_transfer."""

    def fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(cleanup_module.os, "_exit", fake_exit)
from express.base.progress import TransferSession
from express.base.remote import push
from express.base.transfer import pull_file, push_storage_blob
from express.base.storage.base import StorageBlob


def test_cleanup_scope_local_and_remote(tmp_path):
    partial = tmp_path / "partial.bin"
    partial.write_bytes(b"part")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()

    scope = CleanupScope()
    scope.track_local_file(partial)
    scope.track_remote_key(remote, "deadbeef")
    lines = scope.run()
    assert not partial.exists()
    remote.s3_client.delete_object.assert_called_once()
    assert any("local" in line for line in lines)
    assert any("remote" in line for line in lines)


def test_cleanup_register_hook():
    seen = []
    scope = CleanupScope()
    scope.register(lambda: seen.append(1))
    scope.run()
    assert seen == [1]


def test_try_delete_remote_failure():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.delete_object.side_effect = RuntimeError("network")
    assert _try_delete_remote(remote, "k") is False


def test_managed_transfer_keyboard_interrupt(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        with managed_transfer("Test"):
            raise KeyboardInterrupt
    assert exc.value.code == 130
    assert "interrupted" in capsys.readouterr().out


def test_push_interrupt_cleans_remote_upload(tmp_path, monkeypatch):
    from express.base.model import Model

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.txt").write_text("x", encoding="utf-8")
    model = Model(model_dir)
    model.create_metadata(name="m", version="0.1.0")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    remote.s3_client.upload_fileobj.side_effect = boom

    with pytest.raises(SystemExit):
        push(model, remote)
    remote.s3_client.delete_object.assert_called()


def test_managed_transfer_success():
    with managed_transfer("Ok") as scope:
        assert isinstance(scope, CleanupScope)


def test_push_chunk_tracks_and_clears_remote(tmp_path):
    path = tmp_path / "c.bin"
    path.write_bytes(b"x")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    scope = CleanupScope()
    session = TransferSession("Push", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Push", total=1)
    from express.base.transfer import push_chunk

    push_chunk(remote, path, "remote-key", session=session)
    session.progress.stop()
    assert scope._partial_remote == set()


def test_push_chunk_interrupt_leaves_remote_tracked(tmp_path):
    path = tmp_path / "c.bin"
    path.write_bytes(b"x")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    remote.s3_client.upload_file.side_effect = KeyboardInterrupt
    scope = CleanupScope()
    session = TransferSession("Push", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Push", total=1)
    from express.base.transfer import push_chunk

    with pytest.raises(KeyboardInterrupt):
        push_chunk(remote, path, "remote-key", session=session)
    session.progress.stop()
    assert scope._partial_remote


def test_materialize_safetensors_with_cleanup(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from express.base.storage.registry import build_file_metadata
    from express.base.storage.safetensors import SafetensorsHandler
    from express.base.storage.transfer_ops import materialize_local_file

    src = tmp_path / "m.safetensors"
    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, src)
    meta = build_file_metadata(src, Path("m.safetensors"))
    processed = SafetensorsHandler().process(src)
    blobs = {b.content_hash: b.read_payload() for b in processed.blobs}

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(blobs[key])

    remote.s3_client.download_file.side_effect = download
    remote.s3_client.head_object.return_value = {"ContentLength": len(next(iter(blobs.values())))}

    dest = tmp_path / "out.safetensors"
    scope = CleanupScope()
    session = TransferSession("Pull", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Pull", total=1)
    materialize_local_file(remote, meta, dest, force=True, session=session)
    session.progress.stop()
    assert dest.exists()
    assert scope._partial_locals == set()


def test_push_storage_blob_interrupt_on_fileobj(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(b"payload-bytes")
    blob = StorageBlob(content_hash="hashkey", source_path=path, byte_offset=0, byte_length=len(b"payload-bytes"))
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    remote.s3_client.upload_fileobj.side_effect = KeyboardInterrupt
    scope = CleanupScope()
    session = TransferSession("Push", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Push", total=1)
    with pytest.raises(KeyboardInterrupt):
        push_storage_blob(remote, blob, "hashkey", session=session)
    session.progress.stop()
    assert scope._partial_remote


def test_pull_file_clears_local_tracking(tmp_path):
    dest = tmp_path / "f.bin"
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 3}
    remote.s3_client.download_file.side_effect = lambda *a, **k: dest.write_bytes(b"new")
    scope = CleanupScope()
    session = TransferSession("Pull", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Pull", total=1)
    pull_file(remote, Path("k"), dest, None, force=True, session=session)
    session.progress.stop()
    assert scope._partial_locals == set()


def test_pull_interrupt_cleans_partial_file(tmp_path):
    dest = tmp_path / "out.bin"
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 100}

    def boom(*_a, **_k):
        dest.write_bytes(b"partial")
        raise KeyboardInterrupt

    remote.s3_client.download_file.side_effect = boom
    scope = __import__("express.base.cleanup", fromlist=["CleanupScope"]).CleanupScope()
    session = TransferSession("Pull", 1, cleanup=scope)
    session.progress.start()
    session._overall_id = session.progress.add_task("Pull", total=1)
    with pytest.raises(KeyboardInterrupt):
        pull_file(remote, Path("key"), dest, None, force=True, session=session)
    session.progress.stop()
    scope.run()
    assert not dest.exists()
