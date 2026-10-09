"""Unit tests for retryable S3 transfer helpers and download resume."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import (
    ClientError,
    IncompleteReadError,
    ProxyConnectionError,
)

from express.base.s3_transfer import (
    RetryAwareProgress,
    call_with_transfer_retries,
    download_file_with_resume,
    is_remote_object_complete,
    is_retryable_transfer_error,
    partial_path_for,
    remote_object_size,
    upload_file_with_retries,
    upload_fileobj_with_retries,
)


def test_retry_aware_progress_dedupes_retries():
    seen: list[int] = []
    progress = RetryAwareProgress(seen.append)
    progress.begin_attempt()
    progress(3)
    progress(2)
    assert seen == [3, 2]
    progress.begin_attempt()
    progress(4)  # below high-water (5) → emit 0 new until past 5
    assert seen == [3, 2]
    progress(2)  # attempt=6 → emit 1
    assert seen == [3, 2, 1]


def test_retry_aware_progress_seed_already():
    seen: list[int] = []
    progress = RetryAwareProgress(seen.append)
    progress.begin_attempt(already=10)
    assert seen == [10]
    progress(2)
    assert seen == [10, 2]


def test_is_retryable_errors():
    assert is_retryable_transfer_error(ProxyConnectionError(proxy_url="http://p"))
    assert is_retryable_transfer_error(ConnectionError("net"))
    assert is_retryable_transfer_error(TimeoutError("t"))
    assert not is_retryable_transfer_error(KeyboardInterrupt())
    assert not is_retryable_transfer_error(SystemExit(1))
    assert is_retryable_transfer_error(
        ClientError({"Error": {"Code": "503", "Message": "busy"}, "ResponseMetadata": {"HTTPStatusCode": 503}}, "GetObject")
    )
    assert not is_retryable_transfer_error(
        ClientError({"Error": {"Code": "404", "Message": "missing"}}, "HeadObject")
    )
    assert is_retryable_transfer_error(
        ClientError(
            {"Error": {"Code": "SlowDown", "Message": "slow"}, "ResponseMetadata": {"HTTPStatusCode": 503}},
            "PutObject",
        )
    )


def test_call_with_transfer_retries_succeeds_after_failures(monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProxyConnectionError(proxy_url="http://p")

    call_with_transfer_retries(flaky, retries=5)
    assert calls["n"] == 3


def test_call_with_transfer_retries_exhausted(monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)

    def always() -> None:
        raise ProxyConnectionError(proxy_url="http://p")

    with pytest.raises(ProxyConnectionError):
        call_with_transfer_retries(always, retries=2)


def test_call_with_transfer_retries_non_retryable():
    with pytest.raises(ValueError):
        call_with_transfer_retries(lambda: (_ for _ in ()).throw(ValueError("x")), retries=5)


def test_call_with_transfer_retries_on_retry_hook(monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    seen: list[int] = []

    def flaky() -> None:
        if not seen:
            raise ProxyConnectionError(proxy_url="http://p")

    def hook(attempt: int, exc: BaseException) -> None:
        seen.append(attempt)
        assert isinstance(exc, ProxyConnectionError)

    call_with_transfer_retries(flaky, retries=3, on_retry=hook)
    assert seen == [1]


def test_retry_aware_progress_ignores_zero_and_missing_sink():
    progress = RetryAwareProgress(None)
    progress.begin_attempt()
    progress(0)
    progress(4)
    assert progress._emitted == 4
    progress.begin_attempt(already=0)


def test_backoff_caps_at_eight_seconds():
    from express.base.s3_transfer import _backoff_seconds

    assert _backoff_seconds(0) == 1
    assert _backoff_seconds(10) == 8


def test_remote_object_size_and_complete():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 10}
    assert remote_object_size(client, "b", "k") == 10
    assert is_remote_object_complete(client, "b", "k", 10)
    assert not is_remote_object_complete(client, "b", "k", 9)

    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    assert remote_object_size(client, "b", "k") is None
    assert not is_remote_object_complete(client, "b", "k", 10)


def test_remote_object_size_reraises_non_404():
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "boom"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        remote_object_size(client, "b", "k")


def test_upload_file_retries_and_skips_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    path = tmp_path / "a.bin"
    path.write_bytes(b"abcd")
    client = MagicMock()
    client.head_object.side_effect = [
        ClientError({"Error": {"Code": "404", "Message": "m"}}, "HeadObject"),
        ClientError({"Error": {"Code": "404", "Message": "m"}}, "HeadObject"),
    ]
    calls = {"n": 0}

    def upload(*_a, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProxyConnectionError(proxy_url="http://p")
        cb = kwargs.get("Callback")
        if cb:
            cb(4)

    client.upload_file.side_effect = upload
    seen: list[int] = []
    upload_file_with_retries(client, path, "b", "k", callback=seen.append, retries=4)
    assert calls["n"] == 2
    assert sum(seen) == 4


def test_upload_file_replaces_wrong_sized_remote(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    path = tmp_path / "a.bin"
    path.write_bytes(b"abcd")
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 1}
    client.upload_file.side_effect = lambda *a, **k: None
    upload_file_with_retries(client, path, "b", "k", expected_size=4, retries=2)
    client.delete_object.assert_called()
    client.upload_file.assert_called_once()


def test_upload_fileobj_with_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    path = tmp_path / "a.bin"
    path.write_bytes(b"xyz")
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "m"}}, "HeadObject"
    )
    opened = {"n": 0}

    def open_reader():
        opened["n"] += 1
        return path.open("rb")

    def upload(reader, bucket, key, Callback=None):
        data = reader.read()
        assert data == b"xyz"
        if Callback:
            Callback(len(data))
        if opened["n"] == 1:
            raise ProxyConnectionError(proxy_url="http://p")

    client.upload_fileobj.side_effect = upload
    upload_fileobj_with_retries(client, open_reader, "b", "k", expected_size=3, retries=3)
    assert opened["n"] == 2


def test_download_file_with_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    dest = tmp_path / "out.bin"
    partial = partial_path_for(dest)
    payload = b"0123456789"
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": len(payload)}

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(payload[:4])
        if Callback:
            Callback(4)
        raise ProxyConnectionError(proxy_url="http://p")

    client.download_file.side_effect = download

    class Body:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def read(self, n: int = -1):
            if self._pos >= len(self._data):
                return b""
            if n < 0:
                chunk = self._data[self._pos :]
                self._pos = len(self._data)
                return chunk
            chunk = self._data[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

        def close(self):
            return None

    def get_object(**kwargs):
        assert kwargs["Range"] == "bytes=4-"
        return {"Body": Body(payload[4:])}

    client.get_object.side_effect = get_object
    seen: list[int] = []
    download_file_with_resume(client, "b", "k", dest, callback=seen.append, retries=4, total_bytes=len(payload))
    assert dest.read_bytes() == payload
    assert not partial.exists()
    assert sum(seen) == len(payload)


def test_download_resume_from_existing_partial(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    dest = tmp_path / "out.bin"
    partial = partial_path_for(dest)
    payload = b"abcdefgh"
    partial.write_bytes(payload[:3])
    client = MagicMock()

    class Body:
        def __init__(self):
            self._data = payload[3:]
            self._i = 0

        def read(self, n: int = -1):
            if self._i >= len(self._data):
                return b""
            chunk = self._data[self._i : self._i + (n if n > 0 else len(self._data))]
            self._i += len(chunk)
            return chunk

        def close(self):
            return None

    client.get_object.return_value = {"Body": Body()}
    download_file_with_resume(client, "b", "k", dest, retries=2, total_bytes=len(payload))
    assert dest.read_bytes() == payload
    client.download_file.assert_not_called()


def test_download_complete_partial_just_renames(tmp_path):
    dest = tmp_path / "out.bin"
    partial = partial_path_for(dest)
    partial.write_bytes(b"done")
    client = MagicMock()
    download_file_with_resume(client, "b", "k", dest, retries=1, total_bytes=4)
    assert dest.read_bytes() == b"done"
    client.download_file.assert_not_called()
    client.get_object.assert_not_called()


def test_download_oversized_partial_is_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    dest = tmp_path / "out.bin"
    partial = partial_path_for(dest)
    partial.write_bytes(b"too-large-partial")
    client = MagicMock()

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"ok")
        if Callback:
            Callback(2)

    client.download_file.side_effect = download
    download_file_with_resume(client, "b", "k", dest, retries=2, total_bytes=2)
    assert dest.read_bytes() == b"ok"


def test_download_unknown_size_full_download(tmp_path):
    dest = tmp_path / "out.bin"
    partial = partial_path_for(dest)
    partial.write_bytes(b"stale")
    client = MagicMock()
    client.head_object.side_effect = RuntimeError("no head")

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"full")

    client.download_file.side_effect = download
    download_file_with_resume(client, "b", "k", dest, retries=1, total_bytes=None)
    assert dest.read_bytes() == b"full"


def test_download_incomplete_read_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    dest = tmp_path / "out.bin"
    client = MagicMock()
    n = {"c": 0}

    def download(bucket, key, filename, Callback=None):
        n["c"] += 1
        if n["c"] == 1:
            Path(filename).write_bytes(b"xx")
            if Callback:
                Callback(2)
            return
        Path(filename).write_bytes(b"abcd")
        if Callback:
            Callback(4)

    client.download_file.side_effect = download
    download_file_with_resume(client, "b", "k", dest, retries=3, total_bytes=4)
    assert dest.read_bytes() == b"abcd"
    assert n["c"] == 2


def test_upload_fileobj_skips_when_complete(tmp_path):
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 3}
    opened = {"n": 0}

    def open_reader():
        opened["n"] += 1
        return (tmp_path / "x").open("wb")

    upload_fileobj_with_retries(client, open_reader, "b", "k", expected_size=3, retries=1)
    assert opened["n"] == 0
    client.upload_fileobj.assert_not_called()


def test_upload_fileobj_replaces_wrong_size(tmp_path, monkeypatch):
    monkeypatch.setattr("express.base.s3_transfer.time.sleep", lambda *_: None)
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 1}

    def upload(reader, bucket, key, Callback=None):
        assert reader.read() == b"abc"
        reader.close()

    client.upload_fileobj.side_effect = upload
    upload_fileobj_with_retries(
        client,
        lambda: path.open("rb"),
        "b",
        "k",
        expected_size=3,
        retries=1,
    )
    client.delete_object.assert_called_once()


def test_upload_fileobj_without_expected_size(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"ab")
    client = MagicMock()
    client.upload_fileobj.side_effect = lambda reader, bucket, key, Callback=None: reader.close()
    upload_fileobj_with_retries(client, lambda: path.open("rb"), "b", "k", retries=1)
    client.head_object.assert_not_called()


def test_upload_file_skips_when_remote_already_complete(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abcd")
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 4}
    seen: list[int] = []
    upload_file_with_retries(client, path, "b", "k", callback=seen.append, retries=1)
    client.upload_file.assert_not_called()
    assert seen == [4]


def test_upload_file_uses_stat_size_when_expected_omitted(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"xy")
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "m"}}, "HeadObject"
    )
    client.upload_file.side_effect = lambda *a, **k: None
    upload_file_with_retries(client, path, "b", "k", retries=1)
    client.upload_file.assert_called_once()
