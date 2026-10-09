"""Retryable S3 uploads/downloads with download resume via partial files."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    ConnectionClosedError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    IncompleteReadError,
    ProxyConnectionError,
    ReadTimeoutError,
    ResponseStreamingError,
)

PARTIAL_SUFFIX = ".express.partial"
DEFAULT_TRANSFER_RETRIES = 5
_READ_CHUNK = 1024 * 1024

_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    ProxyConnectionError,
    EndpointConnectionError,
    BotoConnectionError,
    ConnectionClosedError,
    ConnectTimeoutError,
    ReadTimeoutError,
    IncompleteReadError,
    ResponseStreamingError,
    TimeoutError,
    ConnectionError,
    OSError,
)

try:  # pragma: no cover - optional dependency surface
    from urllib3.exceptions import HTTPError as Urllib3HTTPError

    _RETRYABLE_TYPES = (*_RETRYABLE_TYPES, Urllib3HTTPError)
except ImportError:  # pragma: no cover
    pass


ByteProgress = Callable[[int], None]


class RetryAwareProgress:
    """
    Forward byte deltas to a sink without double-counting across retries.

    boto3 callbacks report per-chunk deltas starting over on each attempt.
    This wrapper only emits bytes beyond the high-water mark already reported.
    """

    def __init__(self, sink: ByteProgress | None = None):
        self._sink = sink
        self._emitted = 0
        self._attempt = 0

    def begin_attempt(self, *, already: int = 0) -> None:
        self._attempt = max(int(already), 0)
        if self._attempt > self._emitted:
            delta = self._attempt - self._emitted
            self._emitted = self._attempt
            if self._sink is not None and delta:
                self._sink(delta)

    def __call__(self, bytes_amount: int) -> None:
        n = max(int(bytes_amount), 0)
        if not n:
            return
        self._attempt += n
        if self._attempt > self._emitted:
            delta = self._attempt - self._emitted
            self._emitted = self._attempt
            if self._sink is not None:
                self._sink(delta)


def is_retryable_transfer_error(exc: BaseException) -> bool:
    """Return True for transient network / proxy / timeout failures."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        # Throttling and 5xx are worth retrying; auth / not-found are not.
        if code in {"500", "502", "503", "504", "SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable"}:
            return True
        http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return http_status in {500, 502, 503, 504}
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    return False


def _backoff_seconds(attempt: int) -> float:
    return min(2 ** attempt, 8)


def partial_path_for(dest: Path) -> Path:
    return dest.with_name(dest.name + PARTIAL_SUFFIX)


def remote_object_size(client, bucket: str, key: str) -> int | None:
    """Return ContentLength when the object exists, else None."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        return int(head.get("ContentLength") or 0)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def is_remote_object_complete(client, bucket: str, key: str, expected_size: int) -> bool:
    size = remote_object_size(client, bucket, key)
    if size is None:
        return False
    return size == int(expected_size)


def delete_remote_object(client, bucket: str, key: str) -> None:
    client.delete_object(Bucket=bucket, Key=key)


def call_with_transfer_retries(
        operation: Callable[[], None],
        *,
        retries: int = DEFAULT_TRANSFER_RETRIES,
        on_retry: Callable[[int, BaseException], None] | None = None,
) -> None:
    """Run ``operation`` until it succeeds or non-retryable / retries exhausted."""
    attempts = max(int(retries), 1)
    for attempt in range(attempts):
        try:
            operation()
            return
        except BaseException as exc:
            if not is_retryable_transfer_error(exc) or attempt + 1 >= attempts:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            time.sleep(_backoff_seconds(attempt))


def upload_file_with_retries(
        client,
        local_path: str | Path,
        bucket: str,
        key: str,
        *,
        callback: ByteProgress | None = None,
        retries: int = DEFAULT_TRANSFER_RETRIES,
        expected_size: int | None = None,
) -> None:
    """Upload a local file, retrying transient failures; skip if remote already complete."""
    path = Path(local_path)
    size = expected_size if expected_size is not None else path.stat().st_size
    progress = RetryAwareProgress(callback)

    def attempt() -> None:
        existing = remote_object_size(client, bucket, key)
        if existing is not None and existing == size:
            progress.begin_attempt(already=size)
            return
        # Wrong-sized leftover from a previous failed put — replace it.
        if existing is not None and existing != size:
            delete_remote_object(client, bucket, key)
        progress.begin_attempt()
        client.upload_file(str(path), bucket, key, Callback=progress)

    call_with_transfer_retries(attempt, retries=retries)


def upload_fileobj_with_retries(
        client,
        open_reader: Callable[[], object],
        bucket: str,
        key: str,
        *,
        callback: ByteProgress | None = None,
        retries: int = DEFAULT_TRANSFER_RETRIES,
        expected_size: int | None = None,
) -> None:
    """
    Upload from a freshly opened file-like object each attempt.

    ``open_reader`` must return a new reader positioned at the payload start
    (``_FileRangeReader`` is recreated per attempt).
    """
    progress = RetryAwareProgress(callback)

    def attempt() -> None:
        if expected_size is not None:
            existing = remote_object_size(client, bucket, key)
            if existing is not None and existing == expected_size:
                progress.begin_attempt(already=expected_size)
                return
            if existing is not None and existing != expected_size:
                delete_remote_object(client, bucket, key)
        progress.begin_attempt()
        reader = open_reader()
        try:
            client.upload_fileobj(reader, bucket, key, Callback=progress)
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

    call_with_transfer_retries(attempt, retries=retries)


def download_file_with_resume(
        client,
        bucket: str,
        key: str,
        dest: Path,
        *,
        callback: ByteProgress | None = None,
        retries: int = DEFAULT_TRANSFER_RETRIES,
        total_bytes: int | None = None,
) -> None:
    """
    Download ``key`` to ``dest``, writing through ``dest.name.express.partial``.

    Partial files are kept across failures so a later attempt resumes with HTTP Range.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path_for(dest)
    progress = RetryAwareProgress(callback)

    def attempt() -> None:
        total = total_bytes
        if total is None:
            try:
                total = remote_object_size(client, bucket, key)
            except Exception:
                total = None

        if total is None:
            # Size unknown: full download to partial (no Range resume).
            if partial.exists():
                partial.unlink()
            progress.begin_attempt()
            client.download_file(bucket, key, str(partial), Callback=progress)
            os.replace(partial, dest)
            return

        already = partial.stat().st_size if partial.exists() else 0
        if already > total:
            partial.unlink(missing_ok=True)
            already = 0
        if already == total:
            progress.begin_attempt(already=total)
            os.replace(partial, dest)
            return

        progress.begin_attempt(already=already)
        resumed = already > 0
        if resumed:
            response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={already}-")
            body = response["Body"]
            try:
                with partial.open("ab") as handle:
                    while True:
                        chunk = body.read(_READ_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        progress(len(chunk))
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        else:
            client.download_file(bucket, key, str(partial), Callback=progress)

        final_size = partial.stat().st_size if partial.exists() else 0
        if final_size != total:
            # A finished download_file that still mismatches is corrupt — restart clean.
            if not resumed:
                partial.unlink(missing_ok=True)
            raise IncompleteReadError(
                actual_bytes=final_size,
                expected_bytes=total,
            )
        os.replace(partial, dest)

    call_with_transfer_retries(attempt, retries=retries)
