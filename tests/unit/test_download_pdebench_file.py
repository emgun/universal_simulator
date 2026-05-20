from pathlib import Path

import pytest

from scripts import download_pdebench_file as downloader


def test_part_ranges_cover_expected_size():
    assert downloader._part_ranges(10, 4) == [(0, 0, 3), (1, 4, 7), (2, 8, 9)]


def test_assemble_parts_preserves_order(tmp_path: Path):
    dest = tmp_path / "out.h5"
    parts = []
    for index, payload in enumerate([b"aa", b"bb", b"cc"]):
        part = tmp_path / f"part-{index}"
        part.write_bytes(payload)
        parts.append(part)

    total = downloader._assemble_parts(dest, parts)

    assert total == 6
    assert dest.read_bytes() == b"aabbcc"


def test_download_skips_existing_file_with_matching_checksum(tmp_path: Path):
    dest = tmp_path / "data.h5"
    dest.write_bytes(b"payload")
    entry = {
        "file_id": 1,
        "size_bytes": len(b"payload"),
        "checksum": downloader.hashlib.md5(b"payload").hexdigest(),
    }

    downloader.download(entry, dest, workers=4)

    assert dest.read_bytes() == b"payload"


def test_download_part_rejects_non_range_response(monkeypatch, tmp_path: Path):
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"abcd"

    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(SystemExit, match="expected 206"):
        downloader._download_part(
            "https://example.test/file",
            tmp_path / "part",
            0,
            3,
            chunk_size=2,
            retries=1,
            part_timeout=30,
        )


def test_download_part_writes_expected_range(monkeypatch, tmp_path: Path):
    seen_headers = []

    class Response:
        status_code = 206

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"ab"
            yield b"cd"

    def fake_get(url, headers=None, stream=True, timeout=None):
        seen_headers.append(headers)
        return Response()

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    part_path = tmp_path / "part"

    total = downloader._download_part(
        "https://example.test/file",
        part_path,
        10,
        13,
        chunk_size=2,
        retries=1,
        part_timeout=30,
    )

    assert total == 4
    assert part_path.read_bytes() == b"abcd"
    assert seen_headers == [{"Range": "bytes=10-13"}]


def test_download_part_retries_after_part_timeout(monkeypatch, tmp_path: Path):
    attempts = 0

    class Response:
        status_code = 206

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"ab"
            yield b"cd"

    def fake_get(url, headers=None, stream=True, timeout=None):
        nonlocal attempts
        attempts += 1
        return Response()

    ticks = iter([0, 100, 0, 1, 2])
    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "monotonic", lambda: next(ticks))
    part_path = tmp_path / "part"

    total = downloader._download_part(
        "https://example.test/file",
        part_path,
        0,
        3,
        chunk_size=2,
        retries=2,
        part_timeout=10,
        retry_backoff=0,
    )

    assert attempts == 2
    assert total == 4
    assert part_path.read_bytes() == b"abcd"


def test_download_part_to_file_writes_at_range_offset(monkeypatch, tmp_path: Path):
    class Response:
        status_code = 206

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"cd"
            yield b"ef"

    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: Response())
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"abcdefgh")

    total = downloader._download_part_to_file(
        "https://example.test/file",
        dest,
        2,
        5,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
    )

    assert total == 4
    assert dest.read_bytes() == b"abcdefgh"


def test_download_ranges_uses_single_temp_file(monkeypatch, tmp_path: Path):
    def fake_download_part_to_file(
        url,
        dest,
        start,
        end,
        *,
        chunk_size,
        retries,
        part_timeout,
        retry_backoff,
        split_after_retries,
        min_split_size,
    ):
        with open(dest, "r+b") as fh:
            fh.seek(start)
            fh.write(bytes([65 + start]) * (end - start + 1))
        return end - start + 1

    monkeypatch.setattr(downloader, "_download_part_to_file", fake_download_part_to_file)
    dest = tmp_path / "data.h5"

    total = downloader._download_ranges(
        "https://example.test/file",
        dest,
        6,
        workers=2,
        part_size=2,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
        split_after_retries=2,
        min_split_size=1,
    )

    assert total == 6
    assert dest.read_bytes() == b"AACCEE"
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()
    assert not dest.with_suffix(dest.suffix + ".parts").exists()


def test_download_part_to_file_backs_off_between_retries(monkeypatch, tmp_path: Path):
    attempts = 0
    sleeps = []

    class Response:
        status_code = 206

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"ab"

    def fake_get(url, headers=None, stream=True, timeout=None):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise downloader.requests.ConnectionError("network down")
        return Response()

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"ab")

    total = downloader._download_part_to_file(
        "https://example.test/file",
        dest,
        0,
        1,
        chunk_size=2,
        retries=3,
        part_timeout=30,
        retry_backoff=5,
    )

    assert attempts == 3
    assert sleeps == [5, 10]
    assert total == 2


def test_download_part_to_file_splits_repeated_timeout(monkeypatch, tmp_path: Path):
    seen_ranges = []

    class Response:
        status_code = 206

        def __init__(self, payload: bytes):
            self.payload = payload

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield self.payload

    def fake_get(url, headers=None, stream=True, timeout=None):
        range_header = headers["Range"]
        seen_ranges.append(range_header)
        if range_header == "bytes=0-7":
            raise downloader.requests.Timeout("stuck range")
        start, end = [int(value) for value in range_header.removeprefix("bytes=").split("-")]
        return Response(bytes([65 + start]) * (end - start + 1))

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"\0" * 8)

    total = downloader._download_part_to_file(
        "https://example.test/file",
        dest,
        0,
        7,
        chunk_size=8,
        retries=3,
        part_timeout=30,
        retry_backoff=0,
        split_after_retries=2,
        min_split_size=2,
    )

    assert total == 8
    assert seen_ranges == ["bytes=0-7", "bytes=0-7", "bytes=0-3", "bytes=4-7"]
    assert dest.read_bytes() == b"AAAAEEEE"
