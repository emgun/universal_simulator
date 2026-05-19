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
    )

    assert total == 4
    assert part_path.read_bytes() == b"abcd"
    assert seen_headers == [{"Range": "bytes=10-13"}]
