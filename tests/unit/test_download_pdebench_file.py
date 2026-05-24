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


def test_entry_url_prefers_manifest_url():
    entry = {"file_id": 1, "path": "a.hdf5", "url": "https://mirror.example/a.hdf5"}

    assert downloader._entry_url(entry) == "https://mirror.example/a.hdf5"


def test_entry_url_can_use_environment_template(monkeypatch):
    monkeypatch.setenv("PDEBENCH_DATAFILE_URL_TEMPLATE", "https://mirror.example/{file_id}/{path}")

    assert (
        downloader._entry_url({"file_id": 7, "path": "a/b.hdf5"})
        == "https://mirror.example/7/a/b.hdf5"
    )


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
        transport,
        resolve_ip=None,
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
    assert not dest.with_suffix(dest.suffix + ".tmp.ranges.json").exists()


def test_download_ranges_resumes_completed_temp_ranges(monkeypatch, tmp_path: Path):
    calls = []

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
        transport,
        resolve_ip=None,
    ):
        calls.append((start, end))
        with open(dest, "r+b") as fh:
            fh.seek(start)
            fh.write(bytes([65 + start]) * (end - start + 1))
        return end - start + 1

    monkeypatch.setattr(downloader, "_download_part_to_file", fake_download_part_to_file)
    dest = tmp_path / "data.h5"
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    temp_dest.write_bytes(b"AA\0\0\0\0")
    downloader._write_completed_ranges(
        downloader._range_sidecar_path(temp_dest),
        6,
        {downloader._range_key(0, 1)},
    )

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
    assert set(calls) == {(2, 3), (4, 5)}
    assert dest.read_bytes() == b"AACCEE"
    assert not temp_dest.exists()
    assert not downloader._range_sidecar_path(temp_dest).exists()


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


def test_download_part_to_file_can_use_curl_transport(monkeypatch, tmp_path: Path):
    seen_cmds = []

    class Stdout:
        def __init__(self):
            self.chunks = iter([b"ab", b"cd", b""])

        def read(self, chunk_size):
            return next(self.chunks)

    class Stderr:
        def read(self):
            return b""

    class Proc:
        def __init__(self, cmd, stdout=None, stderr=None):
            seen_cmds.append(cmd)
            self.stdout = Stdout()
            self.stderr = Stderr()

        def wait(self):
            return 0

    monkeypatch.setattr(downloader.subprocess, "Popen", Proc)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"\0" * 8)

    total = downloader._download_part_to_file(
        "https://example.test/file",
        dest,
        2,
        5,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
        transport="curl",
    )

    assert total == 4
    assert dest.read_bytes() == b"\0\0abcd\0\0"
    assert "--range" in seen_cmds[0]
    assert "2-5" in seen_cmds[0]


def test_curl_transport_accepts_full_range_before_nonzero_exit(monkeypatch, tmp_path: Path):
    class Stdout:
        def __init__(self):
            self.chunks = iter([b"ab", b"cd", b""])

        def read(self, chunk_size):
            return next(self.chunks)

    class Stderr:
        def read(self):
            return b"operation timed out"

    class Proc:
        def __init__(self, cmd, stdout=None, stderr=None):
            self.stdout = Stdout()
            self.stderr = Stderr()

        def wait(self):
            return 28

    monkeypatch.setattr(downloader.subprocess, "Popen", Proc)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"\0" * 4)

    total = downloader._download_part_to_file_curl(
        "https://example.test/file",
        dest,
        0,
        3,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
    )

    assert total == 4
    assert dest.read_bytes() == b"abcd"


def test_download_part_to_file_auto_falls_back_to_curl_on_name_resolution(
    monkeypatch, tmp_path: Path
):
    class Response:
        status_code = 206

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b""

    def fake_get(url, headers=None, stream=True, timeout=None):
        raise downloader.requests.ConnectionError("Failed to resolve 'example.test'")

    def fake_curl(url, dest, start, end, *, chunk_size, retries, part_timeout, retry_backoff):
        with open(dest, "r+b") as fh:
            fh.seek(start)
            fh.write(b"ok")
        return end - start + 1

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader, "_download_part_to_file_curl", fake_curl)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"\0" * 2)

    total = downloader._download_part_to_file(
        "https://example.test/file",
        dest,
        0,
        1,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
        transport="auto",
    )

    assert total == 2
    assert dest.read_bytes() == b"ok"


def test_resolve_redirect_url_with_curl_parses_location(monkeypatch):
    class Proc:
        returncode = 0
        stdout = (
            "HTTP/1.1 303 See Other\r\n"
            "Date: Fri, 22 May 2026 00:00:00 GMT\r\n"
            "Location: https://objects.example/file?signature=abc\r\n"
            "\r\n"
        )
        stderr = ""

    seen = []

    def fake_run(cmd, check=False, capture_output=True, text=True):
        seen.append(cmd)
        return Proc()

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    assert downloader._resolve_redirect_url_with_curl("https://darus.example/file", timeout=5) == (
        "https://objects.example/file?signature=abc"
    )
    assert seen[0][:4] == ["curl", "--silent", "--show-error", "--head"]


def test_resolve_redirect_url_with_curl_retries_transient_failure(monkeypatch):
    class FailedProc:
        returncode = 6
        stdout = ""
        stderr = "Could not resolve host"

    class OkProc:
        returncode = 0
        stdout = "HTTP/1.1 303 See Other\r\nLocation: https://objects.example/file\r\n\r\n"
        stderr = ""

    calls = []

    def fake_run(cmd, check=False, capture_output=True, text=True):
        calls.append(cmd)
        return FailedProc() if len(calls) == 1 else OkProc()

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)

    assert (
        downloader._resolve_redirect_url_with_curl(
            "https://darus.example/file",
            timeout=5,
            retries=2,
            retry_backoff=0.1,
        )
        == "https://objects.example/file"
    )
    assert len(calls) == 2


def test_download_resolves_redirect_before_ranged_curl_download(monkeypatch, tmp_path: Path):
    used_urls = []

    def fake_resolve(url, *, timeout, retries, retry_backoff, resolve_ip=None):
        assert url == "https://darus.example/file"
        return "https://objects.example/file?signature=abc"

    def fake_download_ranges(
        url,
        dest,
        expected_size,
        *,
        workers,
        part_size,
        chunk_size,
        retries,
        part_timeout,
        retry_backoff,
        split_after_retries,
        min_split_size,
        transport,
        resolve_ip=None,
    ):
        used_urls.append(url)
        dest.write_bytes(b"abcd")
        return expected_size

    monkeypatch.setattr(downloader, "_resolve_redirect_url_with_curl", fake_resolve)
    monkeypatch.setattr(downloader, "_download_ranges", fake_download_ranges)

    downloader.download(
        {"url": "https://darus.example/file", "size_bytes": 4},
        tmp_path / "file.h5",
        workers=2,
        transport="curl",
        resolve_redirect=True,
    )

    assert used_urls == ["https://objects.example/file?signature=abc"]


def test_download_ranges_fails_fast_on_name_resolution_error(monkeypatch, tmp_path: Path):
    calls = []

    def fake_download_part_to_file(*args, **kwargs):
        calls.append(args)
        raise downloader.NameResolutionError("curl exited 6: could not resolve host")

    monkeypatch.setattr(downloader, "_download_part_to_file", fake_download_part_to_file)

    with pytest.raises(downloader.NameResolutionError):
        downloader._download_ranges(
            "https://objects.example/file",
            tmp_path / "file.h5",
            8,
            workers=2,
            part_size=2,
            chunk_size=2,
            retries=1,
            part_timeout=30,
            retry_backoff=0,
            split_after_retries=1,
            min_split_size=1,
            transport="curl",
        )

    assert len(calls) < 4


def test_resolve_host_with_doh_parses_a_records(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Answer": [
                    {"type": 28, "data": "2001:db8::1"},
                    {"type": 1, "data": "129.69.5.99"},
                    {"type": 1, "data": "129.69.5.100"},
                ]
            }

    def fake_get(url, headers=None, timeout=None):
        assert "name=s3.tik.uni-stuttgart.de" in url
        return Response()

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    assert downloader._resolve_host_with_doh("s3.tik.uni-stuttgart.de") == [
        "129.69.5.99",
        "129.69.5.100",
    ]


def test_resolve_host_with_doh_falls_back_to_curl(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise downloader.requests.ConnectionError("blocked")

    class Proc:
        returncode = 0
        stdout = '{"Answer":[{"type":1,"data":"129.69.7.87"}]}'
        stderr = ""

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.subprocess, "run", lambda *args, **kwargs: Proc())

    assert downloader._resolve_host_with_doh("darus.uni-stuttgart.de") == ["129.69.7.87"]


def test_curl_command_includes_resolve_ip(monkeypatch, tmp_path: Path):
    seen_cmds = []

    class Stdout:
        def __init__(self):
            self.chunks = iter([b"ab", b"cd", b""])

        def read(self, chunk_size):
            return next(self.chunks)

    class Stderr:
        def read(self):
            return b""

    class Proc:
        def __init__(self, cmd, stdout=None, stderr=None):
            seen_cmds.append(cmd)
            self.stdout = Stdout()
            self.stderr = Stderr()

        def wait(self):
            return 0

    monkeypatch.setattr(downloader.subprocess, "Popen", Proc)
    dest = tmp_path / "data.h5.tmp"
    dest.write_bytes(b"\0" * 4)

    downloader._download_part_to_file_curl(
        "https://s3.tik.uni-stuttgart.de/file",
        dest,
        0,
        3,
        chunk_size=2,
        retries=1,
        part_timeout=30,
        retry_backoff=0,
        resolve_ip="129.69.5.99",
    )

    assert "--resolve" in seen_cmds[0]
    assert "s3.tik.uni-stuttgart.de:443:129.69.5.99" in seen_cmds[0]
