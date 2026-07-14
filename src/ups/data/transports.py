from __future__ import annotations

"""Small, dependency-free transports used by the verified data stager."""

import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path


class TransportError(RuntimeError):
    """Raised when a source cannot be transferred safely."""


_B2_BUCKET = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{4,48}[A-Za-z0-9]")
_ENV_ASSIGNMENT = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)")
_B2_ENV_KEYS = frozenset(
    {
        "B2_KEY_ID",
        "B2_ACCOUNT_ID",
        "B2_APP_KEY",
        "B2_APPLICATION_KEY",
        "B2_BUCKET",
        "B2_BUCKET_NAME",
        "B2_S3_ENDPOINT",
        "B2_S3_REGION",
    }
)


def _local_path(uri: str) -> Path | None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "":
        return Path(uri).expanduser()
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise TransportError(f"Unsupported file URI host: {parsed.netloc!r}")
        return Path(urllib.request.url2pathname(parsed.path))
    return None


def copy_local_resumable(source: Path, destination: Path, *, chunk_size: int = 8 << 20) -> int:
    """Resume a local copy into ``destination`` and return bytes transferred now."""

    source = source.resolve()
    if not source.is_file():
        raise TransportError(f"Local source is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    offset = destination.stat().st_size if destination.exists() else 0
    if offset > source_size:
        destination.unlink()
        offset = 0

    transferred = 0
    with source.open("rb") as src, destination.open("ab" if offset else "wb") as dst:
        src.seek(offset)
        while True:
            block = src.read(chunk_size)
            if not block:
                break
            dst.write(block)
            transferred += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    return transferred


def download_http_resumable(
    uri: str,
    destination: Path,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    chunk_size: int = 8 << 20,
) -> int:
    """Download HTTP(S), resuming only when the server confirms a byte range."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    request_headers = dict(headers or {})
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(uri, headers=request_headers)

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except Exception as exc:
        raise TransportError(f"Unable to download {uri}: {exc}") from exc

    try:
        with response:
            status = getattr(response, "status", response.getcode())
            append = bool(offset and status == 206)
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise TransportError(
                        f"Server returned an unexpected range for {uri}: {content_range!r}"
                    )
            # A server may ignore Range and return 200. Restart instead of appending
            # duplicate bytes to a partial file.
            mode = "ab" if append else "wb"
            transferred = 0
            with destination.open(mode) as dst:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    dst.write(block)
                    transferred += len(block)
                dst.flush()
                os.fsync(dst.fileno())
    except TransportError:
        raise
    except Exception as exc:
        raise TransportError(f"Interrupted while downloading {uri}: {exc}") from exc
    return transferred


def _b2_remote_path(uri: str) -> str:
    """Validate a B2 object URI and map it to the fixed environment remote."""

    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme.lower() != "b2":
        raise TransportError(f"Expected a b2 URI, got {uri!r}")
    if parsed.query or parsed.fragment:
        raise TransportError("B2 object URIs must not contain a query or fragment")
    if parsed.username is not None or parsed.password is not None:
        raise TransportError("B2 object URIs must not contain credentials")
    if not _B2_BUCKET.fullmatch(parsed.netloc):
        raise TransportError(f"Invalid B2 bucket name in URI: {uri!r}")
    if "%" in parsed.path:
        raise TransportError("Percent-encoded B2 object keys are not supported")
    key = parsed.path.removeprefix("/")
    parts = key.split("/")
    if not key or any(part in {"", ".", ".."} for part in parts):
        raise TransportError(f"B2 URI must contain a safe object key: {uri!r}")
    if "\\" in key or any(ord(character) < 32 or ord(character) == 127 for character in key):
        raise TransportError(f"B2 URI contains an unsafe object key: {uri!r}")
    return f"UPSB2:{parsed.netloc}/{key}"


def _load_b2_env_file() -> dict[str, str]:
    """Read only literal B2 assignments; never evaluate the file as shell code."""

    path = Path(os.environ.get("ENV_FILE", ".env"))
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TransportError(f"Unable to read B2 environment file: {path}") from exc
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_ASSIGNMENT.fullmatch(line)
        if match is None or match.group(1) not in _B2_ENV_KEYS:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _rclone_b2_env(bucket: str) -> dict[str, str]:
    """Build an isolated rclone environment, preferring process values."""

    file_values = _load_b2_env_file()

    def value(*names: str) -> str | None:
        for name in names:
            if name in os.environ:
                return os.environ[name] or None
        for name in names:
            file_value = file_values.get(name)
            if file_value:
                return file_value
        return None

    configured_bucket = value("B2_BUCKET", "B2_BUCKET_NAME")
    if configured_bucket and configured_bucket != bucket:
        raise TransportError(f"B2 URI bucket {bucket!r} does not match the configured B2_BUCKET")
    key_id = value("B2_KEY_ID", "B2_ACCOUNT_ID")
    app_key = value("B2_APP_KEY", "B2_APPLICATION_KEY")
    env = os.environ.copy()
    if not key_id or not app_key:
        if env.get("RCLONE_CONFIG_UPSB2_TYPE"):
            return env
        raise TransportError(
            "B2 staging requires B2_KEY_ID and B2_APP_KEY in the environment or ENV_FILE"
        )
    endpoint = value("B2_S3_ENDPOINT")
    region = value("B2_S3_REGION")
    if endpoint:
        env.update(
            {
                "RCLONE_CONFIG_UPSB2_TYPE": "s3",
                "RCLONE_CONFIG_UPSB2_PROVIDER": "Other",
                "RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID": key_id,
                "RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY": app_key,
                "RCLONE_CONFIG_UPSB2_ENDPOINT": endpoint,
            }
        )
        if region:
            env["RCLONE_CONFIG_UPSB2_REGION"] = region
    else:
        env.update(
            {
                "RCLONE_CONFIG_UPSB2_TYPE": "b2",
                "RCLONE_CONFIG_UPSB2_ACCOUNT": key_id,
                "RCLONE_CONFIG_UPSB2_KEY": app_key,
            }
        )
    return env


def download_b2(uri: str, destination: Path) -> int:
    """Fetch one exact B2 object through the environment-configured UPSB2 remote.

    Rclone writes an adjacent temporary file.  A failed command therefore leaves
    any stager partial intact; only a complete regular file is atomically moved
    into its place.
    """

    remote_path = _b2_remote_path(uri)
    bucket = remote_path.removeprefix("UPSB2:").split("/", 1)[0]
    rclone_env = _rclone_b2_env(bucket)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.rclone-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        try:
            result = subprocess.run(
                ["rclone", "copyto", remote_path, str(temporary)],
                check=False,
                capture_output=True,
                text=True,
                env=rclone_env,
            )
        except OSError as exc:
            raise TransportError(
                f"Unable to run rclone for {uri}; configure the UPSB2 remote via environment"
            ) from exc
        if result.returncode != 0:
            # Do not include process output: backend errors may contain sensitive
            # configuration details even though credentials are never command args.
            raise TransportError(
                f"rclone failed with exit code {result.returncode} while fetching {uri}"
            )
        if not temporary.is_file():
            raise TransportError(f"rclone did not produce a regular file while fetching {uri}")
        transferred = temporary.stat().st_size
        os.replace(temporary, destination)
        return transferred
    finally:
        temporary.unlink(missing_ok=True)


def transfer_to_partial(
    uri: str,
    destination: Path,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> int:
    """Transfer a supported URI into a resumable partial file."""

    local = _local_path(uri)
    if local is not None:
        return copy_local_resumable(local, destination)
    scheme = urllib.parse.urlparse(uri).scheme.lower()
    if scheme in ("http", "https"):
        return download_http_resumable(uri, destination, headers=headers, timeout=timeout)
    if scheme == "b2":
        return download_b2(uri, destination)
    raise TransportError(f"Unsupported transport scheme {scheme!r} for {uri}")
