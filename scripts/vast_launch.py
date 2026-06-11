#!/usr/bin/env python
"""Helper utilities for launching Vast.ai training runs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ONSTART_DIR = REPO_ROOT / ".vast"
REDACTED = "<redacted>"
VAST_API_HOST = "console.vast.ai"


def run(
    cmd: list[str],
    *,
    check: bool = True,
    display_cmd: list[str] | None = None,
    retries: int = 0,
    retry_backoff: float = 5.0,
) -> int:
    shown = display_cmd if display_cmd is not None else cmd
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            print("$", " ".join(shlex.quote(part) for part in shown))
        else:
            print(
                f"Retrying command after transient Vast CLI failure "
                f"(attempt {attempt}/{attempts})...",
                file=sys.stderr,
            )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(_redact_text(result.stdout), end="")
        if result.stderr:
            print(_redact_text(result.stderr), end="", file=sys.stderr)
        effective_returncode = result.returncode
        if result.returncode == 0 and _is_vast_cli_error_output(result.stdout, result.stderr):
            effective_returncode = 1
        if effective_returncode == 0:
            return effective_returncode
        if attempt < attempts and _is_transient_vast_cli_failure(result.stdout, result.stderr):
            time.sleep(max(0.0, float(retry_backoff)) * attempt)
            continue
        if check:
            raise SystemExit(effective_returncode)
        return effective_returncode
    return effective_returncode


def _is_vast_cli_error_output(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    error_markers = (
        "failed with error",
        "your account lacks credit",
    )
    return any(marker in combined for marker in error_markers)


def _is_transient_vast_cli_failure(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    transient_markers = (
        "failed to resolve",
        "nameresolutionerror",
        "temporary failure in name resolution",
        "max retries exceeded with url",
        "connectionerror",
    )
    return any(marker in combined for marker in transient_markers)


def preflight_vast_dns(
    host: str = VAST_API_HOST, *, retries: int = 0, retry_backoff: float = 5.0
) -> bool:
    attempts = max(1, int(retries) + 1)
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            return True
        except OSError as exc:
            last_error = exc
            print(
                f"Vast DNS preflight failed for {host} on attempt {attempt}/{attempts}: {exc}",
                file=sys.stderr,
            )
            if attempt < attempts and retry_backoff > 0:
                time.sleep(max(0.0, float(retry_backoff)) * attempt)
    assert last_error is not None
    return False


def git_remote_url() -> str:
    try:
        out = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            "Could not determine git remote URL. Configure remote.origin first."
        ) from err


def _secret_for_dry_run(value: str | None) -> str | None:
    return REDACTED if value else None


def _redact_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    for part in cmd:
        part = _redact_text(part)
        redacted.append(part)
    return redacted


def _redact_text(text: str) -> str:
    text = re.sub(
        r"(instance_api_key['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])",
        rf"\1{REDACTED}\2",
        text,
    )
    text = re.sub(r"(B2_KEY_ID=)[^,\s'\"]+", rf"\1{REDACTED}", text)
    text = re.sub(r"(B2_APP_KEY=)[^,\s'\"]+", rf"\1{REDACTED}", text)
    text = re.sub(r"(WANDB_API_KEY=)[^,\s'\"]+", rf"\1{REDACTED}", text)
    text = re.sub(r"([?&]api_key=)[^&\s'\"]+", rf"\1{REDACTED}", text)
    text = re.sub(r'(export B2_KEY_ID=")[^"]+(")', rf"\1{REDACTED}\2", text)
    text = re.sub(r'(export B2_APP_KEY=")[^"]+(")', rf"\1{REDACTED}\2", text)
    text = re.sub(r'(export WANDB_API_KEY=")[^"]+(")', rf"\1{REDACTED}\2", text)
    return text


def ensure_onstart(
    datasets: str | None,
    overrides: str | None,
    remote_script: str,
    script_args: str | None,
    skip_prefetch: bool,
    git_ref: str | None,
    workdir: str,
    repo_url: str,
    auto_shutdown: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_api_key: str | None,
    b2_key_id: str | None,
    b2_app_key: str | None,
    b2_bucket: str | None,
    b2_prefix: str | None,
    b2_s3_endpoint: str | None,
    b2_s3_region: str | None,
    install_mode: str,
    bootstrap_mode: str,
) -> Path:
    ONSTART_DIR.mkdir(exist_ok=True)
    script_path = ONSTART_DIR / "onstart.sh"
    combined_args = " ".join(part for part in (overrides, script_args) if part)
    if bootstrap_mode == "tracked-script":
        script_args_b64 = base64.b64encode(combined_args.encode("utf-8")).decode("ascii")
        repo_ref = git_ref or "main"
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "export DEBIAN_FRONTEND=noninteractive",
            f"export UPS_REPO_URL={shlex.quote(repo_url)}",
            f"export UPS_GIT_REF={shlex.quote(repo_ref)}",
            f"export UPS_WORKDIR={shlex.quote(workdir)}",
            f"export UPS_REMOTE_SCRIPT={shlex.quote(remote_script)}",
            f"export UPS_SCRIPT_ARGS_B64={shlex.quote(script_args_b64)}",
            f"export UPS_SKIP_PREFETCH={'1' if skip_prefetch else '0'}",
            f"export UPS_AUTO_SHUTDOWN={'1' if auto_shutdown else '0'}",
            f"export UPS_INSTALL_MODE={shlex.quote(install_mode)}",
            'export UPS_BOOTSTRAP_PATH="${UPS_BOOTSTRAP_PATH:-/tmp/ups_vast_remote_bootstrap.sh}"',
            "\"${PYTHON_BIN:-$(command -v python3 || command -v python)}\" - <<'PY'",
            "from pathlib import Path",
            "import os",
            "import urllib.parse",
            "import urllib.request",
            "",
            'repo_url = os.environ["UPS_REPO_URL"].removesuffix(".git")',
            'marker = "github.com/"',
            "if repo_url.startswith('git@github.com:'):",
            "    repo_path = repo_url.split(':', 1)[1]",
            "elif marker in repo_url:",
            "    repo_path = repo_url.split(marker, 1)[1]",
            "else:",
            "    raise SystemExit('tracked-script bootstrap requires a GitHub repository URL')",
            "repo_path = repo_path.strip('/')",
            'ref = urllib.parse.quote(os.environ.get("UPS_GIT_REF") or "main", safe="")',
            'bootstrap_url = f"https://raw.githubusercontent.com/{repo_path}/{ref}/scripts/vast_remote_bootstrap.sh"',
            'target = Path(os.environ["UPS_BOOTSTRAP_PATH"])',
            "target.parent.mkdir(parents=True, exist_ok=True)",
            "urllib.request.urlretrieve(bootstrap_url, target)",
            "PY",
            'chmod +x "$UPS_BOOTSTRAP_PATH"',
            'bash "$UPS_BOOTSTRAP_PATH"',
        ]
        script_path.write_text("\n".join(lines))
        script_path.chmod(0o755)
        return script_path
    if bootstrap_mode != "inline":
        raise SystemExit(f"Unsupported bootstrap mode: {bootstrap_mode}")
    datasets_export = (
        f'export WANDB_DATASETS="{datasets}"' if datasets else "# WANDB_DATASETS optional"
    )
    wandb_project_export = (
        f'export WANDB_PROJECT="{wandb_project}"' if wandb_project else "# WANDB_PROJECT optional"
    )
    wandb_entity_export = (
        f'export WANDB_ENTITY="{wandb_entity}"' if wandb_entity else "# WANDB_ENTITY optional"
    )
    wandb_api_key_export = (
        f'export WANDB_API_KEY="{wandb_api_key}"' if wandb_api_key else "# WANDB_API_KEY optional"
    )
    fetch_cmd = (
        "# Prefetch disabled; the remote script is responsible for hydration"
        if skip_prefetch
        else 'if [ -n "$WANDB_DATASETS" ]; then\n  bash scripts/fetch_datasets_b2.sh\nfi'
    )
    remote_cmd = "bash " + shlex.quote(remote_script)
    if combined_args:
        remote_cmd += " " + combined_args
    checkout_cmds = []
    if git_ref:
        quoted_ref = shlex.quote(git_ref)
        checkout_cmds = [
            "  git fetch --all --prune",
            f"  git checkout {quoted_ref} || git checkout -b {quoted_ref} origin/{quoted_ref}",
        ]
    repo_ref = git_ref or "main"
    if install_mode == "smoke":
        install_cmds = [
            '"$PYTHON_BIN" -m pip install --upgrade pip',
            '"$PYTHON_BIN" -m pip install -e . --no-deps',
            '"$PYTHON_BIN" -m pip install h5py numpy PyYAML',
        ]
    elif install_mode == "experiment":
        install_cmds = [
            '"$PYTHON_BIN" -m pip install --upgrade pip',
            '"$PYTHON_BIN" -m pip install -e . --no-deps',
            '"$PYTHON_BIN" -m pip install h5py numpy PyYAML matplotlib wandb',
        ]
    elif install_mode == "full":
        install_cmds = [
            '"$PYTHON_BIN" -m pip install --upgrade pip',
            '"$PYTHON_BIN" -m pip install -e .[dev]',
        ]
    else:
        raise SystemExit(f"Unsupported install mode: {install_mode}")
    shutdown_cmd = (
        "\nif command -v poweroff >/dev/null 2>&1; then\n  sync\n  poweroff\nfi"
        if auto_shutdown
        else ""
    )
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        'PYTHON_BIN="$(command -v python3 || command -v python || true)"',
        'if [ -z "$PYTHON_BIN" ]; then',
        '  echo "python is required in the remote image; choose an image with python/pip" >&2',
        "  exit 1",
        "fi",
        'if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then',
        '  echo "python pip is required in the remote image; choose an image with pip" >&2',
        "  exit 1",
        "fi",
        "if ! command -v rclone >/dev/null 2>&1; then",
        "  \"$PYTHON_BIN\" - <<'PY'",
        "from pathlib import Path",
        "import os",
        "import shutil",
        "import stat",
        "import tempfile",
        "import urllib.request",
        "import zipfile",
        "",
        'url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"',
        "with tempfile.TemporaryDirectory() as tmp:",
        "    tmp_path = Path(tmp)",
        '    archive = tmp_path / "rclone.zip"',
        "    urllib.request.urlretrieve(url, archive)",
        "    with zipfile.ZipFile(archive) as zf:",
        "        zf.extractall(tmp_path)",
        "    rclone = next(tmp_path.glob('rclone-*-linux-amd64/rclone'))",
        '    target = Path("/usr/local/bin/rclone")',
        "    shutil.copy2(rclone, target)",
        "    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)",
        "PY",
        "fi",
        "",
        f"export UPS_REPO_URL={shlex.quote(repo_url)}",
        f"export UPS_GIT_REF={shlex.quote(repo_ref)}",
        f"mkdir -p {workdir}",
        f"cd {workdir}",
        "",
        "if [ ! -d universal_simulator ]; then",
        "  if command -v git >/dev/null 2>&1; then",
        '    git clone "$UPS_REPO_URL" universal_simulator',
        "  else",
        "    \"$PYTHON_BIN\" - <<'PY'",
        "from pathlib import Path",
        "import os",
        "import shutil",
        "import tempfile",
        "import urllib.parse",
        "import urllib.request",
        "import zipfile",
        "",
        'repo_url = os.environ["UPS_REPO_URL"].removesuffix(".git")',
        'marker = "github.com/"',
        "if repo_url.startswith('git@github.com:'):",
        "    repo_path = repo_url.split(':', 1)[1]",
        "elif marker in repo_url:",
        "    repo_path = repo_url.split(marker, 1)[1]",
        "else:",
        "    raise SystemExit('git unavailable and repo is not a GitHub URL')",
        "repo_path = repo_path.strip('/')",
        'ref = os.environ.get("UPS_GIT_REF") or "main"',
        "quoted_ref = urllib.parse.quote(ref, safe='/')",
        'zip_url = f"https://codeload.github.com/{repo_path}/zip/refs/heads/{quoted_ref}"',
        "with tempfile.TemporaryDirectory() as tmp:",
        "    tmp_path = Path(tmp)",
        '    archive = tmp_path / "repo.zip"',
        "    urllib.request.urlretrieve(zip_url, archive)",
        "    with zipfile.ZipFile(archive) as zf:",
        "        zf.extractall(tmp_path)",
        "    roots = [p for p in tmp_path.iterdir() if p.is_dir()]",
        "    if not roots:",
        "        raise SystemExit('downloaded repo archive was empty')",
        "    shutil.move(str(roots[0]), 'universal_simulator')",
        "PY",
        "  fi",
        "fi",
        "cd universal_simulator",
        "if command -v git >/dev/null 2>&1; then",
        *checkout_cmds,
        "  git pull --ff-only || git pull",
        "else",
        '  echo "git unavailable; using downloaded repo archive for $UPS_GIT_REF"',
        "fi",
        "",
        *install_cmds,
        "",
        datasets_export,
        (f'export B2_KEY_ID="{b2_key_id}"' if b2_key_id else "# B2_KEY_ID optional"),
        (f'export B2_APP_KEY="{b2_app_key}"' if b2_app_key else "# B2_APP_KEY optional"),
        (f'export B2_BUCKET="{b2_bucket}"' if b2_bucket else "# B2_BUCKET optional"),
        (f'export B2_PREFIX="{b2_prefix}"' if b2_prefix else "# B2_PREFIX optional"),
        (
            f'export B2_S3_ENDPOINT="{b2_s3_endpoint}"'
            if b2_s3_endpoint
            else "# B2_S3_ENDPOINT optional"
        ),
        (f'export B2_S3_REGION="{b2_s3_region}"' if b2_s3_region else "# B2_S3_REGION optional"),
        wandb_project_export,
        wandb_entity_export,
        wandb_api_key_export,
        "if [ -f scripts/load_env.sh ] && [ -f .env ]; then",
        "  bash scripts/load_env.sh || true",
        "fi",
        fetch_cmd,
        f"{remote_cmd}{shutdown_cmd}",
    ]
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return script_path


def cmd_set_key(args: argparse.Namespace) -> None:
    key = args.key or os.environ.get("VAST_KEY") or os.environ.get("VAST_API_KEY")
    if not key:
        raise SystemExit("Provide --key or set VAST_KEY / VAST_API_KEY in the environment.")
    run(["vastai", "set", "api-key", key])


def cmd_search(args: argparse.Namespace) -> None:
    cmd = ["vastai", "search", "offers"]
    cmd.extend(args.filters)
    run(cmd, display_cmd=_redact_command(cmd))


def cmd_launch(args: argparse.Namespace) -> None:
    repo_url = args.repo_url or git_remote_url()
    dry_run = bool(args.dry_run)
    onstart = ensure_onstart(
        args.datasets,
        args.overrides,
        args.remote_script,
        args.script_args,
        args.skip_prefetch,
        args.git_ref,
        args.workdir,
        repo_url,
        args.auto_shutdown,
        args.wandb_project,
        args.wandb_entity,
        _secret_for_dry_run(args.wandb_api_key) if dry_run else args.wandb_api_key,
        _secret_for_dry_run(args.b2_key_id) if dry_run else args.b2_key_id,
        _secret_for_dry_run(args.b2_app_key) if dry_run else args.b2_app_key,
        args.b2_bucket,
        args.b2_prefix,
        args.b2_s3_endpoint,
        args.b2_s3_region,
        args.install_mode,
        args.bootstrap_mode,
    )

    env_parts = []
    if args.wandb_project:
        env_parts.append(f"-e WANDB_PROJECT={args.wandb_project}")
    if args.wandb_entity:
        env_parts.append(f"-e WANDB_ENTITY={args.wandb_entity}")
    if args.wandb_api_key:
        env_parts.append(f"-e WANDB_API_KEY={args.wandb_api_key}")
    if args.b2_key_id:
        env_parts.append(f"-e B2_KEY_ID={args.b2_key_id}")
    if args.b2_app_key:
        env_parts.append(f"-e B2_APP_KEY={args.b2_app_key}")
    if args.b2_bucket:
        env_parts.append(f"-e B2_BUCKET={args.b2_bucket}")
    if args.b2_prefix:
        env_parts.append(f"-e B2_PREFIX={args.b2_prefix}")
    if args.b2_s3_endpoint:
        env_parts.append(f"-e B2_S3_ENDPOINT={args.b2_s3_endpoint}")
    if args.b2_s3_region:
        env_parts.append(f"-e B2_S3_REGION={args.b2_s3_region}")
    env_str = " ".join(env_parts) if env_parts else None

    search_cmd: list[str] | None = None
    if args.offer_id:
        offer_token = str(args.offer_id)
    else:
        # The 'vastai launch instance' endpoint rejects requests on current
        # CLI/API versions, so resolve the cheapest matching offer explicitly
        # and use 'vastai create instance' for both paths.
        query = f"gpu_name={args.gpu} num_gpus={args.num_gpus} rentable=true verified=true"
        if args.region:
            query += f" geolocation={args.region}"
        search_cmd = [
            "vastai",
            "search",
            "offers",
            query,
            "-o",
            args.order or "dph_total",
            "--limit",
            str(args.limit or 10),
            "--raw",
        ]
        offer_token = "<cheapest-offer-from-search>"
    cmd = [
        "vastai",
        "create",
        "instance",
        offer_token,
        "--image",
        args.image,
        "--disk",
        str(args.disk),
    ]
    if env_str:
        cmd.extend(["--env", env_str])
    if args.args_mode:
        cmd.extend(["--entrypoint", "bash", "--args", "-lc", onstart.read_text()])
    else:
        if not args.no_ssh:
            cmd.append("--ssh")
        cmd.extend(["--onstart", str(onstart)])
    if args.dry_run:
        if search_cmd is not None:
            print("DRY RUN: would resolve offer via ->", " ".join(search_cmd))
        print("DRY RUN: would execute ->", " ".join(_redact_command(cmd)))
        print("\nGenerated onstart script:\n" + onstart.read_text())
        return
    if not args.skip_launch_preflight and not preflight_vast_dns(
        retries=args.launch_retries,
        retry_backoff=args.launch_retry_backoff,
    ):
        raise SystemExit(
            f"Vast API DNS preflight failed for {VAST_API_HOST}; "
            "not attempting paid instance creation."
        )
    if search_cmd is not None:
        search_out = subprocess.check_output(search_cmd, cwd=REPO_ROOT)
        offers = json.loads(search_out.decode() or "[]")
        if not offers:
            raise SystemExit(f"No rentable Vast offers matched: {search_cmd[3]}")
        resolved_offer = str(offers[0]["id"])
        print(
            f"Resolved cheapest offer {resolved_offer} "
            f"(${offers[0].get('dph_total', '?')}/hr, {offers[0].get('gpu_name', '?')})"
        )
        cmd[cmd.index("<cheapest-offer-from-search>")] = resolved_offer
    run(
        cmd,
        display_cmd=_redact_command(cmd),
        retries=args.launch_retries,
        retry_backoff=args.launch_retry_backoff,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for Vast.ai training launches")
    sub = parser.add_subparsers(dest="command", required=True)

    p_key = sub.add_parser("set-key", help="Set Vast API key (reads VAST_KEY if not provided)")
    p_key.add_argument("--key", help="API key literal")
    p_key.set_defaults(func=cmd_set_key)

    p_search = sub.add_parser("search", help="Wrapper around 'vastai search offers'")
    p_search.add_argument("filters", nargs=argparse.REMAINDER, help="Filters and flags to append")
    p_search.set_defaults(func=cmd_search)

    p_launch = sub.add_parser("launch", help="Launch instance and run training on Vast")
    p_launch.add_argument("--gpu", default="RTX_4090", help="GPU model (default RTX_4090)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    p_launch.add_argument(
        "--image", default="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", help="Docker image"
    )
    p_launch.add_argument("--disk", type=int, default=64, help="Disk in GB")
    p_launch.add_argument("--region", help="Region filter for launch instance")
    p_launch.add_argument("--order", help="Vast launch offer ordering, e.g. dph_total")
    p_launch.add_argument("--limit", type=int, help="Limit launch search candidates")
    p_launch.add_argument(
        "--offer-id",
        help=(
            "Explicit Vast offer ID from search results. Uses 'vastai create instance' "
            "instead of an implicit launch search."
        ),
    )
    p_launch.add_argument(
        "--datasets",
        default=os.environ.get("WANDB_DATASETS"),
        help="WANDB_DATASETS value to pass to training",
    )
    p_launch.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT"), help="WANDB project name"
    )
    p_launch.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="WANDB entity name"
    )
    p_launch.add_argument(
        "--wandb-api-key", default=os.environ.get("WANDB_API_KEY"), help="WANDB API key"
    )
    p_launch.add_argument(
        "--overrides", help="Legacy extra arguments to append to the remote script"
    )
    p_launch.add_argument(
        "--remote-script",
        default="scripts/run_remote_scale.sh",
        help="Remote script to run after setup",
    )
    p_launch.add_argument("--script-args", help="Additional arguments appended after --overrides")
    p_launch.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="Skip onstart dataset prefetch; useful when the remote script fetches data",
    )
    p_launch.add_argument("--git-ref", help="Git branch, tag, or ref to checkout before running")
    p_launch.add_argument("--repo-url", help="Git remote URL (defaults to origin)")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_launch.add_argument(
        "--auto-shutdown", action="store_true", help="Power off instance after training completes"
    )
    p_launch.add_argument(
        "--no-ssh",
        action="store_true",
        help=(
            "Do not request Vast SSH runtime injection. Useful for one-shot jobs when "
            "SSH bootstrap stalls on apt mirrors."
        ),
    )
    p_launch.add_argument(
        "--args-mode",
        action="store_true",
        help=(
            "Run the generated script with 'bash -lc' via Vast --args instead of --onstart. "
            "This avoids Vast SSH/Jupyter/onstart bootstrap for one-shot jobs."
        ),
    )
    p_launch.add_argument(
        "--install-mode",
        choices=("full", "smoke", "experiment"),
        default="full",
        help=(
            "Remote dependency install profile. 'smoke' avoids full Torch/CUDA dev deps; "
            "'experiment' adds lightweight plotting/eval deps without upgrading Torch."
        ),
    )
    p_launch.add_argument(
        "--bootstrap-mode",
        choices=("inline", "tracked-script"),
        default="inline",
        help=(
            "How to build the Vast startup payload. 'tracked-script' keeps the onstart "
            "script small by downloading scripts/vast_remote_bootstrap.sh from the "
            "requested git ref."
        ),
    )
    p_launch.add_argument(
        "--launch-retries",
        type=int,
        default=0,
        help="Retry transient Vast CLI launch/create failures this many times",
    )
    p_launch.add_argument(
        "--launch-retry-backoff",
        type=float,
        default=5.0,
        help="Base seconds to sleep between transient Vast CLI launch/create retries",
    )
    p_launch.add_argument(
        "--skip-launch-preflight",
        action="store_true",
        help="Skip DNS preflight before a paid Vast launch/create request",
    )
    p_launch.add_argument(
        "--b2-key-id",
        default=os.environ.get("B2_KEY_ID"),
        help="B2 application key ID for dataset fetch",
    )
    p_launch.add_argument(
        "--b2-app-key",
        default=os.environ.get("B2_APP_KEY"),
        help="B2 application key secret for dataset fetch",
    )
    p_launch.add_argument(
        "--b2-bucket",
        default=os.environ.get("B2_BUCKET"),
        help="Override B2 bucket for dataset fetch",
    )
    p_launch.add_argument(
        "--b2-prefix",
        default=os.environ.get("B2_PREFIX"),
        help="Override B2 prefix for dataset fetch",
    )
    p_launch.add_argument(
        "--b2-s3-endpoint",
        default=os.environ.get("B2_S3_ENDPOINT"),
        help="Override B2 S3 endpoint for dataset fetch",
    )
    p_launch.add_argument(
        "--b2-s3-region",
        default=os.environ.get("B2_S3_REGION"),
        help="Override B2 S3 region for dataset fetch",
    )
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.set_defaults(func=cmd_launch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
