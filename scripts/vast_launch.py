#!/usr/bin/env python
"""Helper utilities for launching Vast.ai training runs."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ONSTART_DIR = REPO_ROOT / ".vast"
REDACTED = "<redacted>"


def run(cmd: list[str], *, check: bool = True, display_cmd: list[str] | None = None) -> int:
    shown = display_cmd if display_cmd is not None else cmd
    print("$", " ".join(shlex.quote(part) for part in shown))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(_redact_text(result.stdout), end="")
    if result.stderr:
        print(_redact_text(result.stderr), end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


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
) -> Path:
    ONSTART_DIR.mkdir(exist_ok=True)
    script_path = ONSTART_DIR / "onstart.sh"
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
    combined_args = " ".join(part for part in (overrides, script_args) if part)
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

    if args.offer_id:
        cmd = [
            "vastai",
            "create",
            "instance",
            str(args.offer_id),
            "--image",
            args.image,
            "--disk",
            str(args.disk),
        ]
    else:
        cmd = [
            "vastai",
            "launch",
            "instance",
            "-g",
            args.gpu,
            "-n",
            str(args.num_gpus),
            "-i",
            args.image,
            "-d",
            str(args.disk),
        ]
        if args.region:
            cmd.extend(["-r", args.region])
        if args.order:
            cmd.extend(["-o", args.order])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
    if env_str:
        cmd.extend(["--env", env_str])
    if args.args_mode:
        cmd.extend(["--entrypoint", "bash", "--args", "-lc", onstart.read_text()])
    else:
        if not args.no_ssh:
            cmd.append("--ssh")
        cmd.extend(["--onstart", str(onstart)])
    if args.dry_run:
        print("DRY RUN: would execute ->", " ".join(_redact_command(cmd)))
        print("\nGenerated onstart script:\n" + onstart.read_text())
        return
    run(cmd, display_cmd=_redact_command(cmd))


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
