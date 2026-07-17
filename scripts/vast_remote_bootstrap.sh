#!/usr/bin/env bash
set -euo pipefail

: "${UPS_REPO_URL:?UPS_REPO_URL is required}"
: "${UPS_GIT_REF:=main}"
: "${UPS_WORKDIR:=/workspace}"
: "${UPS_REMOTE_SCRIPT:?UPS_REMOTE_SCRIPT is required}"
: "${UPS_SKIP_PREFETCH:=1}"
: "${UPS_SKIP_RCLONE_INSTALL:=0}"
: "${UPS_AUTO_SHUTDOWN:=0}"
: "${UPS_INSTALL_MODE:=experiment}"
: "${UPS_SCRIPT_ARGS_B64:=}"
: "${UPS_EXIT_SENTINEL:=$UPS_WORKDIR/.ups_remote_bootstrap_exit_status}"
: "${UPS_MAX_RUNTIME_SECONDS:=0}"

# The local managed watchdog uses this marker to distinguish a provider that
# never started the tracked bootstrap from a remote experiment failure.
echo "REMOTE_BOOTSTRAP_STARTED=1"

stop_container() {
  sync || true
  if command -v poweroff >/dev/null 2>&1 && poweroff; then
    return 0
  fi
  # Vast images commonly expose a systemd-dependent poweroff binary while the
  # container itself does not run systemd. Stopping PID 1 is the portable
  # container fallback and prevents an exited bootstrap from continuing to bill.
  kill -TERM 1 2>/dev/null || true
}

if [ "$UPS_AUTO_SHUTDOWN" = "1" ] && [ -f "$UPS_EXIT_SENTINEL" ]; then
  previous_status="$(cat "$UPS_EXIT_SENTINEL" 2>/dev/null || echo unknown)"
  echo "REMOTE_BOOTSTRAP_ALREADY_RAN previous_status=${previous_status}"
  stop_container
  exit 0
fi

shutdown_on_exit() {
  local status=$?
  if [ -n "${REMOTE_DEADLINE_PID:-}" ]; then
    kill "$REMOTE_DEADLINE_PID" 2>/dev/null || true
  fi
  echo "REMOTE_BOOTSTRAP_EXIT_STATUS=${status}"
  mkdir -p "$(dirname "$UPS_EXIT_SENTINEL")"
  echo "$status" > "$UPS_EXIT_SENTINEL" || true
  stop_container
  exit "$status"
}

if [ "$UPS_AUTO_SHUTDOWN" = "1" ]; then
  trap shutdown_on_exit EXIT
fi

if [ "$UPS_MAX_RUNTIME_SECONDS" -gt 0 ]; then
  (
    sleep "$UPS_MAX_RUNTIME_SECONDS"
    echo "REMOTE_MAX_RUNTIME_REACHED seconds=${UPS_MAX_RUNTIME_SECONDS}" >&2
    stop_container
  ) &
  REMOTE_DEADLINE_PID=$!
fi

export DEBIAN_FRONTEND=noninteractive
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "python is required in the remote image; choose an image with python/pip" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "python pip is required in the remote image; choose an image with pip" >&2
  exit 1
fi

if [ "$UPS_SKIP_RCLONE_INSTALL" != "1" ] && ! command -v rclone >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  if ! (apt-get update -qq && apt-get install -y -qq rclone); then
    echo "apt rclone install failed; trying the bounded upstream archive fallback" >&2
  fi
fi

if [ "$UPS_SKIP_RCLONE_INSTALL" != "1" ] && ! command -v rclone >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    archive = tmp_path / "rclone.zip"
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as sink:
        shutil.copyfileobj(response, sink)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp_path)
    rclone = next(tmp_path.glob("rclone-*-linux-amd64/rclone"))
    target = Path("/usr/local/bin/rclone")
    shutil.copy2(rclone, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
fi

mkdir -p "$UPS_WORKDIR"
cd "$UPS_WORKDIR"

if [ ! -d universal_simulator ]; then
  if command -v git >/dev/null 2>&1; then
    git clone "$UPS_REPO_URL" universal_simulator
  else
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

repo_url = os.environ["UPS_REPO_URL"].removesuffix(".git")
marker = "github.com/"
if repo_url.startswith("git@github.com:"):
    repo_path = repo_url.split(":", 1)[1]
elif marker in repo_url:
    repo_path = repo_url.split(marker, 1)[1]
else:
    raise SystemExit("git unavailable and repo is not a GitHub URL")
repo_path = repo_path.strip("/")
ref = os.environ.get("UPS_GIT_REF") or "main"
quoted_ref = urllib.parse.quote(ref, safe="/")
zip_url = f"https://codeload.github.com/{repo_path}/zip/refs/heads/{quoted_ref}"
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    archive = tmp_path / "repo.zip"
    urllib.request.urlretrieve(zip_url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp_path)
    roots = [p for p in tmp_path.iterdir() if p.is_dir()]
    if not roots:
        raise SystemExit("downloaded repo archive was empty")
    shutil.move(str(roots[0]), "universal_simulator")
PY
  fi
fi

cd universal_simulator
if command -v git >/dev/null 2>&1; then
  if [ ! -d .git ]; then
    cd ..
    rm -rf universal_simulator
    git clone "$UPS_REPO_URL" universal_simulator
    cd universal_simulator
  fi
  git fetch --all --prune
  if [[ "$UPS_GIT_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
    git checkout --detach "$UPS_GIT_REF"
  else
    git checkout "$UPS_GIT_REF" || git checkout -b "$UPS_GIT_REF" "origin/$UPS_GIT_REF"
    git pull --ff-only || git pull
  fi
else
  echo "git unavailable; using downloaded repo archive for $UPS_GIT_REF"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
case "$UPS_INSTALL_MODE" in
  smoke)
    "$PYTHON_BIN" -m pip install -e . --no-deps
    "$PYTHON_BIN" -m pip install h5py numpy PyYAML
    ;;
  experiment)
    "$PYTHON_BIN" -m pip install -e . --no-deps
    "$PYTHON_BIN" -m pip install h5py numpy PyYAML matplotlib wandb
    ;;
  full)
    "$PYTHON_BIN" -m pip install -e .[dev]
    ;;
  *)
    echo "Unsupported install mode: $UPS_INSTALL_MODE" >&2
    exit 2
    ;;
esac

if [ -f scripts/load_env.sh ] && [ -f .env ]; then
  bash scripts/load_env.sh || true
fi

if [ "$UPS_SKIP_PREFETCH" != "1" ] && [ -n "${DATA_LOCK:-}" ]; then
  bash scripts/fetch_datasets_b2.sh
fi

SCRIPT_ARGS="$("$PYTHON_BIN" - <<'PY'
import base64
import os

raw = os.environ.get("UPS_SCRIPT_ARGS_B64") or ""
print(base64.b64decode(raw).decode("utf-8") if raw else "")
PY
)"

remote_cmd="bash $(printf '%q' "$UPS_REMOTE_SCRIPT")"
if [ -n "$SCRIPT_ARGS" ]; then
  remote_cmd="$remote_cmd $SCRIPT_ARGS"
fi
bash -lc "$remote_cmd"
