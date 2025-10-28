#!/usr/bin/env python3
"""
Provision and manage Vultr GPU instances for Universal Simulator training jobs.

Commands:
  setup-env   Validate local credentials and print export instructions
  plans       List available GPU plans (optionally filtered by region)
  launch      Provision instance + block storage, run bootstrap script
  status      Show instance status and public IP
  logs        Fetch boot log metadata or print SSH instructions
  teardown    Destroy instance and optionally delete block storage

All API calls require `VULTR_KEY` (preferably stored in .env) and the
originating IP must be allow-listed in the Vultr control panel.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cloud_launch import (
    BootstrapConfig,
    CloudLaunchError,
    LaunchSpec,
    VolumeSpec,
)

DEFAULT_VOLUME_MOUNT = Path("/mnt/cache")
PREFERRED_REGIONS = ("sjc", "sea", "lax")
DEFAULT_OS_ID = 1743  # Ubuntu 22.04 LTS x64
VULTR_API_BASE = "https://api.vultr.com/v2"


def load_env() -> dict[str, str]:
    """Parse .env-style key/value pairs from repository root."""
    env_path = REPO_ROOT / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_key() -> str:
    """Return the Vultr API key from env or .env file."""
    env_key = os.environ.get("VULTR_KEY")
    if env_key:
        return env_key
    env_file_values = load_env()
    key = env_file_values.get("VULTR_KEY")
    if key:
        return key
    raise CloudLaunchError(
        "Missing VULTR_KEY. Add it to .env or export it in your shell before running this command."
    )


class VultrProvider:
    """Minimal HTTP client for the Vultr v2 API."""

    name = "vultr"

    def __init__(self, api_key: str, *, base_url: str = VULTR_API_BASE) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    # -------------------- HTTP helpers -------------------- #

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=60) as response:  # nosec - external API call
                if response.status == 204:
                    return None
                body = response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network errors
            msg = exc.read().decode("utf-8", errors="ignore")
            raise CloudLaunchError(
                f"Vultr API {method.upper()} {path} failed ({exc.code}): {msg or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network errors
            raise CloudLaunchError(f"Network error calling Vultr API: {exc}") from exc

        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CloudLaunchError(f"Unexpected non-JSON response from Vultr API: {body!r}") from exc

    # -------------------- CloudProvider interface -------------------- #

    def list_offers(self, *, region: str | None = None) -> Iterable[LaunchSpec]:
        response = None
        for plan_type in ("vcg", "gpu", None):
            params = {"type": plan_type} if plan_type else {}
            try:
                response = self._request("GET", "/plans", params=params)
            except CloudLaunchError as exc:
                if "valid type" in str(exc).lower() and plan_type != "vcg":
                    continue
                raise
            else:
                break
        if response is None:
            raise CloudLaunchError("Unable to retrieve Vultr plans.")
        plans = response.get("plans", []) if isinstance(response, dict) else []
        for plan in plans:
            locations: list[str] = plan.get("locations", []) or []
            monthly = float(plan.get("monthly_cost") or plan.get("monthly_price") or 0.0)
            hourly = float(plan.get("hourly_price") or 0.0)
            if not hourly and monthly:
                hourly = monthly / (30 * 24)

            gpu_ram_gb = (
                plan.get("gpu_vram_gb")
                or plan.get("gpu_ram")
                or plan.get("gpu_memory_gb")
                or 0
            )
            if isinstance(gpu_ram_gb, str):
                try:
                    gpu_ram_gb = float(gpu_ram_gb)
                except ValueError:
                    gpu_ram_gb = 0
            vcpus = int(plan.get("vcpu_count") or plan.get("vcpu") or 0)

            if region:
                if region not in locations:
                    continue
                yield LaunchSpec(
                    plan_id=plan.get("id"),
                    region=region,
                    os_id=DEFAULT_OS_ID,
                    label=f"{plan.get('id')}-{region}",
                    gpu_ram_gb=int(gpu_ram_gb),
                    vcpus=vcpus,
                    hourly_cost=hourly,
                    extra={"raw": plan},
                )
            else:
                for loc in locations:
                    yield LaunchSpec(
                        plan_id=plan.get("id"),
                        region=loc,
                        os_id=DEFAULT_OS_ID,
                        label=f"{plan.get('id')}-{loc}",
                        gpu_ram_gb=int(gpu_ram_gb),
                        vcpus=vcpus,
                        hourly_cost=hourly,
                        extra={"raw": plan},
                    )

    def provision_volume(self, spec: VolumeSpec) -> str:
        payload = {
            "region": spec.extra.get("region") or spec.extra.get("region_id"),
            "size_gb": spec.size_gb,
            "label": spec.label,
        }
        response = self._request("POST", "/blocks", payload=payload)
        block = response.get("block")
        if not block or "id" not in block:
            raise CloudLaunchError(f"Unexpected response while creating block volume: {response}")
        return block["id"]

    def wait_for_volume(self, volume_id: str, *, timeout: float = 600.0) -> None:
        start = time.monotonic()
        while True:
            response = self._request("GET", f"/blocks/{volume_id}")
            block = response.get("block", {})
            status = block.get("status")
            if status == "active":
                return
            if time.monotonic() - start > timeout:
                raise CloudLaunchError(
                    f"Timed out waiting for block volume {volume_id} to become active (status={status})"
                )
            time.sleep(5)

    def create_instance(self, spec: LaunchSpec, *, user_data: str | None = None) -> str:
        payload: dict[str, Any] = {
            "plan": spec.plan_id,
            "region": spec.region,
            "os_id": spec.os_id,
            "label": spec.label,
        }
        if user_data:
            payload["user_data"] = user_data
        response = self._request("POST", "/instances", payload=payload)
        instance = response.get("instance")
        if not instance or "id" not in instance:
            raise CloudLaunchError(f"Unexpected response while creating instance: {response}")
        return instance["id"]

    def wait_for_instance(self, instance_id: str, *, timeout: float = 900.0) -> None:
        start = time.monotonic()
        while True:
            response = self._request("GET", f"/instances/{instance_id}")
            instance = response.get("instance", {})
            status = instance.get("status")
            power = instance.get("power_status")
            if status == "active" and power == "running":
                return
            if time.monotonic() - start > timeout:
                raise CloudLaunchError(
                    f"Timed out waiting for instance {instance_id} (status={status}, power={power})"
                )
            time.sleep(10)

    def attach_volume(self, instance_id: str, volume_id: str) -> None:
        payload = {"instance_id": instance_id}
        self._request("POST", f"/blocks/{volume_id}/attach", payload=payload)

    def run_bootstrap(self, instance_id: str, script: str) -> None:  # pragma: no cover - placeholder
        # Bootstrap runs via user_data; no additional action required.
        _ = (instance_id, script)

    def destroy_instance(self, instance_id: str) -> None:
        self._request("DELETE", f"/instances/{instance_id}")

    def detach_volume(self, instance_id: str, volume_id: str) -> None:
        payload = {"instance_id": instance_id}
        self._request("POST", f"/blocks/{volume_id}/detach", payload=payload)

    def delete_volume(self, volume_id: str) -> None:
        self._request("DELETE", f"/blocks/{volume_id}")

    # -------------------- Extra helpers -------------------- #

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/instances/{instance_id}")
        return response.get("instance", {})

    def list_regions(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/regions", params={"availability": "gpu"})
        return response.get("regions", [])


def preferred_region(regions: Iterable[str]) -> str | None:
    for pref in PREFERRED_REGIONS:
        if pref in regions:
            return pref
    return next(iter(regions), None)


REQUIRED_ENV_KEYS = [
    "B2_KEY_ID",
    "B2_APP_KEY",
    "B2_S3_ENDPOINT",
    "B2_S3_REGION",
    "B2_BUCKET",
]

OPTIONAL_ENV_KEYS = [
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
]


def collect_env_exports() -> dict[str, str]:
    values = load_env()

    def get_value(key: str, required: bool = False) -> str:
        val = os.environ.get(key) or values.get(key)
        if required and not val:
            raise CloudLaunchError(
                f"Missing required credential {key}. Add it to your shell environment or .env file."
            )
        return val or ""

    exports: dict[str, str] = {}
    for key in REQUIRED_ENV_KEYS:
        exports[key] = get_value(key, required=True)
    for key in OPTIONAL_ENV_KEYS:
        val = get_value(key, required=False)
        if val:
            exports[key] = val
    return exports


def env_assignment_str(pairs: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in pairs.items())


def build_bootstrap_script(config: BootstrapConfig, *, mount_device: str = "/dev/vdb") -> str:
    config_path = str(config.config_path)
    repo_url = config.repo_url
    branch = config.branch
    workdir = config.workdir.as_posix()
    env_exports = dict(config.env_exports)
    wandb_env = {
        key: env_exports[key]
        for key in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY")
        if key in env_exports and env_exports[key]
    }
    b2_env = {
        key: env_exports[key]
        for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_S3_ENDPOINT", "B2_S3_REGION", "B2_BUCKET")
        if key in env_exports and env_exports[key]
    }

    profile_entries = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in env_exports.items() if value
    ) or "true"
    profile_cmd = (
        "cat > /etc/profile.d/ups_env.sh <<'EOF_ENV'\n"
        f"{profile_entries}\n"
        "EOF_ENV"
    )

    data_root = f"{DEFAULT_VOLUME_MOUNT}/data"
    rclone_cmd = "true"
    sync_cmd = "true"
    if all(k in b2_env for k in ("B2_KEY_ID", "B2_APP_KEY", "B2_S3_ENDPOINT", "B2_S3_REGION", "B2_BUCKET")):
        rclone_config = textwrap.dedent(
            f"""
            [B2TRAIN]
            type = s3
            provider = Other
            access_key_id = {b2_env['B2_KEY_ID']}
            secret_access_key = {b2_env['B2_APP_KEY']}
            endpoint = {b2_env['B2_S3_ENDPOINT']}
            region = {b2_env['B2_S3_REGION']}
            acl = private
            no_check_bucket = true
            """
        ).strip()
        rclone_cmd = (
            "mkdir -p /root/.config/rclone && "
            "cat > /root/.config/rclone/rclone.conf <<'EOF_RCLONE'\n"
            f"{rclone_config}\n"
            "EOF_RCLONE"
        )
        rclone_env = {
            "RCLONE_CONFIG_B2TRAIN_TYPE": "s3",
            "RCLONE_CONFIG_B2TRAIN_PROVIDER": "Other",
            "RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID": b2_env["B2_KEY_ID"],
            "RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY": b2_env["B2_APP_KEY"],
            "RCLONE_CONFIG_B2TRAIN_ENDPOINT": b2_env["B2_S3_ENDPOINT"],
            "RCLONE_CONFIG_B2TRAIN_REGION": b2_env["B2_S3_REGION"],
            "RCLONE_CONFIG_B2TRAIN_ACL": "private",
            "RCLONE_CONFIG_B2TRAIN_NO_CHECK_BUCKET": "true",
        }
        bucket = b2_env["B2_BUCKET"].rstrip("/")
        sync_cmd = (
            f"{env_assignment_str(rclone_env)} rclone sync B2TRAIN:{bucket} {data_root} "
            "--fast-list --progress --create-empty-src-dirs || true"
        )

    cmd_parts = []
    if wandb_env:
        cmd_parts.append(env_assignment_str(wandb_env))
    cmd_parts = [
        *cmd_parts,
        "WANDB_MODE=online",
        "python",
        "scripts/run_fast_to_sota.py",
        "--train-config",
        shlex.quote(config_path),
        "--train-stage",
        shlex.quote(config.stage),
        "--skip-small-eval",
        "--eval-device",
        "cuda",
        "--run-dir",
        "artifacts/runs",
        "--leaderboard-csv",
        "reports/leaderboard.csv",
        "--wandb-mode",
        "online",
        "--wandb-sync",
    ]
    for tag in config.wandb_tags + [f"environment={config.environment_tag}"]:
        cmd_parts.extend(["--wandb-tags", shlex.quote(tag)])
    for arg in config.run_args:
        cmd_parts.append(shlex.quote(arg))
    run_command = " ".join(cmd_parts)
    script = textwrap.dedent(
        f"""\
        #cloud-config
        runcmd:
          - [ bash, -lc, "set -euo pipefail" ]
          - [ bash, -lc, "export DEBIAN_FRONTEND=noninteractive" ]
          - [ bash, -lc, "apt-get update" ]
          - [ bash, -lc, "apt-get install -y git build-essential python3 python3-pip python3-venv awscli rclone nvidia-driver-550 nvidia-cuda-toolkit || true" ]
          - [ bash, -lc, "{profile_cmd}" ]
          - [ bash, -lc, "mkdir -p {workdir}" ]
          - [ bash, -lc, "cd {workdir} && if [ ! -d universal_simulator ]; then git clone {repo_url} universal_simulator; fi" ]
          - [ bash, -lc, "cd {workdir}/universal_simulator && git fetch origin && git checkout {branch} && git pull origin {branch}" ]
          - [ bash, -lc, "python3 -m pip install --upgrade pip && python3 -m pip install -e .[dev]" ]
          - [ bash, -lc, "mkdir -p {DEFAULT_VOLUME_MOUNT}" ]
          - [ bash, -lc, "if [ -b {mount_device} ]; then if ! blkid {mount_device}; then mkfs.ext4 -F {mount_device}; fi; mount {mount_device} {DEFAULT_VOLUME_MOUNT}; grep -q {mount_device} /etc/fstab || echo '{mount_device} {DEFAULT_VOLUME_MOUNT} ext4 defaults,nofail 0 2' >> /etc/fstab; fi" ]
          - [ bash, -lc, "mkdir -p {DEFAULT_VOLUME_MOUNT}/data {DEFAULT_VOLUME_MOUNT}/checkpoints {DEFAULT_VOLUME_MOUNT}/artifacts" ]
          - [ bash, -lc, "{rclone_cmd}" ]
          - [ bash, -lc, "cd {workdir}/universal_simulator && mkdir -p reports artifacts/runs" ]
          - [ bash, -lc, "cd {workdir}/universal_simulator && ln -snf {DEFAULT_VOLUME_MOUNT}/data data && ln -snf {DEFAULT_VOLUME_MOUNT}/checkpoints checkpoints && ln -snf {DEFAULT_VOLUME_MOUNT}/artifacts artifacts" ]
          - [ bash, -lc, "cd {workdir}/universal_simulator && {sync_cmd}" ]
          - [ bash, -lc, "nvidia-smi || true" ]
          - [ bash, -lc, "dd if=/dev/zero of={DEFAULT_VOLUME_MOUNT}/.nvme_test bs=1M count=512 oflag=direct status=none && rm -f {DEFAULT_VOLUME_MOUNT}/.nvme_test" ]
          - [ bash, -lc, "find {DEFAULT_VOLUME_MOUNT}/data -maxdepth 2 -type f | head" ]
          - [ bash, -lc, "cd {workdir}/universal_simulator && {run_command}" ]
        """
    ).strip()
    return script


# -------------------- CLI commands -------------------- #

def cmd_setup_env(_: argparse.Namespace) -> None:
    values = load_env()
    key = values.get("VULTR_KEY")
    print("═══════════════════════════════════════════════")
    print("Vultr Launcher — Environment Setup")
    print("═══════════════════════════════════════════════")
    if key:
        print("✓ Found VULTR_KEY in .env (value hidden).")
        print("  Export it before running commands:\n")
        print("    export VULTR_KEY=$(cat .env | grep '^VULTR_KEY' | cut -d'=' -f2)")
    else:
        print("⚠️  No VULTR_KEY entry in .env.")
        print("   Add the key as `VULTR_KEY=...` to proceed.")
    print()
    print("Reminder: Update the Vultr API access list with this machine's IP address.")


def iter_plans(provider: VultrProvider, region: str | None) -> Iterator[LaunchSpec]:
    try:
        return iter(provider.list_offers(region=region))
    except CloudLaunchError as exc:
        raise SystemExit(str(exc)) from exc


def cmd_plans(args: argparse.Namespace) -> None:
    api = VultrProvider(resolve_key())
    specs = list(iter_plans(api, args.region))
    if not specs:
        print("No GPU plans available (check IP allow list or region filter).")
        return
    print(f"{'Plan':<18} {'Region':<8} {'GPU RAM (GB)':<12} {'vCPUs':<6} {'$/hr':<8}")
    print("-" * 60)
    for spec in specs:
        print(
            f"{spec.plan_id:<18} {spec.region:<8} "
            f"{spec.gpu_ram_gb:<12} {spec.vcpus:<6} {spec.hourly_cost:<8.3f}"
        )


@dataclass
class LaunchArgs:
    instance_label: str
    plan_id: str
    region: str
    os_id: int
    volume_size: int
    keep_volume: bool
    config: str
    stage: str
    repo_url: str | None
    branch: str | None
    workdir: str
    run_args: list[str]
    precompute: bool
    dry_run: bool
    wandb_tag: str
    env_exports: dict[str, str]


def generate_launch_args(
    args: argparse.Namespace, provider: VultrProvider, *, dry_run: bool = False
) -> LaunchArgs:
    branch = args.branch or auto_branch()
    repo_url = args.repo_url or auto_repo_url()
    region = args.region
    if not region:
        if dry_run:
            region = f"{PREFERRED_REGIONS[0]}1"
        else:
            specs = list(provider.list_offers())
            unique_regions = {spec.region for spec in specs}
            region = preferred_region(sorted(unique_regions))
            if not region:
                raise CloudLaunchError("Unable to auto-select region; specify --region.")

    plan_id = args.plan_id
    if not plan_id:
        if dry_run:
            plan_id = "vcg-placeholder"
        else:
            for spec in provider.list_offers(region=region):
                if spec.gpu_ram_gb >= args.min_gpu_ram:
                    plan_id = spec.plan_id
                    break
            if not plan_id:
                raise CloudLaunchError(
                    f"No plans in region {region!r} satisfy minimum GPU RAM requirement ({args.min_gpu_ram} GB)."
                )
    label = args.label or f"ups-{plan_id}-{region}"
    return LaunchArgs(
        instance_label=label,
        plan_id=plan_id,
        region=region,
        os_id=args.os_id,
        volume_size=args.volume_size,
        keep_volume=args.keep_volume,
        config=args.config,
        stage=args.stage,
        repo_url=repo_url,
        branch=branch,
        workdir=args.workdir,
        run_args=args.run_arg,
        precompute=not args.no_precompute,
        dry_run=dry_run,
        wandb_tag="vultr",
        env_exports=collect_env_exports(),
    )


def auto_branch() -> str:
    try:
        out = os.popen("git branch --show-current").read().strip()
        return out or "main"
    except Exception:  # pragma: no cover - best effort
        return "main"


def auto_repo_url() -> str:
    try:
        out = os.popen("git config --get remote.origin.url").read().strip()
        return out or "https://github.com/emgun/universal_simulator.git"
    except Exception:  # pragma: no cover - best effort
        return "https://github.com/emgun/universal_simulator.git"


def to_launch_spec(provider: VultrProvider, launch_args: LaunchArgs) -> LaunchSpec:
    specs = list(provider.list_offers(region=launch_args.region))
    for spec in specs:
        if spec.plan_id == launch_args.plan_id:
            return spec
    raise CloudLaunchError(
        f"Plan {launch_args.plan_id!r} not available in region {launch_args.region!r}. Use `vultr_launch.py plans` to inspect."
    )


def build_bootstrap_config(args: LaunchArgs) -> BootstrapConfig:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    try:
        rel_config = config_path.relative_to(REPO_ROOT)
    except ValueError:
        rel_config = config_path
    repo_url = args.repo_url or auto_repo_url()
    return BootstrapConfig(
        repo_url=repo_url,
        branch=args.branch or auto_branch(),
        workdir=Path(args.workdir),
        config_path=rel_config,
        stage=args.stage,
        run_args=args.run_args,
        precompute_latent_cache=args.precompute,
        wandb_tags=["fast-to-sota"],
        environment_tag=args.wandb_tag,
        env_exports=args.env_exports,
    )


def cmd_launch(args: argparse.Namespace) -> None:
    api_key = resolve_key()
    provider = VultrProvider(api_key)
    launch_args = generate_launch_args(args, provider, dry_run=args.dry_run)
    if launch_args.dry_run:
        spec = LaunchSpec(
            plan_id=launch_args.plan_id,
            region=launch_args.region,
            os_id=launch_args.os_id,
            label=launch_args.instance_label,
            gpu_ram_gb=args.min_gpu_ram,
            vcpus=0,
            hourly_cost=0.0,
        )
    else:
        spec = to_launch_spec(provider, launch_args)
    volume_spec = VolumeSpec(
        size_gb=launch_args.volume_size,
        label=f"{launch_args.instance_label}-cache",
        mount_path=DEFAULT_VOLUME_MOUNT,
        filesystem="ext4",
        extra={"region": launch_args.region},
    )
    bootstrap_cfg = build_bootstrap_config(launch_args)
    cloud_init = build_bootstrap_script(bootstrap_cfg)
    user_data = base64.b64encode(cloud_init.encode("utf-8")).decode("ascii")

    if launch_args.dry_run:
        print("──── Vultr Launch Dry Run ────")
        print(f"Region:          {launch_args.region}")
        print(f"Plan ID:         {spec.plan_id}")
        print(f"Instance label:  {launch_args.instance_label}")
        print(f"Volume size:     {volume_spec.size_gb} GB")
        print(f"Bootstrap stage: {launch_args.stage}")
        print()
        print("Bootstrap (cloud-init):")
        print(cloud_init)
        return

    print("Creating block volume…")
    volume_id = provider.provision_volume(volume_spec)
    provider.wait_for_volume(volume_id)
    print(f"✓ Volume {volume_id} active.")

    launch_spec = LaunchSpec(
        plan_id=spec.plan_id,
        region=launch_args.region,
        os_id=launch_args.os_id,
        label=launch_args.instance_label,
        gpu_ram_gb=spec.gpu_ram_gb,
        vcpus=spec.vcpus,
        hourly_cost=spec.hourly_cost,
    )
    print("Creating instance…")
    instance_id = provider.create_instance(launch_spec, user_data=user_data)
    print(f"✓ Instance ID: {instance_id}")
    print("Waiting for instance to boot…")
    provider.wait_for_instance(instance_id)

    print("Attaching block volume…")
    provider.attach_volume(instance_id, volume_id)
    print("✓ Attach request submitted.")

    print("Launch complete.")
    print(f"Instance ID: {instance_id}")
    print(f"Volume ID:   {volume_id}")
    print("Use `python scripts/vultr_launch.py status --instance <id>` to monitor.")


def cmd_status(args: argparse.Namespace) -> None:
    api = VultrProvider(resolve_key())
    instance = api.get_instance(args.instance)
    if not instance:
        print(f"No instance with id {args.instance!r} found.")
        return
    print(json.dumps(instance, indent=2))


def cmd_logs(args: argparse.Namespace) -> None:
    api = VultrProvider(resolve_key())
    instance = api.get_instance(args.instance)
    ip = instance.get("main_ip")
    if ip:
        print(f"Instance IP: {ip}")
        print()
        print("SSH command (adjust username if necessary):")
        print(f"  ssh root@{ip} 'tail -n 200 /var/log/cloud-init-output.log'")
    else:
        print("Instance IP unavailable. Ensure the instance exists and is running.")


def cmd_teardown(args: argparse.Namespace) -> None:
    api = VultrProvider(resolve_key())
    if args.instance:
        print(f"Destroying instance {args.instance}…")
        try:
            api.destroy_instance(args.instance)
        except CloudLaunchError as exc:
            print(f"⚠️  Failed to delete instance: {exc}")
        else:
            print("✓ Instance deleted.")
    if args.volume:
        if args.keep_volume:
            print(f"Skipping deletion of volume {args.volume} (--keep-volume set).")
        else:
            if args.instance:
                print(f"Detaching volume {args.volume} from instance {args.instance}…")
                try:
                    api.detach_volume(args.instance, args.volume)
                except CloudLaunchError as exc:
                    print(f"⚠️  Failed to detach volume: {exc}")
            print(f"Deleting volume {args.volume}…")
            try:
                api.delete_volume(args.volume)
            except CloudLaunchError as exc:
                print(f"⚠️  Failed to delete volume: {exc}")
            else:
                print("✓ Volume deleted.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vultr GPU launcher for Universal Simulator runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-env
    p_setup = sub.add_parser("setup-env", help="Validate local environment for Vultr access")
    p_setup.set_defaults(func=cmd_setup_env)

    # plans
    p_plans = sub.add_parser("plans", help="List available GPU plans")
    p_plans.add_argument("--region", help="Filter plans by region code (e.g., sjc1)")
    p_plans.set_defaults(func=cmd_plans)

    # launch
    p_launch = sub.add_parser("launch", help="Provision a Vultr instance and kick off training")
    p_launch.add_argument("--plan-id", help="Explicit plan identifier (e.g., vcg-3-32-1c-a100-80gb)")
    p_launch.add_argument("--region", help="Region code (sjc1, sea1, lax1, etc.)")
    p_launch.add_argument("--min-gpu-ram", type=int, default=40, help="Minimum GPU RAM (GB) required")
    p_launch.add_argument("--volume-size", type=int, default=1024, help="Block volume size in GB")
    p_launch.add_argument("--label", help="Instance label")
    p_launch.add_argument("--config", required=True, help="Training config path")
    p_launch.add_argument(
        "--os-id",
        type=int,
        default=DEFAULT_OS_ID,
        help="Operating system ID to install on the instance (default: Ubuntu 22.04 LTS x64)",
    )
    p_launch.add_argument("--stage", default="all", choices=["all", "operator", "diffusion", "distill"])
    p_launch.add_argument("--repo-url", help="Git remote URL (default: current origin)")
    p_launch.add_argument("--branch", help="Git branch (default: current branch)")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote work directory")
    p_launch.add_argument("--run-arg", action="append", default=[], help="Extra args for run_fast_to_sota.py")
    p_launch.add_argument("--no-precompute", action="store_true", help="Skip latent cache precompute")
    p_launch.add_argument("--keep-volume", action="store_true", help="Keep block volume on teardown")
    p_launch.add_argument("--dry-run", action="store_true", help="Print planned actions without contacting Vultr")
    p_launch.set_defaults(func=cmd_launch)

    # status
    p_status = sub.add_parser("status", help="Show instance metadata")
    p_status.add_argument("--instance", required=True, help="Instance identifier")
    p_status.set_defaults(func=cmd_status)

    # logs
    p_logs = sub.add_parser("logs", help="Print instructions to view boot logs via SSH")
    p_logs.add_argument("--instance", required=True, help="Instance identifier")
    p_logs.set_defaults(func=cmd_logs)

    # teardown
    p_teardown = sub.add_parser("teardown", help="Destroy instance and (optionally) volume")
    p_teardown.add_argument("--instance", help="Instance identifier")
    p_teardown.add_argument("--volume", help="Volume identifier")
    p_teardown.add_argument("--keep-volume", action="store_true", help="Skip volume deletion")
    p_teardown.set_defaults(func=cmd_teardown)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except CloudLaunchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
