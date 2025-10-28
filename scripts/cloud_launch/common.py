"""Shared abstractions for cloud training launchers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable


class CloudLaunchError(RuntimeError):
    """Raised when a cloud provisioning step fails."""


@dataclass(slots=True)
class LaunchSpec:
    """Describe the resources required for provisioning a training instance."""

    plan_id: str
    region: str
    os_id: int
    label: str
    gpu_ram_gb: int
    vcpus: int
    hourly_cost: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VolumeSpec:
    """Describe a block volume to attach to the training instance."""

    size_gb: int
    label: str
    mount_path: Path
    filesystem: str = "ext4"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BootstrapConfig:
    """Parameters required to build and run an on-start bootstrap script."""

    repo_url: str
    branch: str
    workdir: Path
    config_path: Path
    stage: str
    run_args: list[str] = field(default_factory=list)
    precompute_latent_cache: bool = True
    wandb_tags: list[str] = field(default_factory=list)
    environment_tag: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)
    env_exports: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class CloudProvider(Protocol):
    """Minimal interface implemented by cloud-specific launchers."""

    name: str

    def list_offers(self, *, region: str | None = None) -> Iterable[LaunchSpec]:
        """Return available launch specs, optionally filtered by region."""

    def provision_volume(self, spec: VolumeSpec) -> str:
        """Create a block volume and return its provider-specific identifier."""

    def wait_for_volume(self, volume_id: str, *, timeout: float = 600.0) -> None:
        """Block until the volume is ready for attachment."""

    def create_instance(self, spec: LaunchSpec, *, user_data: str | None = None) -> str:
        """Provision a compute instance and return its provider-specific identifier."""

    def wait_for_instance(self, instance_id: str, *, timeout: float = 900.0) -> None:
        """Block until the instance is powered on and reachable."""

    def attach_volume(self, instance_id: str, volume_id: str) -> None:
        """Attach an existing volume to an instance."""

    def run_bootstrap(self, instance_id: str, script: str) -> None:
        """Trigger execution of the bootstrap script on the remote instance."""

    def destroy_instance(self, instance_id: str) -> None:
        """Terminate the instance."""

    def detach_volume(self, instance_id: str, volume_id: str) -> None:
        """Detach a volume without deleting it."""

    def delete_volume(self, volume_id: str) -> None:
        """Delete a previously provisioned volume."""
