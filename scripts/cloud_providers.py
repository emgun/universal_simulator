"""Unified interface for GPU cloud providers (VastAI, Vultr)."""

from __future__ import annotations

import abc
import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class GPUInstance:
    """Unified GPU instance representation."""

    instance_id: str
    provider: str
    num_gpus: int
    gpu_model: str
    gpu_ram_gb: int
    cost_per_hour: float
    status: str
    ssh_host: str | None = None
    ssh_port: int | None = None


class CloudProvider(abc.ABC):
    """Abstract base class for GPU cloud providers."""

    @abc.abstractmethod
    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
    ) -> list[GPUInstance]:
        """Search for available GPU instances."""
        pass

    @abc.abstractmethod
    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
        startup_script: str | None = None,
    ) -> GPUInstance:
        """Create a new GPU instance."""
        pass

    @abc.abstractmethod
    def destroy_instance(self, instance_id: str) -> None:
        """Destroy a GPU instance."""
        pass

    @abc.abstractmethod
    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get instance status and details."""
        pass


class VastAIProvider(CloudProvider):
    """VastAI GPU cloud provider."""

    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
    ) -> list[GPUInstance]:
        """Search VastAI for available instances."""
        gpu_filter = (
            f"gpu_ram >= {min_gpu_ram} reliability > 0.95 num_gpus={num_gpus} disk_space >= 64"
        )
        if gpu_model:
            gpu_filter += f" gpu_name={gpu_model}"

        cmd = ["vastai", "search", "offers", gpu_filter, "--order", "dph_total", "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        instances = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            instances.append(
                GPUInstance(
                    instance_id=str(data["id"]),
                    provider="vastai",
                    num_gpus=data["num_gpus"],
                    gpu_model=data["gpu_name"],
                    gpu_ram_gb=data["gpu_ram"],
                    cost_per_hour=data["dph_total"],
                    status="available",
                )
            )

        return instances[:10]  # Top 10 by price

    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
        startup_script: str | None = None,
    ) -> GPUInstance:
        """Create VastAI instance."""
        # Search for best offer
        instances = self.search_instances(num_gpus, min_gpu_ram, gpu_model)
        if not instances:
            raise RuntimeError("No instances available")

        best_offer = instances[0]
        offer_id = best_offer.instance_id

        # Create instance with onstart script
        cmd = ["vastai", "create", "instance", offer_id]
        if startup_script:
            cmd.extend(["--onstart-cmd", startup_script])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        instance_id = result.stdout.strip()

        return GPUInstance(
            instance_id=instance_id,
            provider="vastai",
            num_gpus=best_offer.num_gpus,
            gpu_model=best_offer.gpu_model,
            gpu_ram_gb=best_offer.gpu_ram_gb,
            cost_per_hour=best_offer.cost_per_hour,
            status="creating",
        )

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy VastAI instance."""
        cmd = ["vastai", "destroy", "instance", instance_id]
        subprocess.run(cmd, check=True)

    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get VastAI instance status."""
        cmd = ["vastai", "show", "instances", "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in result.stdout.strip().split("\n"):
            data = json.loads(line)
            if str(data["id"]) == instance_id:
                return GPUInstance(
                    instance_id=instance_id,
                    provider="vastai",
                    num_gpus=data["num_gpus"],
                    gpu_model=data["gpu_name"],
                    gpu_ram_gb=data["gpu_ram"],
                    cost_per_hour=data["dph_total"],
                    status=data["actual_status"],
                    ssh_host=data.get("ssh_host"),
                    ssh_port=data.get("ssh_port"),
                )

        raise ValueError(f"Instance {instance_id} not found")


class VultrProvider(CloudProvider):
    """Vultr GPU cloud provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize Vultr provider with API key."""
        self.api_key = api_key or os.environ.get("VULTR_API_KEY")
        if not self.api_key:
            raise ValueError("VULTR_API_KEY environment variable not set")

    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
    ) -> list[GPUInstance]:
        """Search Vultr for available GPU plans."""
        # Use vultr-cli to list GPU plans
        cmd = ["vultr-cli", "plans", "list", "--type", "vhf", "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        plans = json.loads(result.stdout)

        instances = []
        for plan in plans:
            # Filter by GPU specs
            if "gpu" not in plan.get("id", "").lower():
                continue

            # Parse GPU count from plan name (e.g., "vhf-8c-32gb-a100-2gpu")
            plan_name = plan.get("id", "")
            if f"{num_gpus}gpu" not in plan_name.lower():
                continue

            # Parse GPU RAM from plan description
            # Vultr A100 plans have 80GB HBM2e per GPU
            gpu_ram = 80 if "a100" in plan_name.lower() else 48

            if gpu_ram < min_gpu_ram:
                continue

            if gpu_model and gpu_model.lower() not in plan_name.lower():
                continue

            instances.append(
                GPUInstance(
                    instance_id=plan["id"],
                    provider="vultr",
                    num_gpus=num_gpus,
                    gpu_model=gpu_model or "A100",
                    gpu_ram_gb=gpu_ram,
                    cost_per_hour=plan.get("monthly_cost", 0) / 730,  # Approximate hourly
                    status="available",
                )
            )

        return instances

    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: str | None = None,
        startup_script: str | None = None,
    ) -> GPUInstance:
        """Create Vultr GPU instance."""
        # Find appropriate plan
        plans = self.search_instances(num_gpus, min_gpu_ram, gpu_model)
        if not plans:
            raise RuntimeError("No Vultr plans available for specified GPU config")

        best_plan = plans[0]
        plan_id = best_plan.instance_id

        # Create instance
        cmd = [
            "vultr-cli",
            "instance",
            "create",
            "--region",
            "ewr",  # Newark (closest to most US users)
            "--plan",
            plan_id,
            "--os",
            "387",  # Ubuntu 22.04
            "--label",
            f"ups-ddp-{num_gpus}gpu",
            "--output",
            "json",
        ]

        if startup_script:
            # Write startup script to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(startup_script)
                script_path = f.name

            cmd.extend(["--script-id", script_path])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        instance_data = json.loads(result.stdout)

        return GPUInstance(
            instance_id=instance_data["id"],
            provider="vultr",
            num_gpus=best_plan.num_gpus,
            gpu_model=best_plan.gpu_model,
            gpu_ram_gb=best_plan.gpu_ram_gb,
            cost_per_hour=best_plan.cost_per_hour,
            status="creating",
        )

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy Vultr instance."""
        cmd = ["vultr-cli", "instance", "delete", instance_id]
        subprocess.run(cmd, check=True)

    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get Vultr instance status."""
        cmd = ["vultr-cli", "instance", "get", instance_id, "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        return GPUInstance(
            instance_id=instance_id,
            provider="vultr",
            num_gpus=int(data.get("gpu_count", 0)),
            gpu_model=data.get("gpu_type", "unknown"),
            gpu_ram_gb=80,  # Assume A100
            cost_per_hour=0,  # Not returned by get
            status=data["status"],
            ssh_host=data.get("main_ip"),
            ssh_port=22,
        )


def get_provider(provider_name: str) -> CloudProvider:
    """Factory function to get cloud provider instance."""
    if provider_name.lower() == "vastai":
        return VastAIProvider()
    elif provider_name.lower() == "vultr":
        return VultrProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Choose 'vastai' or 'vultr'.")
