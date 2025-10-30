"""Integration tests for Muon optimizer in training loop."""
import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path


def test_muon_optimizer_creation_from_config():
    """Test that Muon optimizer can be created from config."""
    from scripts.train import _create_optimizer

    # Simple model
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.LayerNorm(32),
        nn.Linear(32, 8),
    )

    # Config with muon_hybrid optimizer
    cfg = {
        "stages": {
            "operator": {
                "optimizer": {
                    "name": "muon_hybrid",
                    "lr": 1e-3,
                    "weight_decay": 0.03,
                    "muon_momentum": 0.95,
                    "muon_ns_steps": 5,
                }
            }
        }
    }

    try:
        optimizer = _create_optimizer(cfg, model, "operator")
        assert optimizer is not None

    except RuntimeError as e:
        if "No Muon optimizer implementation found" in str(e):
            pytest.skip("No Muon implementation available")
        raise


def test_muon_training_loop():
    """Test a minimal training loop with Muon."""
    from ups.training.param_groups import build_param_groups
    from ups.training.hybrid_optimizer import HybridOptimizer
    from ups.training.muon_factory import create_muon_optimizer

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    try:
        muon_params, adamw_params = build_param_groups(model)
        muon_opt, _ = create_muon_optimizer(muon_params, lr=1e-2, weight_decay=1e-3)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=1e-2, weight_decay=1e-3)
        optimizer = HybridOptimizer([muon_opt, adamw_opt])

        # Train for 10 steps
        initial_loss = None
        final_loss = None

        for step in range(10):
            x = torch.randn(2, 4)
            y = torch.randn(2, 4)

            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)

            if step == 0:
                initial_loss = loss.item()
            if step == 9:
                final_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should decrease (model is learning)
        assert final_loss < initial_loss, f"Loss should decrease, got {initial_loss:.4f} -> {final_loss:.4f}"

    except RuntimeError as e:
        if "No Muon optimizer implementation found" in str(e):
            pytest.skip("No Muon implementation available")
        raise


def test_muon_config_validation():
    """Test that Muon config validates correctly."""
    config_path = Path("configs/train_burgers_muon.yaml")

    if not config_path.exists():
        pytest.skip("Muon config not yet created")

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Check Muon parameters are present
    opt_cfg = cfg["stages"]["operator"]["optimizer"]
    assert opt_cfg["name"] == "muon_hybrid"
    assert "muon_momentum" in opt_cfg
    assert "muon_ns_steps" in opt_cfg
    assert "lr" in opt_cfg
    assert "weight_decay" in opt_cfg

    # Check dimensions match
    assert cfg["latent"]["dim"] == cfg["operator"]["pdet"]["input_dim"]
    assert cfg["latent"]["dim"] == cfg["diffusion"]["latent_dim"]
