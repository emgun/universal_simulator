import torch

from ups.training.lightning_modules import ConsistencyLightningModule, DiffusionLightningModule


def _dummy_cfg():
    return {
        "latent": {"dim": 8, "tokens": 4},
        "operator": {"architecture_type": "pdet_stack", "pdet": {"input_dim": 8, "hidden_dim": 16, "depth": 1, "num_heads": 2}},
        "diffusion": {"hidden_dim": 16},
        "training": {"dt": 0.1},
        "stages": {"operator": {"epochs": 1}, "diff_residual": {"epochs": 1}, "consistency_distill": {"epochs": 1}},
        "checkpoint": {"dir": "checkpoints"},
    }


def test_diffusion_lightning_step_runs():
    cfg = _dummy_cfg()
    module = DiffusionLightningModule(cfg)
    batch = {
        "z0": torch.randn(2, 3, 8),
        "z1": torch.randn(2, 3, 8),
        "cond": {},
    }
    loss = module.training_step(batch, 0)
    assert torch.is_tensor(loss)


def test_consistency_lightning_step_runs():
    cfg = _dummy_cfg()
    module = ConsistencyLightningModule(cfg)
    batch = {
        "z0": torch.randn(2, 3, 8),
        "cond": {},
    }
    loss = module.training_step(batch, 0)
    assert torch.is_tensor(loss)
