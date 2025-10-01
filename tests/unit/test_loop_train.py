import sys

import torch
from torch.utils.data import DataLoader

from ups.core.blocks_pdet import PDETransformerConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.training.loop_train import CurriculumConfig, LatentTrainer


def make_dummy_batch(batch=2, tokens=8, dim=16):
    encoded = torch.randn(batch, tokens, dim)
    batch_dict = {
        "encoded": encoded,
        "reconstructed": encoded + 0.01 * torch.randn_like(encoded),
        "decoded_pred": torch.randn(batch, tokens, 1, requires_grad=True),
        "decoded_target": torch.randn(batch, tokens, 1),
        "pred_next": torch.randn(batch, tokens, dim, requires_grad=True),
        "target_next": torch.randn(batch, tokens, dim),
        "pred_rollout": torch.randn(batch, 2, tokens, dim, requires_grad=True),
        "target_rollout": torch.randn(batch, 2, tokens, dim),
    }
    return batch_dict


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.batch = make_dummy_batch()

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.batch


def test_latent_trainer_runs_one_step():
    latent_dim = 16
    operator_cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        pdet=PDETransformerConfig(input_dim=latent_dim, hidden_dim=32, depths=(1, 1, 1), group_size=8, num_heads=2),
    )
    operator = LatentOperator(operator_cfg)
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
    dataloader = DataLoader(DummyDataset(), batch_size=None)
    curriculum = CurriculumConfig(stages=[{}], rollout_lengths=[1], max_steps=2, grad_clip=0.5, ema_decay=0.99)

    trainer = LatentTrainer(operator, optimizer, dataloader, curriculum)
    trainer.train()

