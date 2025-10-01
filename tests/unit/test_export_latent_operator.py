from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from export import export_latent_operator as export_mod


def test_export_torchscript(tmp_path):
    cfg = {
        "latent": {"dim": 8, "tokens": 4},
        "operator": {
            "pdet": {
                "input_dim": 8,
                "hidden_dim": 16,
                "depths": [1, 1, 1],
                "group_size": 4,
                "num_heads": 4,
            }
        },
        "training": {"dt": 0.1},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = export_mod.build_operator(cfg)
    ckpt = tmp_path / "operator.pt"
    torch.save(operator.state_dict(), ckpt)

    out_dir = tmp_path / "exported"
    diffs = export_mod.export(cfg, ckpt, out_dir, export_onnx=False)

    assert (out_dir / "operator.ts").exists()
    assert diffs["torchscript_max_error"] <= 1e-6
