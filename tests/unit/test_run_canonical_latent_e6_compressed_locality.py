from pathlib import Path

import pytest
import torch

from scripts.run_canonical_latent_e2_benchmark import (
    CanonicalCodec,
    Representation,
    _representations,
    representation_points,
)
from scripts.run_canonical_latent_e4_capacity_ladder import _state_dict_sha256
from scripts.run_canonical_latent_e6_compressed_locality import (
    CompressedLocalityConfig,
    RegionalLocalCodec,
    materialize_covering_radius,
    run_compressed_locality,
)


def test_e6_arms_share_exact_e3_encoder_and_respect_decoder_budget() -> None:
    cfg = CompressedLocalityConfig()
    torch.manual_seed(cfg.seed)
    control = CanonicalCodec(cfg.benchmark_config())
    torch.manual_seed(cfg.seed)
    challenger = RegionalLocalCodec(cfg)

    assert _state_dict_sha256(control) == (
        "1656fe58c8d7e826b69ac07f9a17451d8943f0b3732fcdd10895a70c54cc47c8"
    )
    assert _state_dict_sha256(control.encoder) == _state_dict_sha256(challenger.encoder)
    assert sum(p.numel() for p in challenger.decoder.parameters()) <= sum(
        p.numel() for p in control.decoder.parameters()
    )


def test_frozen_radius_covers_all_materialized_representations() -> None:
    cfg = CompressedLocalityConfig()
    representations = _representations(cfg.benchmark_config())
    query, _ = representation_points(
        Representation("canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0),
        batch=1,
    )
    report = materialize_covering_radius(RegionalLocalCodec(cfg).encoder, representations, query)

    assert report["maximum"] == pytest.approx(0.4444444477558136)
    assert cfg.local_support_radius >= cfg.covering_radius_margin * report["maximum"]


def test_tiny_e6_run_materializes_no_bypass_decision(tmp_path: Path) -> None:
    cfg = CompressedLocalityConfig(
        train_states=4,
        validation_states=4,
        epochs=1,
        batch_size=2,
        train_low_resolution=4,
        train_high_resolution=5,
        validation_resolution=6,
        canonical_query_resolution=6,
    )

    result = run_compressed_locality(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e6_compressed_locality"
    assert set(result["arms"]) == {"global_control", "local_integral"}
    assert result["causal_decision"]["classification"] in {
        "compressed_spatial_latent_qualified",
        "compressed_locality_helpful_but_insufficient",
        "compressed_locality_not_qualified",
    }
    assert result["arms"]["local_integral"]["architecture"]["latent_tokens"] == 8
    assert result["arms"]["local_integral"]["architecture"]["source_bypass"] is False
    for arm in result["arms"].values():
        for family in ("grid", "mesh"):
            assert arm["families"][family]["training"]["optimizer_updates"] == 2
            assert arm["families"][family]["training"]["scheduled_source_examples"] == 8
    assert result["boundary"] == {
        "operator_instantiated": False,
        "heldout_reads": 0,
        "representation_label_model_inputs": False,
        "task_label_model_inputs": False,
        "provider_calls": 0,
        "routing_paths": 0,
        "original_source_features_available_to_decoder": False,
    }
    assert Path(result["result_path"]).is_file()
