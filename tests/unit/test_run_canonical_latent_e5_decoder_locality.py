from pathlib import Path

import pytest
import torch

from scripts.run_canonical_latent_e4_capacity_ladder import (
    DirectPointCodec,
    _state_dict_sha256,
)
from scripts.run_canonical_latent_e5_decoder_locality import (
    DecoderLocalityConfig,
    LocalIntegralDecoder,
    LocalIntegralDirectPointCodec,
    run_decoder_locality,
)


def _sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    source_coords = torch.rand(2, 24, 2, generator=generator)
    query = source_coords[:, :7] + 0.005
    source_tokens = torch.rand(2, 24, 32, generator=generator)
    measure = torch.rand(2, 24, 1, generator=generator) + 0.1
    return source_tokens, source_coords, measure, query


def test_local_decoder_is_quadrature_sensitive_and_source_order_invariant() -> None:
    torch.manual_seed(17)
    decoder = LocalIntegralDecoder(DecoderLocalityConfig()).eval()
    source_tokens, source_coords, measure, query = _sample()
    weighted = measure.clone()
    weighted[:, :12] *= 5.0

    baseline = decoder(source_tokens, source_coords, measure, query)
    reweighted = decoder(source_tokens, source_coords, weighted, query)
    permutation = torch.randperm(24, generator=torch.Generator().manual_seed(8))
    permuted = decoder(
        source_tokens[:, permutation],
        source_coords[:, permutation],
        measure[:, permutation],
        query,
    )

    assert not torch.allclose(baseline, reweighted)
    torch.testing.assert_close(baseline, permuted, rtol=0.0, atol=1e-6)


def test_local_decoder_uses_fixed_physical_support_and_fails_closed() -> None:
    decoder = LocalIntegralDecoder(DecoderLocalityConfig()).eval()
    source_tokens, source_coords, measure, _ = _sample()
    far_query = torch.full((2, 1, 2), 3.0)

    with pytest.raises(ValueError, match="inside local support"):
        decoder(source_tokens, source_coords, measure, far_query)

    report = decoder.neighbor_report(source_coords, source_coords[:, :4])
    assert report["minimum"] >= 1
    assert report["maximum"] <= 32
    assert report["furthest_retained_distance"] <= 0.20


def test_challenger_reuses_encoder_and_stays_below_decoder_parameter_budget() -> None:
    cfg = DecoderLocalityConfig()
    torch.manual_seed(cfg.seed)
    control = DirectPointCodec(cfg.capacity_config())
    torch.manual_seed(cfg.seed)
    challenger = LocalIntegralDirectPointCodec(cfg)

    assert _state_dict_sha256(control.encoder) == _state_dict_sha256(challenger.encoder)
    assert sum(p.numel() for p in challenger.decoder.parameters()) <= sum(
        p.numel() for p in control.decoder.parameters()
    )
    assert not any("attention" in name for name, _ in challenger.decoder.named_parameters())


def test_tiny_e5_run_materializes_decision_and_boundaries(tmp_path: Path) -> None:
    cfg = DecoderLocalityConfig(
        train_states=4,
        validation_states=4,
        epochs=1,
        batch_size=2,
        train_low_resolution=4,
        train_high_resolution=5,
        validation_resolution=6,
        canonical_query_resolution=6,
    )

    result = run_decoder_locality(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e5_decoder_locality"
    assert set(result["arms"]) == {"global_control", "local_integral"}
    assert result["causal_decision"]["classification"] in {
        "decoder_locality_causal",
        "decoder_locality_helpful_but_insufficient",
        "decoder_locality_not_causal",
    }
    assert (
        result["arms"]["local_integral"]["architecture"]["decoder_parameters"]
        <= result["arms"]["global_control"]["architecture"]["decoder_parameters"]
    )
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
    }
    assert Path(result["result_path"]).is_file()
