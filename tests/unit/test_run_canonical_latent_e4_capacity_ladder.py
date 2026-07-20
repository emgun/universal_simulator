import json

import torch

from scripts.run_canonical_latent_e4_capacity_ladder import (
    CapacityLadderConfig,
    CapacityRung,
    DirectPointCodec,
    high_frequency_spectral_report,
    run_capacity_ladder,
)


def test_direct_point_codec_preserves_every_source_token_and_uses_measure() -> None:
    cfg = CapacityLadderConfig(
        train_states=4,
        validation_states=4,
        epochs=1,
        batch_size=4,
        latent_dim=16,
        hidden_dim=16,
        rungs=(CapacityRung("C4", 4, 6),),
    )
    torch.manual_seed(17)
    codec = DirectPointCodec(cfg)
    coords = torch.rand(2, 11, 2, generator=torch.Generator().manual_seed(5))
    values = coords[..., :1] - coords[..., 1:2]
    uniform = torch.ones(2, 11, 1)
    weighted = uniform.clone()
    weighted[:, :5] *= 4.0

    uniform_tokens = codec.encode(values, coords, uniform)
    weighted_tokens = codec.encode(values, coords, weighted)

    assert uniform_tokens.shape == (2, 11, 16)
    assert not torch.allclose(uniform_tokens, weighted_tokens)


def test_direct_point_decoding_is_invariant_to_source_storage_order() -> None:
    cfg = CapacityLadderConfig(
        train_states=4,
        validation_states=4,
        epochs=1,
        batch_size=4,
        latent_dim=16,
        hidden_dim=16,
        rungs=(CapacityRung("C4", 4, 6),),
    )
    torch.manual_seed(17)
    codec = DirectPointCodec(cfg).eval()
    coords = torch.rand(2, 11, 2, generator=torch.Generator().manual_seed(5))
    values = coords[..., :1] - coords[..., 1:2]
    measure = torch.ones(2, 11, 1)
    query = torch.rand(2, 7, 2, generator=torch.Generator().manual_seed(7))
    permutation = torch.randperm(11, generator=torch.Generator().manual_seed(9))

    original = codec.decode(codec.encode(values, coords, measure), query)
    permuted = codec.decode(
        codec.encode(values[:, permutation], coords[:, permutation], measure[:, permutation]),
        query,
    )

    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)


def test_high_frequency_spectral_report_is_exact_for_identity() -> None:
    prediction = torch.randn(3, 36, 1, generator=torch.Generator().manual_seed(8))

    report = high_frequency_spectral_report(
        prediction,
        prediction,
        resolution=6,
        minimum_radius=2.0,
    )

    assert report["nrmse"] == 0.0
    assert report["amplitude_ratio"] == 1.0


def test_tiny_capacity_ladder_is_source_bound_and_classifies(tmp_path) -> None:
    result = run_capacity_ladder(
        CapacityLadderConfig(
            train_states=4,
            validation_states=4,
            epochs=1,
            batch_size=4,
            latent_dim=16,
            hidden_dim=16,
            supernode_neighbors=4,
            train_low_resolution=4,
            train_high_resolution=5,
            validation_resolution=6,
            canonical_query_resolution=6,
            high_frequency_radius=2.0,
            rungs=(
                CapacityRung("C4", 4, 6),
                CapacityRung("C8", 8, 12),
            ),
        ),
        run_dir=tmp_path,
    )

    assert set(result["compressed_ladder"]) == {"C4", "C8"}
    assert set(result["direct_point_ceiling"]["families"]) == {"grid", "mesh"}
    assert result["causal_decision"]["classification"] in {
        "fixed_latent_capacity_causal",
        "compression_tokenization_causal",
        "decoder_objective_or_schedule_blocker",
    }
    assert result["boundary"] == {
        "operator_instantiated": False,
        "heldout_reads": 0,
        "representation_label_model_inputs": False,
        "task_label_model_inputs": False,
        "provider_calls": 0,
    }
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["config_sha256"] == result["config_sha256"]
