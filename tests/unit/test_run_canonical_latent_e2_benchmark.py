import json

import torch

from scripts.run_canonical_latent_e2_benchmark import (
    BenchmarkConfig,
    Representation,
    inverse_distance_interpolate,
    representation_points,
    run_benchmark,
)


def test_warped_representation_has_positive_normalized_measure() -> None:
    coords, measure = representation_points(Representation("mesh", 8, 0.24, -0.17), batch=3)

    assert coords.shape == (3, 64, 2)
    assert measure.shape == (3, 64, 1)
    assert torch.all(measure > 0)
    torch.testing.assert_close(measure.sum(dim=1), torch.ones(3, 1))


def test_inverse_distance_interpolation_preserves_sampled_values() -> None:
    coords, _ = representation_points(Representation("grid", 4, 0.0, 0.0), batch=1)
    values = coords[..., :1] + 2.0 * coords[..., 1:2]

    prediction = inverse_distance_interpolate(values, coords, coords)

    torch.testing.assert_close(prediction, values, rtol=1e-5, atol=1e-5)


def test_tiny_benchmark_is_source_bound_and_never_calls_operator(tmp_path) -> None:
    result = run_benchmark(
        BenchmarkConfig(
            train_states=4,
            validation_states=4,
            epochs=1,
            batch_size=4,
            latent_len=4,
            latent_dim=16,
            hidden_dim=16,
            supernodes=6,
            supernode_neighbors=4,
            train_low_resolution=4,
            train_high_resolution=5,
            validation_resolution=6,
            canonical_query_resolution=6,
            permutation_trials=99,
        ),
        run_dir=tmp_path,
    )

    assert result["evaluation"]["status"] in {"qualified", "not_qualified"}
    assert result["evaluation"]["boundary"] == {
        "operator_instantiated": False,
        "heldout_reads": 0,
        "representation_label_model_inputs": False,
        "task_label_model_inputs": False,
        "provider_calls": 0,
    }
    assert len(result["initial_checkpoint_sha256"]) == 64
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["config_sha256"] == result["config_sha256"]


def test_tiny_regional_challenger_reuses_frozen_gate_and_boundary(tmp_path) -> None:
    result = run_benchmark(
        BenchmarkConfig(
            encoder_kind="regional_interaction",
            train_states=4,
            validation_states=4,
            epochs=1,
            batch_size=4,
            latent_len=4,
            latent_dim=16,
            hidden_dim=16,
            supernodes=6,
            supernode_neighbors=4,
            train_low_resolution=4,
            train_high_resolution=5,
            validation_resolution=6,
            canonical_query_resolution=6,
            permutation_trials=99,
        ),
        run_dir=tmp_path,
    )

    assert result["experiment"] == "canonical_latent_e3_regional_interaction_analytic"
    assert result["architecture"]["encoder_kind"] == "regional_interaction"
    assert result["architecture"]["latent_sequence"] == "processed_regional_nodes"
    assert set(result["evaluation"]["gates"]) == {
        "identity",
        "within_codec",
        "absolute_reconstruction",
        "cross_codec",
        "canonical_queries",
        "paired_identity",
        "alignment_margin",
        "rank",
        "remeshing",
        "resolution_convergence",
        "boundary",
    }
    assert result["evaluation"]["boundary"]["operator_instantiated"] is False
