from pathlib import Path

from scripts.run_canonical_latent_e10_geometry_universal_projection import (
    frozen_e10_config,
    run_e10,
)


def test_e10_config_freezes_fresh_states_geometries_and_e7_conditioning() -> None:
    cfg = frozen_e10_config()

    assert cfg.seed == 23
    assert cfg.seed + 10_000 == 10_023
    assert cfg.geometry_seed_start == 30_000
    assert cfg.geometry_seed_start + cfg.geometry_realizations - 1 == 30_007
    assert cfg.max_weighted_design_condition_number == 10.0
    assert cfg.max_condition_number == 100.0


def test_tiny_e10_run_uses_every_realization_pair_and_binds_provenance(
    tmp_path: Path,
) -> None:
    cfg = frozen_e10_config(
        validation_states=4,
        calibration_resolution=32,
        geometry_realizations=2,
    )

    result = run_e10(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e10_geometry_universal_projection"
    assert result["state_split"]["validation_seed"] == 10_023
    assert result["state_split"]["training_states_read"] == 0
    assert result["state_split"]["heldout_states_read"] == 0
    assert result["causal_decision"]["gates"]["design"] is True
    assert result["causal_decision"]["gates"]["provenance"] is True
    realization_counts = {
        family: len(budgets["high"]) for family, budgets in result["geometry_families"].items()
    }
    assert all(
        pair["realization_pairs_evaluated"]
        == realization_counts[name.split("__vs__")[0]] * realization_counts[name.split("__vs__")[1]]
        for name, pair in result["evaluation"]["exact_semantics"].items()
    )
    assert result["provenance"]["all_source_hashes_present"] is True
    assert result["provenance"]["git_head_present"] is True
    assert Path(result["result_path"]).is_file()
