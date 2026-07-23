import json
from dataclasses import replace
from pathlib import Path

import pytest

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


def test_e10_run_uses_every_realization_pair_and_binds_provenance(
    tmp_path: Path,
) -> None:
    cfg = frozen_e10_config()

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
    exact_families = result["evaluation"]["paths"]["exact_gram_projection"]["families"]
    assert all(
        len(realization["paths"]["exact_gram_projection"]["states"]) == 24
        and all(
            {
                "state_index",
                "coefficient_nrmse_to_canonical",
                "canonical_query_nrmse",
                "high_frequency_spectral",
                "design_rank",
                "source_order_coefficient_max_abs_error",
                "source_order_decoded_max_abs_error",
            }
            <= state.keys()
            for state in realization["paths"]["exact_gram_projection"]["states"]
        )
        for budgets in exact_families.values()
        for budget in budgets.values()
        for realization in budget["realizations"]
    )
    assert result["provenance"]["source_files_match_git_head"] is True
    assert result["provenance"]["worktree_clean"] is True
    assert result["provenance"]["git_head_present"] is True
    assert Path(result["result_path"]).is_file()
    assert "result_path" not in json.loads(Path(result["result_path"]).read_text())


def test_e10_rejects_every_off_contract_configuration_before_state_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exact frozen configuration"):
        run_e10(
            replace(frozen_e10_config(), validation_states=4),
            run_dir=tmp_path,
        )
