# P2 Model-Side Transport Head Real-Shard Validation Plan

Date: 2026-06-24

Status: no-provider plan prepared; real-shard run blocked in the current
checkout because required local beta-provenance data and checkpoint source are
absent. No GPU/provider work ran, no held-out test was used, no claim evidence
changed, and no public language changed.

## Purpose

The synthetic smoke proved only schema mechanics for the model-side beta
transport head. The next meaningful step is a validation-only real-shard run
that checks whether the model-side head can reproduce the beta-conditioned
transport signal on the same bounded `light-v1`-style surfaces without enabling
the older evaluator sidecar.

This plan records the exact preflight, command path, artifact schema, gates,
and stop conditions before any provider/GPU or held-out work.

## Live Preflight

Commands run:

```bash
git status --short --branch
find data -maxdepth 3 -type f \( -name '*_train.h5' -o -name '*_val.h5' -o -name '*_test.h5' \) 2>/dev/null | sort
python scripts/run_light_experiment.py --help
python - <<'PY'
import h5py
from pathlib import Path
for path in sorted(Path("data/pdebench").glob("*_val.h5")):
    with h5py.File(path, "r") as handle:
        print(path, {key: list(value.shape) for key, value in handle.items()}, dict(handle.attrs))
PY
```

Findings:

- Git was clean on `codex/poseidon-channel-lift-vast`.
- Local standard train/val shards exist under `data/pdebench`:
  - `advection1d_train.h5`: `[128, 201, 1024, 1]`;
  - `advection1d_val.h5`: `[32, 201, 1024, 1]`;
  - `burgers1d_train.h5`: `[128, 201, 1024, 1]`;
  - `burgers1d_val.h5`: `[32, 201, 1024, 1]`;
  - `darcy2d_train.h5`: `[128, 128, 128, 1]`;
  - `darcy2d_val.h5`: `[32, 128, 128, 1]`.
- The standard `data/pdebench/advection1d_*.h5` shards do not contain
  `source_file_index` or `source_paths`, so `params.beta` cannot be derived.
  Running the model-side beta head on this root would skip every advection
  sample and is not a valid real-shard candidate.
- The prior canonical beta-provenance paths from existing evidence are not
  present locally:
  - `data/pdebench_official_advection_light`;
  - `reports/research/sota_loop/p2_parameter_canonical_root_sidecar/full_task_beta_val_root`.
- The prior checkpoint source used by the sidecar evidence is also absent
  locally:
  - `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`.

The real-shard run is therefore blocked on local artifact availability, not on
the model-side transport-head code.

## Required Inputs

The first real-shard validation candidate requires:

1. `data/pdebench` with Burgers/Darcy validation shards.
2. `data/pdebench_official_advection_light/advection1d_val.h5` with:
   - `data`;
   - `source_file_index`;
   - file attribute `source_paths` containing beta-coded official source paths.
3. A decoded UPS checkpoint source with encoder, decoder, and operator
   checkpoints equivalent to the prior sidecar run:
   `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`.

Do not use standard `data/pdebench/advection1d_val.h5` for this candidate
unless the purpose is an explicit negative preflight showing beta is absent.

## Root Build

Once `data/pdebench_official_advection_light` is present, build a single
validation-only full-task root:

```bash
python scripts/build_p2_parameter_full_task_root.py \
  --base-root data/pdebench \
  --advection-root data/pdebench_official_advection_light \
  --out-root reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root \
  --manifest-json reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root_manifest.json \
  --split val \
  --overwrite
```

This command refuses `split=test`, validates advection beta provenance, and
links/copies only validation shards.

## Validation Command

Run only after the root build and checkpoint preflight pass:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name ups_light_p2_model_side_beta_transport_head_val \
  --output-root reports/research/sota_loop/model_side_transport_head_real_shard \
  --stage operator_decoded \
  --skip-training \
  --checkpoint-source reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val \
  --decoded \
  --decoded-rollout-steps 16 \
  --device cpu \
  --override data.root=reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root \
  --override data.split=val \
  --override data.max_samples=32 \
  --override data.param_keys=[beta] \
  --override 'operator.conditioning.sources={task_id: 3, equation_signature: 15}' \
  --eval-override evaluation.skip_missing_tasks=false \
  --eval-override evaluation.decoded_persistence_residual_alpha=0.0 \
  --eval-override evaluation.report_all_horizon_metrics=true \
  --eval-override 'model_side_transport_head={enabled: true, tasks: [advection1d], required_params: [beta], features: ["param:beta", bias], init: {"param:beta": 10.236877359639507, bias: -0.08098891730605368}, mode: periodic_roll, apply_at: decoded_rollout, missing_param_policy: skip}'
```

The command intentionally does not set
`evaluation.decoded_data_conditioned_roll_shift_estimator`,
`evaluation.decoded_context_roll_shift_estimator`,
`evaluation.decoded_observed_roll_shift_estimator`, or
`evaluation.decoded_prediction_roll_shift_estimator`.

## Validator

Validate the summary immediately:

```bash
python scripts/validate_model_side_transport_head_summary.py \
  reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json
```

The validator must reject:

- held-out use or held-out data reads;
- `split=test` in command arguments;
- absent resolved `extra.model_side_transport_head`;
- task scope broader than `["advection1d"]`;
- missing `beta` in required params;
- nonzero beta-missing count;
- active context/data-conditioned/observed/prediction roll-shift estimators;
- failed aggregate, advection rollout, advection h16, Burgers, or Darcy gates.

## Gates

The validation candidate must clear all gates:

- aggregate decoded rollout NRMSE `< 0.35078329353213156`;
- advection decoded rollout NRMSE `< 0.4866576789288726`;
- advection h16 NRMSE `<= 0.44444171136384397`;
- Burgers decoded rollout NRMSE `<= 0.15674926288225416`;
- Darcy decoded rollout NRMSE `<= 0.2071060212271272`;
- `held_out_test_used = false`;
- `held_out_test_data_read = false`;
- `extra.model_side_transport_head_metrics.beta_missing_count = 0`;
- no incompatible roll-shift estimator config is active.

## Stop Conditions

Stop before running if any preflight fails:

- `data/pdebench_official_advection_light` is missing;
- the generated full-task root cannot prove advection beta provenance;
- the checkpoint source is missing encoder/decoder/operator checkpoints;
- the command would read `test`;
- the command would enable an evaluator roll-shift sidecar;
- the run would require GPU/provider spend, credentials, B2 upload, Vast top-up,
  or any external service mutation.

Stop after running if the summary validator fails. Record the validator errors
instead of rerunning with broadened scope.

## Current Blocker

In this checkout, the plan cannot run yet because the required official
advection beta root and the prior decoded checkpoint source are absent. The
existing hydration plan at
`reports/research/sota_loop/official_advection_hydration_plan.json` estimates
about `61.34 GiB` of official train-file downloads and explicitly says
downloads require approval for network and disk use. That remains outside this
tick.

## Recommendation

Next best path: restore or hydrate only the missing validation prerequisites
under a bounded no-held-out plan, then run the CPU validation command above. Do
not run provider/GPU work or held-out pretests until this real-shard validation
candidate produces a validator-clean summary.
