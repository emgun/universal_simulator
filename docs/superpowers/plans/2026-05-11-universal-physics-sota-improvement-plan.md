# Universal Physics Simulator SOTA Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `autoresearch` for measured experiment loops. Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` only when implementing a concrete task from this plan. Do not change datasets, splits, promotion rules, or held-out test selection during an experiment unless this file is explicitly updated first.

**Goal:** Lower held-out decoded rollout NRMSE and improve concrete physics-simulator evals while moving UPS toward a credible general/foundation physics simulator.

**Architecture:** Keep the current UPS latent-operator/encoder/decoder stack as the stable harness, then add targeted high-ROI improvements in layers: learned residual gating, transport-aware dynamics, rollout-stability/refinement, stronger cross-PDE conditioning, and finally foundation-scale pretraining. Each layer must win on validation before it gets one frozen held-out test.

**Tech Stack:** PyTorch, UPS `src/ups` modules, PDEBench HDF5 shards on B2, W&B benchmark-summary tracking, Vast GPU workers for training, local CPU eval-only probes, `scripts/run_light_experiment.py`, `scripts/plan_demo_experiments.py`, `scripts/calibrate_residual_gate.py`, `scripts/build_demo_report.py`.

---

## Execution Goal Hierarchy

The goal tool could not create a new active goal in this thread because the thread already has a completed goal record. Treat the following as the active repo-local goal contract.

### G0: Protect Benchmark Integrity

Maintain a clean, comparable benchmark story.

- Primary metric: held-out `decoded_rollout_nrmse`.
- Main baseline: `persistence_light_v1_test`, `decoded_rollout_nrmse = 0.5701633411507036`.
- Current clean general-model best: validation-calibrated transport residual alpha `0.20`, held-out `decoded_rollout_nrmse = 0.5283710326453532`.
- Current validation-selected demo best: advection roll shift `+40`, held-out `decoded_rollout_nrmse = 0.30780652221851373`.
- Current state-conditioned transport best: observed roll-shift estimator, held-out `decoded_rollout_nrmse = 0.20177292896682064`.
- Current state-conditioned improvement over persistence: `64.61138161573093%`.
- Demo gate: at least `20%` improvement over persistence unless the gate is explicitly revised with evidence.
- Never tune on `test`; use `val` for selection and run `test` once per selected candidate.
- Label test-swept or exploratory results as non-promoted evidence.

### G1: Beat The Current Clean Best

First milestone: produce a clean held-out candidate below `0.515` decoded rollout NRMSE.

This is a practical near-term target because the test-swept transport alpha `0.42` reached `0.5126627282110727` but is not benchmark-clean. A learned gate or transport head should recover or beat that score without test leakage.

### G2: Pass The 20% Persistence Gate

Second milestone: reach `decoded_rollout_nrmse <= 0.4561306729205629` on held-out `light-v1`.

This is the current threshold for a credible light-v1 demo claim:

```text
0.5701633411507036 * (1.0 - 0.20) = 0.4561306729205629
```

This gate is passed by the validation-selected roll-shift demo candidate and the observed state-conditioned estimator, but not yet by a fully causal learned transport mechanism.

### G2.5: Replace The Hard-Coded Transport Shift

North-star near-term target: preserve the roll-shift gain with a learned or parameter-conditioned transport mechanism.

Concrete target:

- Match or beat `ups_light_advection_roll_shift_40_test`, held-out `decoded_rollout_nrmse <= 0.30780652221851373`.
- New causal-head target: match or beat `ups_light_observed_shift_estimator_test`, held-out `decoded_rollout_nrmse <= 0.20177292896682064`, without using ground-truth future transitions.
- If the mechanism is clearly learned/parameter-conditioned and not hard-coded by task/split, accept a near-miss only if it still passes the persistence gate with margin: held-out `decoded_rollout_nrmse <= 0.330`.
- Improve held-out Advection to `task_advection1d_decoded_rollout_nrmse <= 0.4065598205949988`.
- Preserve Burgers and Darcy within `2%` of persistence/roll-shift values: `task_burgers1d_decoded_rollout_nrmse <= 0.17795817395661427`, `task_darcy2d_decoded_rollout_nrmse <= 0.21327743107503315`.
- Preserve spectral behavior at or below the roll-shift candidate: `decoded_rollout_spectral_energy_error <= 0.06721626079052936`.

Immediate executable path:

- Fit a transport-shift head on `advection1d_train.h5` only with `scripts/fit_transport_shift_head.py`.
- Validate the train-fitted shift on `advection1d_val.h5`; do not select a new shift from validation.
- Export the selected override only if train-fitted validation clears the guard.
- Run held-out `test` once through `scripts/run_light_experiment.py` only after the validation guard passes.
- Local same-split smoke runs must use `--allow-same-split-smoke` and are never benchmark evidence.
- If constant train-fitted shift fails validation, move directly to a per-sample/per-trajectory transport head; do not repeat constant-shift sweeps.
- Use `scripts/run_transport_shift_gate.py` as the authoritative train/val gate before any transport-shift held-out test run.

Stretch target:

- Held-out `decoded_rollout_nrmse <= 0.285`.
- Held-out `task_advection1d_decoded_rollout_nrmse <= 0.35`.
- Same or lower cost than the current light-v1 run class.

### G3: Improve Concrete Secondary Evals

Do not optimize NRMSE alone. A candidate is only worth scaling if it also improves or preserves:

- `task_advection1d_decoded_rollout_nrmse`
- `task_burgers1d_decoded_rollout_nrmse`
- `task_darcy2d_decoded_rollout_nrmse`
- `family_transport_decoded_rollout_nrmse`
- `decoded_h4_nrmse`, `decoded_h8_nrmse`, `decoded_h16_nrmse`
- `decoded_rollout_spectral_energy_error`
- decoded step-1 NRMSE
- conservation/BC residual metrics once wired into the scorecard
- throughput and GPU-hour cost once comparing larger candidates

### G4: Establish A Foundation-System Direction

Move from hand-tuned residual gates toward a general simulator:

- Cross-resolution and cross-task latent set interface.
- Equation/metadata-conditioned dynamics.
- Stable autoregressive rollout objective.
- Multiscale spectral/local dynamics.
- Pretraining that transfers across PDE families.
- Optional uncertainty/refinement stage for long-rollout reliability.

### G5: Scale Only After Light Evidence

Do not move to medium/full or expensive foundation pretraining until at least one light-v1 candidate either:

- passes the 20% persistence gate, or
- strongly improves the transport family and has a clear reason to scale, such as consistent validation wins across seeds and no Burgers/Darcy regression.

---

## Current State And Failure Pattern

The repo has a working light-v1 experiment loop with B2 data, W&B-capable remote execution, local scorecards, and validation-calibrated eval-only probes.

Current evidence:

- Original `ups_light_v1_task_signature_only`: `decoded_rollout_nrmse = 0.8881691012411048`.
- Persistence baseline: `0.5701633411507036`.
- Trained residual/stability candidate: `0.530536668470072`.
- Clean validation-calibrated transport residual gate alpha `0.20`: `0.5283710326453532`.
- Validation-selected advection roll shift `+40`: `0.30780652221851373`, passes the demo gate but remains a task-specific diagnostic until replaced by learned or parameter-conditioned transport logic.
- Observed state-conditioned roll-shift estimator: `0.20177292896682064`, passes roll-shift parity and shows transport state signal is recoverable, but still uses previous observed transitions and is not yet a fully causal learned rollout mechanism.
- Causal model-prediction shift estimator: validation `0.5584609221453186`, discarded before test because inferred shifts were unstable and Advection regressed to `0.8005553475932097`.
- Real `light-v1` train-fitted constant transport shift: train selected shift `0`, validation `0.5140249729156494`, discarded before test because validation failed the roll-shift parity guard; validation oracle remains shift `40`.
- Split-regime diagnostic: real `light-v1` Advection best shifts are train `0`, val `40`, test `72`, so the current split construction has incompatible constant-transport regimes.
- Transport-shift gate runner: train/val-only gate reports `test_eligible=false` on current `light-v1`; held-out test remains blocked.
- Train-derived compatible split proof: train/val best shifts both `0`, validation NRMSE `0.012206551618874073`, but this is not benchmark evidence because validation is derived from the train source split.
- Exploratory test-swept transport alpha `0.42`: `0.5126627282110727`, useful as an upper-bound clue but not a clean benchmark result.
- Horizon schedule overfit validation: validation `0.3562364331301045` vs constant validation `0.35679104424840724`, but exploratory test `0.5352231399077773`; keep as diagnostic, not promoted.

Interpretation:

- Manual scalar/horizon residual blending is near exhaustion.
- Fixed roll shifting shows transport phase error is the dominant short-term opportunity, but hard-coded task/split postprocessing is not the target system.
- Observed transition-conditioned shifting is now the local upper bound to distill into a causal transport head.
- The current trained-residual model's own next-step decoded prediction is not a sufficient causal phase estimator; a train/fitted transport head needs actual train/val trajectories.
- A constant train-fitted transport shift is exhausted: real `light-v1` train prefers shift `0` while validation prefers shift `40`, so the next mechanism must be per-sample/per-trajectory or the shard construction must be revisited.
- Before claiming a benchmark-clean transport-shift result, either rebuild the Advection split so train/val share the same transport-rate distribution or fit a per-trajectory head whose validation performance proves it learned the split variation from available trajectory features.
- The train-derived split proof shows the first option is viable mechanically; it still needs a true held-out successor test split before it can satisfy the benchmark-clean objective.
- Burgers and Darcy are less urgent than transport/advection.
- The high-value next step is a learned transport-aware residual/gating mechanism, not another hand-tuned alpha sweep.
- The foundation-model direction should be built behind this cheap evidence gate, not ahead of it.

---

## Research Synthesis

Use these research directions as a practical implementation menu, not as claims that the repo already matches them.

### Universal Physics Transformers

Universal Physics Transformers propose a general neural-operator framework for spatiotemporal problems using a unified tokenized representation rather than task-specific grids.

Relevance to UPS:

- UPS already has latent states, grid encoders, any-point decoders, and task conditioning.
- A UPT-like latent set with inducing tokens is a natural long-term backbone upgrade.
- This is the most aligned direction for a general/foundation simulator.

Source: [Universal Physics Transformers](https://arxiv.org/abs/2402.12365).

### Poseidon

Poseidon frames PDE foundation modeling around efficient PDE pretraining, multiscale operator design, time conditioning, and transfer to unseen PDEs.

Relevance to UPS:

- Supports the strategy of pretraining on a small diverse PDE set before scaling.
- Suggests semigroup/time-conditioning should be revisited after the current transport failure is fixed.
- Pushes toward validating transfer, not only same-family held-out NRMSE.

Source: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101).

### DPOT

DPOT uses autoregressive denoising operator-transformer pretraining and Fourier attention for large-scale PDE pretraining.

Relevance to UPS:

- The current UPS failures are rollout/stability failures, so denoising pretraining and corruption recovery are directly relevant.
- A DPOT-style objective can be implemented as a staged pretraining task before full foundation-scale runs.
- Fourier attention or spectral mixers are promising for lowering spectral energy error.

Source: [DPOT](https://arxiv.org/abs/2403.03542).

### PDE-Refiner

PDE-Refiner uses iterative refinement to improve long autoregressive rollouts and stability.

Relevance to UPS:

- Current residual gating is a crude one-shot correction. A learned refiner can replace hand-tuned alpha.
- A small decoded corrector can be tested cheaply on top of existing checkpoints.
- Refinement can target high-frequency/spectral error explicitly.

Source: [PDE-Refiner](https://arxiv.org/abs/2308.05732).

### AFNO / FourCastNet

FourCastNet demonstrates the effectiveness of adaptive Fourier neural operators for large-scale physical forecasting.

Relevance to UPS:

- Spectral energy error has been a recurring failure mode.
- AFNO/Fourier blocks are a plausible replacement or sidecar mixer for the latent operator.
- This is a medium-complexity experiment after learned gate/refiner tests.

Source: [FourCastNet](https://arxiv.org/abs/2202.11214).

### PDEBench, APEBench, LagrangeBench

Benchmarks should shape the eval surface:

- PDEBench gives PDE family breadth.
- APEBench emphasizes autoregressive emulator behavior and long rollout failure modes.
- LagrangeBench becomes relevant when particle/fluid generality is ready.

Sources: [PDEBench](https://arxiv.org/abs/2210.07182), [APEBench](https://arxiv.org/abs/2411.00180), [LagrangeBench](https://arxiv.org/abs/2309.16342).

---

## Target Metrics And Gates

### Promotion Metrics

Primary:

- `decoded_rollout_nrmse`

Required secondaries:

- `task_*_decoded_rollout_nrmse`
- `family_*_decoded_rollout_nrmse`
- `decoded_h1_nrmse`
- `decoded_h4_nrmse`
- `decoded_h8_nrmse`
- `decoded_h16_nrmse`
- `decoded_rollout_spectral_energy_error`

Add next:

- mass/conservation residual where applicable
- boundary-condition residual where applicable
- rollout stability slope, measured as `h16_nrmse / h1_nrmse`
- throughput samples/sec or tokens/sec
- GPU-hours and estimated dollars
- W&B run URL and B2 artifact URL

### Keep / Discard Rules

Keep a candidate if:

- validation primary metric improves by at least `1%` over the current best, and
- no task/family regresses by more than `3%` unless the primary gain exceeds `5%`, and
- spectral energy error does not regress by more than `10%`, and
- the implementation is simpler or justifiably more expressive than the previous best.

Discard a candidate if:

- it only wins on `test`,
- it wins by less than the complexity threshold,
- it improves Advection while catastrophically regressing Burgers or Darcy,
- it requires medium/full spend before winning on light-v1,
- it changes the eval harness instead of the model or training strategy.

Crash classification:

- no valid summary artifact,
- W&B requested but not recorded for paid remote,
- missing B2 shard,
- promotion metric absent,
- run exceeds cost/time budget.

---

## Experiment Loop

Every experiment follows the same loop:

1. Start from the current best commit on `codex/residual-light-candidate` or a fresh branch from it.
2. Write one clear hypothesis and name the target gate: clean-best, demo floor, roll-shift parity, or stretch.
3. Add or modify one bounded model/training feature.
4. Run unit tests for the touched surface.
5. Run `val` only for selection.
6. Record the required loop evals in a local experiment ledger.
7. Compare validation against the target gate and run exactly one `test` only if the validation guard passes.
8. Rebuild the scorecard.
9. Keep, discard, or crash.
10. Commit only kept harness improvements and measured winners.

Required loop eval contract:

- Selection split: `val`, not `test`, with the light-v1 evaluation shape unless this file is explicitly updated.
- Primary promotion metric: held-out `test` `decoded_rollout_nrmse`.
- Demo floor: held-out `decoded_rollout_nrmse <= 0.4561306729205629`.
- Roll-shift parity north star: held-out `decoded_rollout_nrmse <= 0.30780652221851373`.
- Learned near-miss allowance: held-out `decoded_rollout_nrmse <= 0.330` only when the mechanism is learned or parameter-conditioned and not hard-coded by task/split.
- Stretch north star: held-out `decoded_rollout_nrmse <= 0.285`.
- Required secondary evals: `task_advection1d_decoded_rollout_nrmse`, `task_burgers1d_decoded_rollout_nrmse`, `task_darcy2d_decoded_rollout_nrmse`, `decoded_h4_nrmse`, `decoded_h16_nrmse`, and `decoded_rollout_spectral_energy_error`.
- Safety bounds for roll-shift replacement: Advection `<= 0.4065598205949988`, Burgers `<= 0.17795817395661427`, Darcy `<= 0.21327743107503315`, spectral error `<= 0.06721626079052936`.
- Hard-coded or task-specific postprocesses can be logged as diagnostics or demos, but they do not satisfy the north-star generality goal until replaced by learned or parameter-conditioned logic.

Until the ledger schema is expanded, record `target_gate=...`, `selection_split=...`, `generality_level=...`, and `promotion_decision=...` in `description` and `reports/research/sota_loop/notes.md`.

Recommended ledger:

```text
reports/research/sota_loop/results.tsv
reports/research/sota_loop/run.log
reports/research/sota_loop/notes.md
```

Recommended columns:

```text
timestamp	branch	commit	run_name	summary_json	status	primary_metric	primary_metric_value	baseline_metric_value	baseline_ratio	baseline_improvement_fraction	advection_nrmse	burgers_nrmse	darcy_nrmse	transport_nrmse	conservation_nrmse	elliptic_nrmse	h1_nrmse	h4_nrmse	h8_nrmse	h16_nrmse	spectral_error	duration_sec	wandb_urls	description
```

---

## Phase 0: Harness And Logging

Objective: make the next iterations cheap, comparable, and hard to accidentally overfit.

Tasks:

- Create a research ledger under `reports/research/sota_loop/`.
- Add a compact metric extraction helper for summaries if one does not already exist.
- Ensure scorecard includes horizon metrics when present.
- Keep W&B `benchmark-summary` logging enabled for paid remote runs.
- Keep B2 dataset hydration read-only for experiments.
- Keep local runs eval-only unless explicitly using tiny synthetic/unit tests.

Validation:

```bash
pytest tests/unit/test_calibrate_residual_gate.py tests/unit/test_pdebench_runner_eval.py tests/unit/test_plan_demo_experiments.py tests/unit/test_demo_scorecard.py -q
python -m py_compile scripts/calibrate_residual_gate.py src/ups/eval/pdebench_runner.py scripts/plan_demo_experiments.py
git diff --check
```

Exit condition:

- A new experiment can be logged without changing the benchmark harness.

---

## Phase 1: Learned Residual Gate

Objective: replace hand-tuned transport alpha with a learned gate that selects how much UPS prediction to trust.

Hypothesis:

The current best hand-tuned alpha proves UPS contains useful residual signal for Advection, but fixed scalar gates underfit. A learned gate conditioned on task, family, horizon, latent state, and decoder residual should recover the exploratory `0.5126` score cleanly and potentially pass `0.515`.

Implementation options:

1. **Decoded sidecar gate**
   - Inputs: task/family embedding, horizon embedding, persistence field stats, decoded residual stats.
   - Output: scalar alpha per task/horizon/sample.
   - Lowest risk; can reuse current decoded eval path.

2. **Latent gate**
   - Inputs: latent tokens, task/family conditioning, horizon.
   - Output: token-wise or sample-wise blend between persistence-encoded and operator-predicted states.
   - Higher upside but more invasive.

3. **Family-specific gate table with learned regularization**
   - Inputs: family and horizon.
   - Output: learned alpha table with monotonic/smoothness regularization.
   - Useful bridge between scalar sweeps and full gating.

Recommended first experiment:

- decoded sidecar gate trained on `val`-like training split only, selected on `val`, tested once.

Success criteria:

- validation `decoded_rollout_nrmse` improves at least `1%` over `0.35679104424840724` on the comparable val setup,
- held-out test beats `0.5283710326453532`,
- target stretch: `<0.515`.

---

## Phase 2: Transport / Advection Dynamics Head

Objective: fix the known transport-family weakness directly.

Hypothesis:

Advection errors are partly phase/shift errors. A residual scalar gate cannot correct spatial displacement well. A learned warp or semi-Lagrangian head should improve transport rollouts without requiring a full backbone rewrite.

Candidate designs:

1. **Learned warp head**
   - Predict displacement field from latent/operator state.
   - Warp persistence field, then add residual correction.
   - Particularly appropriate for Advection.

2. **Velocity-conditioned transport head**
   - Infer or condition on advection velocity/coefficients if available.
   - Apply differentiable shift in physical space or latent coordinate space.
   - Use `scripts/fit_transport_shift_head.py` as the first constant-shift baseline before adding a learned per-sample head.

3. **Local stencil head**
   - Predict local finite-difference update/residual.
   - Bias toward stable local dynamics.

Success criteria:

- transport-family validation NRMSE improves by `5%`,
- overall validation NRMSE improves by `1%`,
- held-out test beats the clean alpha `0.20` result,
- Burgers/Darcy do not regress materially.

---

## Phase 3: Rollout Stability And Refinement

Objective: reduce long-horizon compounding error and spectral artifacts.

Hypothesis:

The current model needs a stability mechanism, not only a better one-step predictor. Iterative refinement and denoising objectives can directly attack long-rollout instability.

Candidate designs:

1. **PDE-Refiner-style decoded corrector**
   - Train a small corrector on decoded prediction, persistence field, residual, horizon, and task embedding.
   - Run 1-3 refinement steps during eval.

2. **DPOT-style denoising pretraining**
   - Corrupt trajectories or latent states.
   - Train model to recover clean next state autoregressively.

3. **Horizon-balanced rollout loss**
   - Upweight later horizons h8/h16.
   - Prevent h1-only improvement that worsens rollout.

4. **Spectral/gradient loss**
   - Penalize high-frequency energy mismatch and gradient mismatch.
   - Use cautiously because previous residual spectral loss helped but did not solve transport.

Success criteria:

- `decoded_h16_nrmse` improves more than h1 regresses,
- spectral energy error improves or stays stable,
- overall NRMSE improves cleanly on validation.

---

## Phase 4: Foundation Backbone Upgrade

Objective: evolve UPS from a demo model into a general physics foundation simulator.

Candidate designs:

1. **UPT-style latent set backbone**
   - Use learned inducing tokens as a simulation state.
   - Encode arbitrary grid/mesh/particle observations into latent set.
   - Decode any-point queries from latent state.

2. **Poseidon-style multiscale operator transformer**
   - Add time-conditioned normalization.
   - Add semigroup/multistep consistency after transport failures are controlled.
   - Validate transfer to unseen PDE families or resolutions.

3. **AFNO/Fourier latent mixer**
   - Replace or augment attention blocks with Fourier mixing.
   - Target spectral energy and rollout speed.

4. **Equation/text/metadata conditioning**
   - Encode PDE family, coefficients, boundary conditions, and equation signature.
   - Later: natural language PDE descriptions if the dataset supports it.

Success criteria:

- beats current best on light-v1,
- improves transfer split or unseen-task metrics,
- improves throughput or scaling behavior,
- does not require a bespoke head per PDE family.

---

## Phase 5: Scale-Up Protocol

Objective: scale only proven candidates.

Light tier:

- `train_max_samples=128`
- `eval_max_samples=32`
- `decoded_rollout_steps=16`
- B2 prefix: `light-v1`
- role: screen ideas cheaply

Medium tier:

- `train_max_samples=512`
- `eval_max_samples=128`
- `decoded_rollout_steps=32`
- B2 prefix: `medium-v1`
- role: validate that light wins scale

Full tier:

- use only after medium success
- include stronger baselines: FNO/AFNO/U-Net/Poseidon-like/DPOT-like if reproducible
- produce a formal benchmark report

Escalation rule:

- move light -> medium only after a clean light-v1 winner reaches either the 20% gate or a clear task-specific breakthrough.

---

## Concrete Experiment Queue

Run in this order unless a result changes the diagnosis.

| Priority | Experiment | Type | Expected Cost | Primary Target | Keep Threshold |
| --- | --- | --- | --- | --- | --- |
| 1 | Learned decoded residual gate | Train/eval | Low | overall + transport NRMSE | test `<0.52837`, stretch `<0.515` |
| 2 | Learned family/horizon gate with smoothness | Train/eval | Low | recover test-swept alpha without leakage | val +1%, no overfit |
| 3 | Advection warp head | Train/eval | Low-medium | transport NRMSE | transport +5% |
| 4 | Local stencil residual head | Train/eval | Low-medium | transport and h16 | h16 improves |
| 5 | PDE-Refiner sidecar | Train/eval | Medium | rollout/spectral stability | NRMSE + spectral win |
| 6 | DPOT-style denoising pretraining | Train/eval | Medium | rollout robustness | validation + test win |
| 7 | AFNO/Fourier mixer | Architecture | Medium | spectral + throughput | no task regression |
| 8 | UPT-style latent set prototype | Architecture | Medium-high | generality/transfer | transfer win |
| 9 | Poseidon-style multiscale/time-conditioned backbone | Architecture | High | foundation capability | medium-tier win |
| 10 | Full benchmark report | Eval/report | High | SOTA claim | reproducible comparisons |

---

## Immediate Sprint Plan

### Task 1: Initialize The Research Ledger

Create ignored or lightweight artifacts:

```bash
mkdir -p reports/research/sota_loop
cat > reports/research/sota_loop/results.tsv <<'EOF'
timestamp	branch	commit	run_name	summary_json	status	primary_metric	primary_metric_value	baseline_metric_value	baseline_ratio	baseline_improvement_fraction	advection_nrmse	burgers_nrmse	darcy_nrmse	transport_nrmse	conservation_nrmse	elliptic_nrmse	h1_nrmse	h4_nrmse	h8_nrmse	h16_nrmse	spectral_error	duration_sec	wandb_urls	description
EOF
cat > reports/research/sota_loop/notes.md <<'EOF'
# SOTA Loop Notes

Current clean best: validation-calibrated transport residual alpha 0.20.
Primary metric: held-out decoded_rollout_nrmse.
Benchmark baseline: persistence_light_v1_test.
EOF
```

### Task 2: Add A Metric Extraction Helper

Add a script that reads a `summary.json` and prints TSV-compatible metrics. This keeps every experiment comparable and reduces manual transcription mistakes.

Target path:

```text
scripts/extract_experiment_metrics.py
```

Required fields:

- run name
- primary metric
- baseline ratio if present
- task NRMSEs
- family NRMSEs
- spectral error
- W&B URLs if present

### Task 3: Add Planner Variants For Learned Gate Experiments

Extend `scripts/plan_demo_experiments.py` with placeholder-free variants only after the model/training code exists:

- `task_signature_learned_transport_gate`
- `task_signature_advection_warp_head`
- `task_signature_refiner_sidecar`

Do not add variants that cannot run.

### Task 4: Implement Learned Decoded Residual Gate

Minimal first implementation:

- Small MLP gate.
- Inputs: task/family one-hot, normalized horizon, persistence statistics, UPS residual statistics.
- Output: alpha in `[0, 1]`.
- Regularization: stay near persistence unless residual reduces validation loss.
- Eval path: report learned alpha mean/std by task/family/horizon.

Tests:

- alpha is bounded,
- alpha can differ by task/family,
- alpha can differ by horizon,
- decoded evaluation can consume learned gate outputs,
- constant alpha path remains unchanged.

### Task 5: Run Validation-Only Selection

Use the existing trained residual checkpoint as baseline input:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name ups_light_learned_transport_gate_val \
  --output-root reports/light_experiments_remote \
  --checkpoint-source reports/light_experiments_remote/ups_light_task_signature_trained_residual \
  --skip-training \
  --device cpu \
  --decoded \
  --override data.root=data/pdebench \
  --eval-override data.root=data/pdebench \
  --eval-override data.split=val \
  --eval-override data.max_samples=32 \
  --decoded-rollout-steps 16 \
  --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}' \
  --promotion-rule 'decoded_rollout_nrmse<=1.0'
```

Adapt this command when the learned-gate training stage exists.

### Task 6: Run One Frozen Test Only If Validation Wins

If validation passes the keep threshold, run:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name ups_light_learned_transport_gate_test \
  --output-root reports/light_experiments_remote \
  --checkpoint-source reports/light_experiments_remote/ups_light_task_signature_trained_residual \
  --skip-training \
  --device cpu \
  --decoded \
  --override data.root=data/pdebench \
  --eval-override data.root=data/pdebench \
  --eval-override data.split=test \
  --eval-override data.max_samples=32 \
  --decoded-rollout-steps 16 \
  --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}' \
  --promotion-rule 'decoded_rollout_nrmse<=1.0'
```

### Task 7: Rebuild Scorecard

```bash
python scripts/build_demo_report.py \
  --glob 'reports/light_experiments_remote/*/summary.json' \
  --output-dir reports/demo/light_latest \
  --title 'UPS Light-v1 Demo Scorecard' \
  --data-manifest docs/demo_data_manifest.yaml \
  --baseline-run persistence_light_v1_test \
  --baseline-metric decoded_rollout_nrmse \
  --baseline-min-improvement 0.2 \
  --promotion-rule 'decoded_rollout_nrmse<=1.0' \
  --copy-summaries
```

---

## Remote Execution Contract

Use remote training for any learned model. Local machine should remain eval-only except for unit/synthetic smoke tests.

Dry-run first:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
ALLOW_WANDB=1 \
WANDB_GROUP=light-v1-sota-loop \
WANDB_TAGS=light-v1,sota-loop,baseline-gated \
TASKS=burgers1d,advection1d,darcy2d \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=val \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
RUN_NAME=ups_light_learned_transport_gate \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16 --override operator.conditioning.sources={\"task_id\":3,\"equation_signature\":15}" \
bash scripts/run_remote_light_promotion.sh
```

Live run only after dry-run review:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=0 \
ALLOW_WANDB=1 \
WANDB_GROUP=light-v1-sota-loop \
WANDB_TAGS=light-v1,sota-loop,baseline-gated \
TASKS=burgers1d,advection1d,darcy2d \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=val \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
RUN_NAME=ups_light_learned_transport_gate \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16 --override operator.conditioning.sources={\"task_id\":3,\"equation_signature\":15}" \
bash scripts/run_remote_light_promotion.sh
```

Remote guardrails:

- W&B must be enabled for paid runs.
- B2 data must come from `light-v1` or explicitly documented successor prefixes.
- Use reviewed Vast offers; prefer low-cost GPUs until validation wins justify scale.
- Destroy instances after artifact publication.
- Copy summaries back into `reports/light_experiments_remote/`.

---

## Branch And Commit Policy

Branch naming:

- planning/docs: current branch is acceptable.
- implementation experiments: `codex/sota-learned-gate`, `codex/sota-transport-head`, or `autoresearch/YYYY-MM-DD-sota-loop`.

Commit rules:

- Commit harness/documentation improvements when tested.
- Commit model changes before remote runs if the remote worker needs the branch.
- Keep losing experimental commits only if they add reusable harness capability; otherwise revert or supersede.
- Never rewrite unrelated user work.

---

## Stop Conditions

Stop and reassess if:

- validation wins do not transfer to held-out test for two consecutive learned-gate/refiner attempts,
- transport improves but Burgers/Darcy regress enough to lose overall NRMSE,
- W&B/B2 tracking is absent for a paid run,
- local disk pressure threatens data/artifact integrity,
- medium/full scale is being considered without a clean light-v1 winner,
- benchmark comparison requires a new baseline implementation not currently in repo.

---

## Definition Of SOTA-Ready Demo

A SOTA-style claim is not allowed until all are true:

- held-out `light-v1` passes the 20% persistence improvement gate,
- medium or larger split confirms the same ordering,
- at least one strong neural baseline is reproduced or fairly compared,
- scorecard includes NRMSE, per-task metrics, spectral/stability metrics, and cost/throughput,
- W&B and B2 artifact handles exist for every claimed run,
- the plan documents exact splits, commands, commits, and checkpoint sources.

The fastest credible path is not to claim SOTA early. It is to produce a clean learned transport-gate/refiner win, scale only that, and then compare against stronger baselines once the UPS candidate is actually competitive.

---

## Execution Progress

### 2026-05-11: Learned Gate Hook

Status: Phase 0 complete; Phase 1 scaffolding started on `codex/sota-learned-gate`.

Implemented:

- `evaluation.decoded_persistence_residual_gate` in decoded evaluation.
- Bounded logistic gate parameters over the resolved residual alpha.
- Target-free gate features: horizon, normalized horizon, residual magnitude, persistence magnitude, prediction magnitude.
- Gate alpha metrics: global, per-task, per-family, and per-horizon mean/std.

Validation-only findings:

- `ups_light_gate_hook_constant_alpha0p2_val`: `decoded_rollout_nrmse = 0.36417941757537725`.
- `ups_light_gate_hook_transport_base_val`: `decoded_rollout_nrmse = 0.3567910081081011`.
- Learning: `base_alpha=0.2` applies to every family and regresses validation. Learned gates should usually omit `base_alpha` and operate around the already-resolved task/family alpha unless intentionally overriding all families.

Next step:

- Add a calibration/training path that learns gate parameters on train/validation data and exports a frozen `decoded_persistence_residual_gate` config. Do not run held-out `test` until validation beats the clean constant gate by the plan threshold.

### 2026-05-11: Learned Gate Calibration Export

Status: Phase 1 calibration/export path implemented; still validation-only.

Implemented:

- `scripts/calibrate_residual_gate.py --use-decoded-residual-gate`.
- Repeatable `--gate-config-candidate` JSON configs for decoded gate sweeps.
- `--gate-feature-weight name=value` convenience wiring into gate `feature_weights`.
- `--export-selected-gate-config` to write the selected validation gate as reusable overrides.

Smoke validation:

- Command used `--skip-test`, `--eval-max-samples 8`, and `--decoded-rollout-steps 4`.
- Exported selected gate: `reports/research/sota_loop/gate_calibrator_smoke/selected_gate.json`.
- Selected alpha `0.2` over `0.3`, with smoke `decoded_rollout_nrmse = 0.2900588529988161`.
- This is an integration smoke only, not benchmark-comparable.

Next step:

- Run a full comparable validation-only decoded gate config sweep over 32 samples and 16 rollout steps. Candidate weights should start with small residual/horizon terms around the clean transport alpha `0.2`. Only run held-out `test` if the validation result improves at least `1%` over `0.35679104424840724`.

### 2026-05-11: Learned Gate Feature Sweep

Status: validation-only feature sweep completed; no held-out test run.

Implemented:

- Calibrator held-out test guard: `--reference-metric-value` plus `--test-min-relative-improvement`.
- Selected-gate records now include `test_guard` and `test_skipped` when the guard blocks test.

Comparable validation results:

- Reference clean constant transport alpha: `0.35679104424840724`.
- Neutral decoded gate: `0.3567910081081011`.
- Best feature gate: alpha `0.2`, `feature_weights.horizon_norm=-0.5`.
- Best validation decoded rollout NRMSE: `0.35560983348888475`.
- Relative validation improvement: `0.0033106513702179665`.
- Guard threshold: `0.01`; held-out test skipped.

Learning:

- Decreasing residual trust over rollout horizon is directionally useful.
- The scalar target-free gate is too weak to justify test budget.
- Next implementation should target transport/advection dynamics directly or train a richer sidecar with per-sample supervision rather than continue manual scalar-feature sweeps.

### 2026-05-11: Advection Roll-Shift Transport Correction

Status: light-v1 held-out test win found; needs generalization into a learned/parameterized transport head.

Implemented:

- `evaluation.decoded_roll_shift_by_task`.
- `evaluation.decoded_roll_shift_by_family`.
- `evaluation.decoded_roll_shift_by_task_horizon`.
- `evaluation.decoded_roll_shift_by_family_horizon`.
- Synthetic unit coverage for exact one-cell periodic advection correction.

Validation selection:

- Setup: persistence residual alpha `0.0`, advection-only periodic roll shift, 32 validation samples, 16 rollout steps.
- Best selected shift: `+40`.
- Selected validation run: `ups_light_advection_roll_shift_40_val`.
- Selected validation decoded rollout NRMSE: `0.11155091371736849`.
- Local neighbors: `+38 -> 0.11443604286804047`, `+42 -> 0.11160543416536953`, `+44 -> 0.11459499323130662`.

Frozen held-out test:

- Run: `ups_light_advection_roll_shift_40_test`.
- Summary: `reports/light_experiments_remote/ups_light_advection_roll_shift_40_test/summary.json`.
- Decoded rollout NRMSE: `0.30780652221851373`.
- Persistence baseline: `0.5701633411507036`.
- Clean transport alpha gate: `0.5283710326453532`.
- Baseline ratio vs persistence: `0.5398567392938639`.
- Baseline improvement fraction: `0.46014326070613615`.
- Rebuilt scorecard: `reports/demo/light_latest/scorecard.json`.
- Scorecard gate: `baseline_improvement_passed=true`.

Interpretation:

- This is the first light-v1 held-out result in this loop that clears the 20% persistence improvement gate.
- The result shows the transport/advection weakness is largely phase/translation error.
- It should not yet be presented as SOTA or foundation-model evidence by itself because the shift is validation-selected and task-specific.

Next step:

- Convert the hand-selected shift into a learned or parameter-conditioned transport head, then rerun the same validation/test discipline. The target is to preserve the held-out gain while removing the hard-coded advection postprocess.

### 2026-05-11: Roll-Shift Calibration Harness

Status: transport-shift result is reproducible through a script; still not a learned general mechanism.

Implemented:

- `scripts/calibrate_roll_shift.py`.
- `task_signature_advection_roll_shift40` planner variant.
- Unit tests for roll-shift override serialization and horizon schedule selection.

Reconstructed calibration:

- Command used `--reuse-existing` over shifts `36`, `38`, `40`, `42`, and `44`.
- Selected validation shift: `+40`.
- Selected validation decoded rollout NRMSE: `0.11155091371736849`.
- Selected validation guard improvement vs clean transport gate: `0.6873494570124249`.
- Exported selected shift: `reports/research/sota_loop/transport_shift_sweep/selected_shift.json`.
- Reused frozen held-out test summary: `reports/light_experiments_remote/ups_light_advection_roll_shift_40_test/summary.json`.

Learning:

- The result is now reproducible without a manual shell loop.
- The validation/test gap is important: validation advection is nearly solved by `+40`, but held-out test advection remains `0.4065598205949988`.
- The fixed shift is therefore a useful diagnostic and demo candidate, not a universal transport law.
- Local hydration currently has `advection1d_val.h5` and `advection1d_test.h5`, but not `advection1d_train.h5`; learned transport-head work needs remote train data hydration or a dedicated small train split.

Next step:

- Build a learned or parameter-conditioned shift estimator for Advection. The safest design is to train/select on train-like data, validate on `val`, and run exactly one `test` only if it beats the fixed-shift candidate or preserves most of its gain while removing the hard-coded shift.

### 2026-05-15: Transport Shift Gate And Successor Split Result

Status: official current `light-v1` gate is blocked; corrected successor split gate passes with one held-out test.

Implemented:

- `scripts/fit_transport_shift_head.py` for train-only constant-shift fitting and validation measurement.
- `scripts/diagnose_transport_shift_splits.py` for split-regime diagnostics.
- `scripts/run_transport_shift_gate.py` for the benchmark discipline: fit on train, validate against the guard, and measure held-out test only when train/validation pass.
- Optional `--test-split` support in the gate runner so the one-test measurement is produced by the same guarded command.

Current `light-v1` evidence:

- Real hydrated train shard: `data/pdebench/advection1d_train.h5`.
- Best shifts from diagnostic: train `0`, val `40`, test `72`.
- Train-fitted constant shift: `0`.
- Validation NRMSE for train-fitted shift: `0.5140249729156494`.
- Fixed-shift held-out reference from prior selected `+40` run: `0.30780652221851373`.
- Validation guard result: failed, relative improvement `-0.6699612770087436`.
- Decision: no held-out test run on the official current `light-v1` path.

Successor split evidence:

- Built non-overlapping train/validation/test Advection shards from the real train source shard: train `64`, val `32`, test `32`.
- Successor train/val best shifts: train `0`, val `0`.
- Train-fitted shift: `0`.
- Successor validation NRMSE: `0.012206551618874073`.
- Validation relative improvement vs `0.30780652221851373`: `0.9603434276476813`.
- Held-out successor test NRMSE after gate pass: `0.015408719889819622`.
- Held-out successor test oracle shift: `0`.

Learning:

- The gate and data discipline are now implemented and verified end to end.
- The original current `light-v1` split is not a valid target for a constant global transport shift because split regimes differ.
- The successor result demonstrates the achievable path when train/validation/test are transport-compatible, but it is not official benchmark evidence because it derives all splits from the train source shard.

Next step:

- Choose between two paths before spending more experiment budget: rebuild a compatible benchmark shard from raw/source data with split-balanced transport regimes, or implement a per-trajectory transport head that can infer shift/rate from trajectory features and survive the current `light-v1` train/val/test mismatch.

### 2026-05-15: Official Light-v1 Observed-Transition Transport Gate

Status: official current `light-v1` state-conditioned transport gate passes validation and measured one held-out test.

Implemented:

- `scripts/run_observed_transport_shift_gate.py`.
- Unit tests for guarded test measurement and validation-guard blocking.
- A strict train/val/test gate for the lagged observed-transition transport estimator.

Result:

- Data root: `data/pdebench`.
- Task: `advection1d`.
- Train max samples: `128`.
- Validation/test max samples: `32`.
- Rollout steps: `16`.
- Candidate shifts: `-96` through `96` in steps of `8`.
- Reference metric: fixed validation-selected `+40` held-out NRMSE `0.30780652221851373`.
- Train NRMSE: `0.014504759572446346`; inferred shift mean/std `0.0` / `0.0`.
- Validation NRMSE: `0.012846261262893677`; inferred shift mean/std `40.0` / `0.0`.
- Validation relative improvement: `0.9582651427581705`.
- Held-out test NRMSE after gate pass: `0.004225204233080149`; inferred shift mean/std `72.0` / `0.0`.

Interpretation:

- This satisfies the official current `light-v1` train/val/test gate for a state-conditioned transport-shift estimator.
- It does not rescue the global constant train-fitted shift; that path remains invalid because train/val/test shifts are `0`/`40`/`72`.
- It is not a fully autonomous causal rollout head because it uses the previous observed transition at each step. Treat it as a clean upper-bound target and a concrete mechanism spec for the next learned head.

Next step:

- Train or fit a causal transport head to predict the same per-trajectory shift/rate from allowed model state or metadata, with `observed_transport_shift_gate` as the target upper bound and `0.004225204233080149` as the official light-v1 Advection transport-shift test target to preserve.

### 2026-05-15: Train-Window Scan For Constant Shift Recovery

Status: local current `light-v1` train shard cannot produce a constant train-fitted shift compatible with official validation.

Implemented:

- `scripts/scan_transport_train_windows.py`.
- Unit coverage for source-window shift histograms.

Local current `light-v1` scan:

- Source: `data/pdebench/advection1d_train.h5`.
- Source shape: `[128, 201, 1024, 1]`.
- Window size/stride: `32` / `32`.
- Windows scanned: `4`.
- Best-shift histogram: `{"0": 4}`.

Learning:

- The current official light train shard contains only shift-`0` windows.
- The official validation shard is shift `40` and official test is shift `72`, so the constant train-fitted shift objective cannot be achieved from the already-hydrated local train shard.
- This is a data-construction/source-coverage blocker, not an optimizer or candidate-grid blocker.

Next step:

- Run `scripts/scan_transport_train_windows.py` remotely against the full Advection train source on B2-backed storage, then build a corrected light shard from train-source windows with the desired transport-regime coverage before rerunning `scripts/run_transport_shift_gate.py`.

### 2026-05-15: Remote Constant-Shift Candidate Pipeline

Status: remote/full-source execution path is implemented; actual full-source run remains the next external step.

Implemented:

- `scripts/make_light_hdf5_shards.py --split-start-index SPLIT=N`.
- `scripts/run_remote_transport_shift_candidate.sh`.
- `scripts/launch_remote_transport_shift_candidate_vast.sh`.
- Dry-run/unit coverage for local-safe planning.

Pipeline:

- Hydrate full Advection train/val/test shards from B2 prefix `full`.
- Scan full train windows for `TARGET_SHIFT`, default `40`, without using validation/test selection.
- Build a small candidate shard with the selected train-source start and native val/test starts.
- Run `scripts/run_transport_shift_gate.py` with `--test-split test`, so held-out test is measured only if validation passes.

Execution:

```bash
DRY_RUN=1 ENV_FILE=/path/to/.env bash scripts/launch_remote_transport_shift_candidate_vast.sh
DRY_RUN=0 ENV_FILE=/path/to/.env bash scripts/launch_remote_transport_shift_candidate_vast.sh
```

Learning:

- This keeps the original constant train-fitted shift objective intact while moving the blocked local path to the only viable evidence source: full train coverage on remote storage.
- The result is not complete until the remote scan finds a compatible train window, the gate passes validation, and exactly one held-out test is measured and recorded.

### 2026-05-16: Full-Train Scan Result

Status: full-source scan falsified the constant train-fitted shift path for official current `light-v1` validation.

Remote evidence:

- Vast instance: `36855174`.
- Branch: `codex/sota-learned-gate`.
- B2 prefix: `full`.
- Full train source shape: `[60000, 201, 1024, 1]`.
- Window size/stride: `32` / `32`.
- Windows scanned: `1875`.
- Best-shift histogram: `{"0": 625, "8": 937, "16": 1, "24": 312}`.
- Target validation shift `40`: `0` matching train windows.
- Local evidence copy: `reports/research/sota_loop/remote_transport_shift_candidate/train_window_scan.json`.

Interpretation:

- The original train-only constant-shift objective cannot be completed against the current official validation regime from available full train windows.
- This is now a source-coverage and benchmark-construction blocker, not a local data, compute, or candidate-grid blocker.
- The pipeline correctly stopped before candidate build/test because a held-out test would not be benchmark-clean without a train-supported validation pass.

Next step:

- Choose an explicit benchmark-policy change: either rebuild the light benchmark so train/val/test share transport-rate support, or retire the constant-shift objective and pursue a learned/state-conditioned transport mechanism with a different gate.

### 2026-05-16: Full Train/Val/Test Compatibility Scan

Status: full-source compatibility scan proves no native train/val/test constant-shift candidate exists under the current split construction.

Implemented:

- `scripts/select_transport_compatible_windows.py`.
- `SCAN_ALL_SPLITS=1 REQUIRE_TEST_COMPATIBLE=1` mode in `scripts/run_remote_transport_shift_candidate.sh`.

Remote evidence:

- Vast instance: `36856643`.
- Branch: `codex/sota-learned-gate`.
- Local evidence directory: `reports/research/sota_loop/remote_transport_shift_candidate_all_splits/`.
- Train source shape: `[60000, 201, 1024, 1]`; windows scanned `1875`; histogram `{"0": 625, "8": 937, "16": 1, "24": 312}`.
- Validation source shape: `[10000, 201, 1024, 1]`; windows scanned `313`; histogram `{"40": 313}`.
- Test source shape: `[10000, 201, 1024, 1]`; windows scanned `313`; histogram `{"72": 313}`.
- Compatible train/val/test shifts: `[]`.

Interpretation:

- No native full-source train/val/test window triplet supports a constant train-fitted shift.
- The original constant-shift objective is incompatible with the current Advection split construction.
- No held-out test was run, because the validation-gate precondition cannot be satisfied benchmark-cleanly.

Decision needed:

- Rebuild a new benchmark split with shared transport-rate support if the goal remains constant-shift transport correction.
- Otherwise retire the constant-shift objective and use the observed-transition result as the upper bound for a learned/state-conditioned transport head.

### 2026-05-19: Transport-Shift Goal Audit

Status: added a machine-readable audit for the original benchmark-clean constant-shift objective.

Implemented:

- `scripts/audit_transport_shift_goal.py`.
- `tests/unit/test_audit_transport_shift_goal.py`.

Current audit result:

- Command: `/opt/anaconda3/bin/python scripts/audit_transport_shift_goal.py --output-json reports/research/sota_loop/transport_shift_goal_audit.json`.
- Status: `blocked_incompatible_splits`.
- `test_allowed`: `false`.
- Satisfied: real `light-v1` train/val evidence exists; train-only shift fit exists; current results are recorded.
- Failed: the train-fitted shift did not pass the validation SOTA guard.
- Blocked: no held-out test is permitted because the validation and split-compatibility preconditions are not met.
- Data schema: the real `light-v1` Advection train/val/test HDF5 files contain only `data` tensors, with no file attrs, dataset attrs, coefficient datasets, velocity metadata, or sample-aligned auxiliary parameter fields.

Why this matters:

- The original objective now has a reproducible requirement audit rather than an ambiguous narrative checkpoint.
- The audit prevents accidental benchmark leakage by explicitly disallowing a held-out test while train/val/test constant-shift support is incompatible.
- The metadata route is also closed for the current artifact: there is no benchmark-provided parameter field from which to learn a train-only shift extrapolator.
- This is not a positive SOTA result; it formalizes the blocker so the next step must be a benchmark-policy choice or a new learned/state-conditioned objective.

Recommendation:

- Stop spending cycles on constant train-fitted shift under the current official split.
- Either rebuild a compatible train/val/test benchmark with explicit approval, or define a new benchmark-clean learned/state-conditioned transport head gate.

Audit enforcement:

- `scripts/audit_transport_shift_goal.py --require-status test-ready` now exits nonzero unless the evidence permits the one held-out test.
- Against current artifacts it exits `2`, preserving the stop condition while status remains `blocked_incompatible_splits`.
- `--require-status achieved` is available for any future release gate that should pass only after the validation guard and the authorized held-out test are both recorded.
- `scripts/run_remote_transport_shift_candidate.sh` now runs that audit after all-split candidate gates, defaulting to `AUDIT_REQUIRE_STATUS=achieved`; this makes the remote/full-source path fail closed unless the complete benchmark-clean result is actually present.
- `scripts/run_official_transport_shift_audit.sh` is the local official refresh command. It reruns the train-only gate on real `light-v1` Advection train/val, passes test through the gated path only, and audits the resulting evidence. A report-only refresh still returns `blocked_incompatible_splits`, `test_eligible=false`, and no held-out test result.
- The audit now fingerprints the local real `light-v1` Advection files: train SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`, val SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`, test SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`.
- The audit now has explicit held-out test policy checks. Current official evidence has `test_result_count=0`, `leaked_test_result=false`, and `test_allowed_next=false`; future artifacts with test leakage or multiple held-out tests are classified as invalid instead of merely blocked.
- The local official runner now enforces those `light-v1` SHA-256 identities by default; current `data_identity_policy.passed=true`, so the remaining blocker is still validation/split incompatibility.
- The official local runner also requires complete identity coverage for every inspected existing split. Current `data_identity_policy.require_all_inspected_splits=true`, with no missing expected hashes and no mismatches.
- `scripts/run_transport_shift_gate.py` now writes `data_sources` into the gate artifact, so the train-only fit result itself records the exact train/val/test HDF5 paths, byte counts, and SHA-256 values it used.
- The audit cross-checks those gate `data_sources` against independently inspected HDF5 files. Current `data_identity_policy.gate_source_mismatches=[]`, so provenance is consistent while validation remains blocked.
- The default official promotion command `bash scripts/run_official_transport_shift_audit.sh` exits `2` on current evidence. This is the intended fail-closed behavior: provenance passes, but `status=blocked_incompatible_splits`, `test_eligible=false`, and no held-out test result is recorded.
- The official audit now enforces result recording in both `worklog.md` and this plan. Current `result_record_policy.passed=true` with required token `blocked_incompatible_splits`.
- The result-record audit now also requires the measured selected-validation NRMSE. Current `result_record_policy.passed=true` with required tokens `blocked_incompatible_splits` and `0.5140249729156494`, so status-only handoffs fail the official audit.

### 2026-05-19: Lagged Observed-Transition Transport Gate

Status: real local `light-v1` Advection validation passed and the guarded held-out test was measured for a state-conditioned, two-frame observed-context transport estimator.

Implemented:

- `scripts/run_observed_transport_shift_gate.py` now records `data_sources` fingerprints for train/val/test.
- `scripts/run_observed_transport_shift_gate.py` supports `--test-ledger-json` to prevent accidental repeated held-out test measurement for the same observed-gate configuration.
- `scripts/run_official_observed_transport_shift_gate.sh` is the local official observed-context transport command.
- `scripts/audit_observed_transport_shift_result.py` is the machine-readable observed-context result audit.
- `tests/unit/test_run_observed_transport_shift_gate.py` covers the source-fingerprint output.
- `tests/unit/test_run_observed_transport_shift_gate.py` covers first-write ledger behavior, repeat-test refusal, and explicit debugging repeat behavior.
- `tests/unit/test_official_observed_transport_shift_gate.py` covers the dry-run contract for the official observed command.
- `tests/unit/test_audit_observed_transport_shift_result.py` covers achieved, test-ready, leakage, result-record, and exit-policy cases.

Evidence:

- Command output artifact: `reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json` (ignored local report).
- Data identities: train SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`, val SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`, test SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`.
- Estimator: `lagged_observed_transition_shift`; train locks candidate shift support, validation does not select a split-level shift.
- Train NRMSE `0.014504759572446346`, shift mean/std `0.0` / `0.0`.
- Validation NRMSE `0.012846261262893677`, shift mean/std `40.0` / `0.0`, relative improvement `0.9582651427581705` versus reference `0.30780652221851373`; `validation_guard.passed=true`.
- Held-out test ran only after validation passed; test NRMSE `0.004225204233080149`, shift mean/std `72.0` / `0.0`.

Interpretation:

- This is a benchmark-clean result only under a two-frame observed-context policy where `t-1 -> t` is allowed context for predicting `t -> t+1`.
- It should not be conflated with the blocked constant train-fitted-shift objective, which remains `blocked_incompatible_splits`.
- If the final benchmark must be fully autonomous from the initial frame only, treat this as an upper-bound/state-signal result and train a causal transport head next.

Held-out test policy:

- Official observed-context command: `bash scripts/run_official_observed_transport_shift_gate.sh`.
- Safe command preview: `DRY_RUN=1 bash scripts/run_official_observed_transport_shift_gate.sh`.
- Future official observed-gate runs should pass `--test-ledger-json reports/research/sota_loop/observed_transport_shift_test_ledger.json`.
- The ledger key is derived from the estimator, candidate shifts, train/val/test fingerprints, split names, sample caps, rollout steps, metric, reference metric, and validation threshold.
- Reusing the same key fails before measuring held-out test again unless `--allow-repeat-test` is explicitly set for debugging; debugging repeats do not append another official ledger entry.
- The official observed command now audits the generated result with default `AUDIT_REQUIRE_STATUS=achieved`; report-only audit mode is available through `AUDIT_REQUIRE_STATUS=report`.
- A direct audit of the existing ignored observed gate artifact returned `status=achieved` without re-running held-out test. It verified data identity, no gate/source mismatches, validation pass, exactly one authorized test result, and result-record tokens `achieved`, `0.012846261262893677`, and `0.004225204233080149`.
- The existing ignored observed gate artifact predates the exactly-once ledger, so its audit has `held_out_test_policy.ledger=null`; future official reruns should use the ledger path.

### 2026-05-19: Train-Only Feature Diagnostic

Status: no-test train/val diagnostic found no train support for the validation transport regime using first-frame train-fitted features.

Evidence:

- Script: `scripts/diagnose_train_only_transport_features.py`.
- Output: `reports/research/sota_loop/train_only_transport_feature_diagnostic.json` (ignored local report).
- Train labels: `{"0": 128}`.
- Validation labels: `{"40": 32}`.
- Validation predictions from train-fitted centroids: `{"0": 32}`.
- Unsupported validation shifts: `[40]`.
- Validation accuracy: `0.0`.
- Conclusion: `blocked_no_train_support_for_validation_shift`.

Interpretation:

- The diagnostic uses train split only to fit the feature-to-shift rule, uses validation only for measurement, and does not read held-out test.
- A simple first-frame feature head cannot recover the validation shift because the train sample used here contains no shift-`40` support.
- The literal train-only transport-shift path remains blocked under current local `light-v1`; the remaining viable paths are a split-compatible shard rebuild, an accepted two-frame observed-context policy, or a richer causal mechanism with additional allowed signal.

Full local train/val rerun:

- Command used `--max-samples -1 --val-max-samples -1`, without reading held-out test.
- Output: `reports/research/sota_loop/train_only_transport_feature_diagnostic_full.json` (ignored local report).
- Train labels remained `{"0": 128}` and validation labels remained `{"40": 32}`.
- Validation predictions remained `{"0": 32}`, unsupported validation shifts remained `[40]`, and validation accuracy remained `0.0`.
- Train best-margin mean/min/max: `0.13357117772102356` / `0.037119414657354355` / `0.29049214720726013`.
- Validation best-margin mean/min/max: `0.11928332597017288` / `0.04894868656992912` / `0.22282543778419495`.
- This rules out the earlier 128-sample cap as the reason the train-only first-frame feature probe failed.

Objective-level audit:

- Script: `scripts/audit_transport_objective_status.py`.
- Official command: `bash scripts/run_official_transport_objective_status.sh`.
- Default command behavior requires literal objective status `literal-achieved`; it does not rerun gates and does not touch held-out test.
- Observed-context acceptance command: `ACCEPT_OBSERVED_CONTEXT=1 REQUIRE_STATUS=observed-accepted bash scripts/run_official_transport_objective_status.sh`.
- Literal mode output: `reports/research/sota_loop/transport_objective_status.json` (ignored local report), `status=literal_blocked`.
- Observed-accepted mode output: `reports/research/sota_loop/transport_objective_status_observed_accepted.json` (ignored local report), `status=observed_context_achieved`.
- Current literal blockers: constant train-only audit is `blocked_incompatible_splits`; train-only feature diagnostic is `blocked_no_train_support_for_validation_shift`; observed-context result is achieved but not accepted for the literal objective.
- Current observed-accepted caveat: literal train-only shift objective remains unproven and status depends on accepting two-frame observed context.
- Default official command refresh: `bash scripts/run_official_transport_objective_status.sh` exits `2` with `status=literal_blocked`.
- Observed-context command refresh: `ACCEPT_OBSERVED_CONTEXT=1 REQUIRE_STATUS=observed-accepted OBJECTIVE_STATUS_JSON=reports/research/sota_loop/transport_objective_status_observed_accepted.json bash scripts/run_official_transport_objective_status.sh` exits `0` with `status=observed_context_achieved`.
- Both command modes read existing evidence only; neither reruns gates or touches held-out test.

Temporal-window diagnostic:

- Script: `scripts/diagnose_transport_temporal_windows.py`.
- Output: `reports/research/sota_loop/transport_temporal_window_diagnostic.json` (ignored local report).
- It scans train/val temporal start windows only and does not read held-out test.
- Full local train/val result over 16-step windows starting at `0,16,32,...,176`: train labels `{"0": 12}`, validation labels `{"40": 12}`, common temporal best shifts `[]`, conclusion `blocked_no_temporal_common_shift`.
- This rules out the plausible temporal-offset rescue path for the literal train-only objective under the current local `light-v1` shards.

### 2026-05-19: Two-Frame Context Transport Gate

Status: real local `light-v1` Advection validation passed and the guarded held-out test was measured for a two-frame context transport estimator.

Implemented:

- `scripts/run_context_transport_shift_gate.py` gates a context estimator that reads `t0 -> t1` once, estimates the sample shift, and then rolls out autoregressively without reading future observed transitions.
- `tests/unit/test_run_context_transport_shift_gate.py` covers validation-pass test measurement, validation-fail blocking, one-shot held-out test ledger behavior, and explicit debugging repeat behavior.

Evidence:

- Command output artifact: `reports/research/sota_loop/context_transport_shift_gate.json` (ignored local report).
- Ledger artifact: `reports/research/sota_loop/context_transport_shift_test_ledger.json` (ignored local ledger).
- Data identities: train SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`, val SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`, test SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`.
- Estimator: `two_frame_context_shift`; train locks candidate shift support and the estimator contract, validation does not select a split-level model.
- Train NRMSE `0.13868984580039978`, shift mean/std `0.0` / `0.0`.
- Validation NRMSE `0.12336619943380356`, shift mean/std `40.0` / `0.0`, relative improvement `0.5992086244805913` versus reference `0.30780652221851373`; `validation_guard.passed=true`.
- Held-out test ran only after validation passed; test NRMSE `0.040703773498535156`, shift mean/std `72.0` / `0.0`.
- Ledger measurement key: `aad43ee28e0606013d01f8fcbfb525ea406a41925d16fbcf12084aecfeca2d06`.

Interpretation:

- This is a stricter state-conditioned result than the lagged observed-transition gate because it does not read future observed transitions after the two-frame context seed.
- It is not the literal one-frame train-only constant-shift objective; the literal objective remains blocked by train/val/test regime incompatibility.
- If the benchmark policy allows two initial context frames, this is the highest-signal current result: validation-clean, provenance-fingerprinted, ledger-guarded, and tested once.

Audit enforcement:

- `scripts/audit_context_transport_shift_result.py` is the machine-readable result audit for the two-frame context gate.
- `scripts/run_official_context_transport_shift_gate.sh` is the official context-gate command; it measures held-out test only after validation passes and uses `reports/research/sota_loop/context_transport_shift_test_ledger.json` to block accidental repeats.
- A direct audit of the existing ignored context gate artifact returned `status=achieved` without re-running held-out test. It verified data identity, no gate/source mismatches, validation pass, exactly one authorized held-out test result, and result-record tokens `achieved`, `0.12336619943380356`, and `0.040703773498535156`.
- `scripts/run_official_transport_objective_status.sh` now supports `ACCEPT_CONTEXT_TRANSPORT=1 REQUIRE_STATUS=context-accepted` for explicit two-frame context policy acceptance.
- The context-accepted objective command returned `status=context_transport_achieved`.
- The default literal objective command still exits `2` with `status=literal_blocked`; the context result is achieved but not accepted for the literal objective unless the policy flag is set.

### 2026-05-19: Train-Only Identifiability Audit

Status: the literal train-only shift objective is underidentified on the current real local `light-v1` train/val shards.

Evidence:

- Script: `scripts/audit_train_only_transport_identifiability.py`.
- Output: `reports/research/sota_loop/train_only_transport_identifiability_audit.json` (ignored local report).
- The audit uses train and validation only; it does not read held-out test.
- Full-shard train shift support is `[0]`.
- Full-shard validation shift support is `[40]`.
- Unsupported validation shifts are `[40]`.
- Status: `blocked_underidentified_train_only_shift`.

Interpretation:

- A supervised train-only shift-label learner has no evidence for selecting validation shift `40`, because train contains only shift `0`.
- This is stronger than a failed feature probe: it says the label support needed for literal train-only shift extrapolation is absent from the current shard.
- Recent operator-learning directions such as equivariant Fourier operators and adaptive coordinate transforms are relevant for architecture design, but they do not remove the need for either train regime support, allowed context, parameter metadata, or split-compatible data.

Objective audit impact:

- `scripts/run_official_transport_objective_status.sh` now reads `reports/research/sota_loop/train_only_transport_identifiability_audit.json`.
- Default literal mode still exits `2` with `status=literal_blocked`, now explicitly including `blocked_underidentified_train_only_shift`.
- Context-accepted mode still exits `0` with `status=context_transport_achieved`.
