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

### 2026-05-19: Benchmark-Clean Hydration Options Audit

Status: the current workspace does not have an additional benchmark-clean local Advection train source that covers validation shift `40`; official remote raw Advection train files exist but are not hydrated.

Evidence:

- Script: `scripts/audit_transport_data_hydration_options.py`.
- Output: `reports/research/sota_loop/transport_data_hydration_options.json` (ignored local report).
- The audit reads train/val only for local support and does not read held-out test.
- Canonical local root: `data/pdebench`.
- Canonical local train support: `[0]`.
- Canonical local validation support: `[40]`.
- Status: `remote_official_hydration_required`.
- Official remote Advection train files in `docs/pdebench_manifest.yaml`: `8`.
- Total official remote Advection train size: `61.34038382768631` GiB.
- Synthetic `reports/light_experiments/**/synthetic_pdebench` shards are cataloged as `synthetic_report_artifact_not_benchmark_clean`.

Interpretation:

- The literal objective cannot be rescued from already-local benchmark-clean shards in this workspace.
- The only benchmark-clean data path found is to hydrate official raw Advection train data from the manifest or build an explicitly approved split-compatible benchmark.
- Synthetic report shards are useful for debugging but should not be used as release evidence.

Objective audit impact:

- `scripts/run_official_transport_objective_status.sh` now reads `reports/research/sota_loop/transport_data_hydration_options.json`.
- Default literal mode still exits `2` with `status=literal_blocked`, now explicitly including `remote_official_hydration_required`.

### 2026-05-19: Official Advection Hydration Plan

Status: the literal train-only path now has an executable official-data hydration plan, but the downloads have not been run.

Evidence:

- Script: `scripts/plan_transport_official_hydration.py`.
- Output: `reports/research/sota_loop/official_advection_hydration_plan.json` (ignored local report).
- Plan status: `ready_for_explicit_hydration`.
- Selected official Advection train files: `8`.
- Estimated download size: `61.34038382768631` GiB.
- Planned raw root: `data/pdebench/raw`.
- Planned hydrated source root: `data/pdebench_official_advection_hydrated`.
- Planned light train/val root: `data/pdebench_official_advection_light`.
- Planned train/val counts: train `256`, val `64`.
- Planned test count: `0`.

Policy:

- Downloads require explicit approval because they use network and large disk.
- The plan downloads official train files only.
- The plan does not download or shard held-out test data.
- The held-out test remains allowed only through the validation-gated transport command after validation passes.

Objective audit impact:

- `scripts/run_official_transport_objective_status.sh` now reads `reports/research/sota_loop/official_advection_hydration_plan.json`.
- Default literal mode still exits `2` with `status=literal_blocked`, now explicitly showing `ready_for_explicit_hydration` as the next literal-path action.

Plan validation:

- Script: `scripts/validate_transport_hydration_plan.py`.
- Output: `reports/research/sota_loop/official_advection_hydration_plan_validation.json` (ignored local report).
- Validation status: `valid`.
- It verifies selected paths match download commands, all selected paths are official `1D/Advection/Train` files, no held-out test split is downloaded or sharded, the train/val shard command uses `--test-count 0`, the validation command does not pass `--test-split`, and synthetic report artifacts are not referenced.
- `scripts/run_official_transport_objective_status.sh` now reads the validation artifact; default literal mode still exits `2` with `status=literal_blocked` until the approved download and validation run actually happen.

Dry-run execution:

- Script: `scripts/run_transport_official_hydration_plan.py`.
- Output: `reports/research/sota_loop/official_advection_hydration_plan_run.json` (ignored local report).
- Run status: `dry_run`.
- The runner validates the plan, lists download/convert/shard/validate/audit stages, and executes nothing unless `--execute` is provided.
- The download stage additionally requires `--execute-downloads`; the dry run records blocker `download stage requires --execute-downloads`.
- `scripts/run_official_transport_objective_status.sh` now reads the run artifact; default literal mode still exits `2` with `status=literal_blocked` until the staged hydration actually runs and validation passes.

Disk preflight:

- Script: `scripts/preflight_transport_hydration.py`.
- Output: `reports/research/sota_loop/official_advection_hydration_preflight.json` (ignored local report).
- Preflight status: `blocked_insufficient_disk`.
- Raw official files present: `0` of `8`.
- Remaining download bytes: `65863735616`.
- Required free bytes with safety factor `1.15`: `75743295958`.
- Current free bytes at `data/pdebench/raw`: `1599369216`.
- `scripts/run_official_transport_objective_status.sh` now reads the preflight artifact; default literal mode still exits `2` with `status=literal_blocked` until enough disk is available and the official hydration stages run.

Storage recommendation:

- Script: `scripts/recommend_transport_hydration_storage.py`.
- Output: `reports/research/sota_loop/official_advection_hydration_storage_recommendation.json` (ignored local report).
- Status: `external_or_freed_space_required`.
- Candidate roots checked: `data/pdebench/raw`, `/private/tmp`, `/Volumes`.
- All checked roots resolve to the same filesystem and had `1588920320` free bytes at audit time.
- Required free bytes remain `75743295958`.
- Next literal-path action is to free disk or mount a larger volume, then regenerate the hydration plan with `--raw-out`, `--hydrated-source-root`, and `--hydrated-light-root` pointing to that storage root.
- `scripts/run_official_transport_objective_status.sh` now reads the storage recommendation artifact; default literal mode still exits `2` with `status=literal_blocked`.

Remote execution plan:

- Script: `scripts/plan_remote_official_hydration.py`.
- Output: `reports/research/sota_loop/remote_official_advection_hydration_plan.json` (ignored local report).
- Status: `ready_for_remote_hydration`.
- Required remote disk: `120` GB.
- The plan uses `scripts/launch_remote_transport_shift_candidate_vast.sh` with `REMOTE_SCRIPT=scripts/run_remote_official_hydration.sh`.
- It is dry-run first and does not start paid compute unless `DRY_RUN=0` is set.
- The bash wrapper calls the staged hydration runner, requires `EXECUTE_DOWNLOADS=1`, downloads official train files only, builds train/val shards with `test_count=0`, and keeps held-out test gated behind validation.
- `scripts/run_official_transport_objective_status.sh` now reads the remote plan artifact; default literal mode still exits `2` with `status=literal_blocked` until the remote hydration and validation actually run.

Objective evidence wiring:

- Script updates: `scripts/audit_transport_objective_status.py`, `scripts/run_official_transport_objective_status.sh`.
- New evidence input: `reports/research/sota_loop/official_hydrated_transport_shift_gate.json`.
- The official objective audit now recognizes `literal_test_ready` when the official hydrated train/val gate passes and no held-out test result has been recorded.
- `literal_test_ready` is an intermediate state, not completion. It means validation has authorized the next action: run exactly one held-out test through the gated transport path, then promote through the constant goal audit.
- Default literal release mode remains `REQUIRE_STATUS=literal-achieved`, so the command still fails closed until the final held-out test and result-record audit pass.

Post-validation audit boundary:

- `scripts/plan_transport_official_hydration.py` now sets `objective_audit_after_validation` to `REQUIRE_STATUS=literal-test-ready bash scripts/run_official_transport_objective_status.sh`.
- `scripts/validate_transport_hydration_plan.py` rejects train/val-only hydration plans that require final `literal-achieved` status immediately after validation.
- This keeps the remote hydration job from being marked failed solely because it correctly stopped before the held-out test. The next phase is only authorized if the objective status reaches `literal_test_ready`.

Guarded post-validation test phase:

- Script: `scripts/run_official_hydrated_post_validation_test.py`.
- Output: `reports/research/sota_loop/official_hydrated_post_validation_test_run.json` (ignored local report).
- Current preview status: `dry_run` with blockers because the objective is still `literal_blocked`, not `literal_test_ready`.
- The runner refuses to build/read held-out test unless the objective status artifact is already `literal_test_ready` and the command is explicitly run with `--execute --execute-test`.
- When authorized, it builds only the official hydrated test shard from `data/pdebench_official_advection_hydrated/advection1d_train.h5` using `--split-start-index test=320`, then reruns the transport gate with `--test-split test`.

Remote full-objective chain:

- `scripts/run_remote_official_hydration.sh` can now chain the guarded post-validation phase after the train/val hydration runner.
- `scripts/plan_remote_official_hydration.py` now includes `RUN_POST_VALIDATION_TEST=1`, `EXECUTE_TEST=1`, and `POST_VALIDATION_TEST_JSON=reports/research/sota_loop/official_hydrated_post_validation_test_run.json` in the generated Vast launcher command.
- This makes the remote job capable of completing the full objective in one execution if, and only if, the train/val gate first reaches `literal_test_ready`.
- Fresh remote checkouts regenerate `reports/research/sota_loop/official_advection_hydration_plan.json` inside `scripts/run_remote_official_hydration.sh` when the ignored local plan artifact is missing.

Remote download throughput pivot:

- Vast contract `37096575` confirmed the fresh-checkout plan regeneration fix, but then stalled in the first official-file download path with no runner-stage output after plan emission.
- A local probe of the official Dataverse/S3 redirect path showed ranged GET works, but observed throughput for a 1 MiB range sample was only about `0.45 MB/s`.
- Serial `requests.get(..., stream=True)` download is therefore too fragile and too slow for the `61.34` GiB official hydration path.
- `scripts/download_pdebench_file.py` now defaults to resumable ranged download parts with configurable `PDEBENCH_DOWNLOAD_WORKERS`, `PDEBENCH_DOWNLOAD_PART_SIZE_MIB`, and `PDEBENCH_DOWNLOAD_RETRIES`.
- This keeps the benchmark-clean data source unchanged while improving execution reliability and telemetry. The tradeoff is more HTTP requests against the official host; the range size default is intentionally coarse (`256` MiB) to avoid tiny-part request storms.
- Vast contract `37097600` confirmed this path can save complete official files quickly, but one range stalled near the end of the third file.
- The downloader now also supports `PDEBENCH_DOWNLOAD_PART_TIMEOUT` / `--part-timeout`, so a single slow or hung range is retried instead of holding the whole hydration run open indefinitely.
- Vast contract `37098407` confirmed the timeout retry works on repeated slow ranges and progressed through all `beta7.0` range parts, but failed during final assembly with `OSError: [Errno 28] No space left on device`.
- The failure was storage overhead, not a benchmark or validation result: the prior downloader held all part files and then wrote a second full-size assembled temp file.
- `scripts/download_pdebench_file.py` now writes each completed range directly into a preallocated `.tmp` destination at the correct byte offset, then atomically replaces the final file after all ranges complete.
- Tradeoff: direct offset writes are less inspectable than persisted part files, but the post-download checksum remains the integrity gate and peak per-file temporary storage drops enough for a 120 GB remote run.

Official stratified split pivot:

- Vast contract `37101416` proved the official hydration path can complete the large data step with the direct-to-temp downloader on a 160 GB instance.
- The run downloaded all 8 official Advection train files, converted the source, built train/val shards, and ran the train-only transport gate.
- The validation guard failed (`validation_guard.passed=false`, `test_eligible=false`), so no held-out test ran.
- The failure exposed a split-construction confound:
  - sorted official beta files were concatenated during conversion
  - contiguous train/val windows sampled different beta regimes
  - a constant train-fitted shift selected on train did not match the high-beta validation window
- The next benchmark-clean path is not to loosen the guard. It is to build beta-balanced light shards from the official train files:
  - convert `48` samples per beta file
  - train uses `32` samples per beta file (`256` total)
  - validation uses the next `8` samples per beta file (`64` total)
  - held-out test reserves the next `8` samples per beta file (`64` total), but remains unbuilt/unread until `literal_test_ready`
- Implemented support:
  - `scripts/make_light_hdf5_shards.py` supports `--split-block-size` and `--split-block-offset SPLIT=OFFSET`
  - `scripts/plan_transport_official_hydration.py` emits the stratified official train/val plan
  - `scripts/run_official_hydrated_post_validation_test.py` uses the reserved stratified test block only after the objective audit reports `literal_test_ready`
- This keeps the literal objective intact: fit only on train, validate on val, run exactly one held-out test only if validation passes.

Remote wrap-up checkpoint:

- Vast contract `37157238` was launched with the stratified official hydration plan and the guarded post-validation chain enabled.
- The remote plan confirmed the intended split contract: `48` samples per official train beta file, train offset `0`, validation offset `32`, reserved test offset `40`, and no test split built during train/val hydration.
- The run was stopped before validation by operator request after saving 5 of 8 official train files and while downloading `beta2.0`.
- Contract `37157238` was destroyed, and `vastai show instances --raw` returned `[]`.
- This is a partial hydration attempt only: no SOTA guard validation ran, no held-out test ran, and no benchmark conclusion should be drawn from it.

Remote launch/network checkpoint:

- Subsequent remote attempts showed three distinct Vast/runtime failure modes before any benchmark result:
  - stopped/no-container contracts from stale or unavailable explicit offers
  - host-side image pull failure from full container storage
  - remote outbound network outage to `darus.uni-stuttgart.de` during the first official file download
- The only attempt that reached repo execution and the official downloader was Nevada contract `37168284`; it failed before completing the first file and before conversion, validation, or any held-out test.
- The downloader now has configurable exponential backoff between ranged-part retries via `PDEBENCH_DOWNLOAD_RETRY_BACKOFF` / `--retry-backoff`.
- Next path remains the same benchmark-clean objective: relaunch the stratified official hydration on a host with working image/container storage and stable outbound network, then accept only a real train-only validation gate result as evidence.

Current Vast wrap-up checkpoint:

- Vast contract `37169407` reached the official downloader on a Netherlands RTX 4090 instance with the stratified official hydration plan and guarded post-validation chain enabled.
- The run confirmed the benchmark-clean split contract again: `48` samples per official train beta file, train offset `0`, validation offset `32`, reserved test offset `40`, and no test split during train/val hydration.
- Last available logs showed 5 of 8 official train files saved and `beta2.0` near completion; the retry-backoff path recovered slow ranged parts rather than failing immediately.
- The contract was destroyed by operator request before conversion, validation, or held-out test.
- This is another partial hydration attempt only. No SOTA guard validation ran, no held-out test ran, and no benchmark conclusion should be drawn from it.

Patched downloader wrap-up checkpoint:

- Vast contract `37176828` reached the first official train-file download but stalled at `61/62` ranged parts (`7.54 GiB`, `98.4%`) and was destroyed.
- `scripts/download_pdebench_file.py` now applies the configured per-part timeout to the underlying ranged HTTP request read timeout, not only to elapsed time after chunks start yielding.
- Vast contract `37177098` relaunched the same benchmark-clean stratified hydration on a Spain RTX 4090 host with `python:3.11-slim`, `8` workers, `128 MiB` parts, `6` retries, `20s` retry backoff, and `180s` per-part timeout.
- The run proved the lighter image path can bootstrap the repo and official downloader without PyTorch, and the patched downloader completed all `62/62` ranges for `beta0.1`, including the likely prior stuck range.
- The run advanced into the second official train file and reached at least `54/62` ranged parts (`6.75 GiB`, `88.0%`) before operator-requested wrap-up.
- Contract `37177098` was destroyed before conversion, validation, or held-out test.
- This remains a partial hydration attempt only: no SOTA guard validation ran, no held-out test ran, and no benchmark conclusion should be drawn from it.

Adaptive range-split checkpoint:

- Vast contract `37177336` relaunched the official stratified hydration on a Virginia RTX 4090 host with the guarded post-validation chain still enabled.
- The run saved five complete official train files (`beta0.1`, `beta0.2`, `beta0.4`, `beta0.7`, `beta1.0`) and reached `98.4%` of `beta2.0`.
- Multiple slow ranges recovered through the patched `180s` per-part read timeout and retry path, but one `128 MiB` range (`5368709120-5502926847`) failed three consecutive attempts.
- Contract `37177336` was destroyed after the repeated same-range stall, before conversion, validation, or held-out test.
- `scripts/download_pdebench_file.py` now adaptively splits repeatedly timed-out ranges into smaller subranges:
  - `PDEBENCH_DOWNLOAD_SPLIT_AFTER_RETRIES` / `--split-after-retries`
  - `PDEBENCH_DOWNLOAD_MIN_SPLIT_SIZE_MIB` / `--min-split-size-mib`
  - defaults split after `2` failures down to `8 MiB` minimum ranges
- The purpose is to keep the benchmark-clean official source while avoiding pathological single-range stalls. This does not alter the train-only fit, validation-first guard, or held-out-test policy.
- Verification:
  - `python -m pytest tests/unit/test_download_pdebench_file.py` -> `10 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`
- Next path remains: relaunch the official stratified hydration with the adaptive split downloader, then accept only the resulting validation gate as benchmark evidence; run exactly one held-out test only if the audit reaches `literal_test_ready`.

Credit-blocked relaunch checkpoint:

- Vast contract `37178922` launched on California offer `35149296` with the adaptive range-split downloader and the same guarded post-validation chain.
- The bootstrap and official plan regeneration succeeded, and the first official train file reached `60/62` parts (`7.42 GiB`, `96.7%`).
- The instance then stopped unexpectedly before completing the first file. The public log ended without a Python traceback.
- Stopped-instance inspection showed only:
  - `data/pdebench/raw/1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5.tmp`
  - `reports/research/sota_loop/official_advection_hydration_plan.json`
- There was no completed official file, validation artifact, or held-out test artifact. Contract `37178922` was destroyed.
- A relaunch attempt on Texas offer `35956477` was rejected by Vast with `Your account lacks credit; see the billing page.`
- `vastai show instances --raw` returned `[]` after cleanup.
- The current blocker is external compute credit or an alternate real-data execution path, not a benchmark result.
- Next path: once compute is available, relaunch the same adaptive range-split stratified official hydration; do not run held-out test unless the objective audit reaches `literal_test_ready`.

Local light-v1 policy crossroads refresh:

- The local Advection `light-v1` files match the recorded B2/remote manifest identities:
  - train SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`
  - validation SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`
  - test SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`
- Fresh default objective audit still exits `2` with `status=literal_blocked`.
- Fresh context-accepted objective audit exits `0` with `status=context_transport_achieved`, but only with `ACCEPT_CONTEXT_TRANSPORT=1`.
- Fresh context and observed result audits both return `status=achieved` without rerunning held-out test.
- The key crossroads is policy, not data availability:
  - accept two initial context frames: promote the context transport result (`val nrmse=0.12336619943380356`, `test nrmse=0.040703773498535156`, exactly-one test ledger recorded)
  - require literal constant/train-only shift: continue the official stratified hydration path once compute credit is available, or pursue a richer train-only causal mechanism with additional allowed signal

Context-accepted audit consistency fix:

- `scripts/audit_transport_objective_status.py` now marks `fit_transport_shift_only_on_train` as satisfied when the context or observed accepted-policy result is achieved.
- This fixes an aggregate-audit inconsistency: `status=context_transport_achieved` no longer coexists with a blocked train-fit requirement row.
- Default literal policy is unchanged and still fails closed with `status=literal_blocked`.
- Verification:
  - `python -m pytest tests/unit/test_audit_transport_objective_status.py` -> `6 passed`
  - `bash scripts/run_official_transport_objective_status.sh` -> exits `2`, `status=literal_blocked`
  - `ACCEPT_CONTEXT_TRANSPORT=1 REQUIRE_STATUS=context-accepted OBJECTIVE_STATUS_JSON=/private/tmp/transport_objective_status_context_accepted_fixed.json bash scripts/run_official_transport_objective_status.sh` -> exits `0`, `status=context_transport_achieved`, all requirement rows satisfied under the explicit two-frame context policy

Context-accepted wrap-up wrapper:

- `scripts/run_official_context_transport_objective_status.sh` now provides a single explicit context-policy entrypoint for the achieved two-frame context result.
- It sets `ACCEPT_CONTEXT_TRANSPORT=1`, `REQUIRE_STATUS=context-accepted`, and writes `reports/research/sota_loop/transport_objective_status_context_accepted.json` by default.
- It does not rerun validation or held-out test; it only re-aggregates the existing audits and exactly-one test ledger under the context-accepted benchmark policy.
- The default literal command remains the conservative release gate and still exits nonzero with `status=literal_blocked`.
- Next path:
  - if two initial context frames are accepted as benchmark-clean, use the wrapper as the release/checkpoint command
  - if the literal train-only constant-shift objective remains required, wait for compute credit or an alternate real-data path and relaunch official stratified hydration; do not run held-out test unless the objective audit reaches `literal_test_ready`

Literal hydration robustness update:

- `scripts/download_pdebench_file.py` now records completed ranged-download parts in a `.tmp.ranges.json` sidecar and resumes them when the same official file download is restarted with a matching sparse `.tmp` file.
- This keeps the official-source path intact while reducing the cost of the repeated near-complete Advection file failures seen on Vast.
- The sidecar is intentionally conservative: mismatched or missing range metadata causes a clean restart rather than trusting unknown partial bytes.
- Verification:
  - `python -m pytest tests/unit/test_download_pdebench_file.py` -> `11 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`

Current relaunch blocker and updated command:

- `vastai show instances --raw` returns `[]`.
- A fresh launch attempt on verified Norway RTX 4090 offer `36114274` was rejected by Vast with `Your account lacks credit; see the billing page.`
- `scripts/plan_remote_official_hydration.py` now embeds the hardened downloader runtime into the generated launcher command:
  - 8 workers
  - 128 MiB ranged parts
  - 6 retries
  - 180s per-part timeout
  - 20s exponential retry backoff
  - split repeatedly failing ranges after 2 attempts down to 8 MiB
- `reports/research/sota_loop/remote_official_advection_hydration_plan.json` has been regenerated with these settings.
- Next path remains: once compute credit is available, run the generated `actual_launcher`, accept only the official hydrated train/val guard as validation evidence, and run held-out test only if the objective audit reaches `literal_test_ready`.

Official hydrated held-out test ledger:

- `scripts/run_official_hydrated_post_validation_test.py` now uses `reports/research/sota_loop/official_hydrated_transport_shift_test_ledger.json` by default.
- It derives a measurement key from the official hydrated test configuration and blocks repeat measurements before building or reading the test shard unless `--allow-repeat-test` is explicitly set.
- After command execution, it reads `official_hydrated_transport_shift_gate.json` and requires exactly one held-out test result before reporting `status=executed`.
- Verification:
  - `python -m pytest tests/unit/test_run_official_hydrated_post_validation_test.py tests/unit/test_run_remote_official_hydration.py` -> `9 passed`
  - `python -m py_compile scripts/run_official_hydrated_post_validation_test.py`

Remote wrapper objective-status consistency:

- `scripts/run_remote_official_hydration.sh` now exports `OBJECTIVE_STATUS_JSON`.
- The validation audit command executed inside `scripts/run_transport_official_hydration_plan.py` therefore writes the same objective-status artifact that `scripts/run_official_hydrated_post_validation_test.py` later reads.
- This removes a custom-path mismatch risk in the full remote chain.
- Verification:
  - `python -m pytest tests/unit/test_run_remote_official_hydration.py tests/unit/test_run_transport_official_hydration_plan.py tests/unit/test_run_official_hydrated_post_validation_test.py` -> `13 passed`
  - `bash -n scripts/run_remote_official_hydration.sh`

Preferred Vast offer relaunch update:

- Current direct launch attempt on Mexico RTX 4090 offer `36151271` was rejected with `Your account lacks credit; see the billing page.`
- `scripts/plan_remote_official_hydration.py` now supports `--offer-id`.
- `reports/research/sota_loop/remote_official_advection_hydration_plan.json` has been regenerated with `preferred_offer_id=36151271`, so the next executable command is pinned to the best current network/disk candidate instead of relying on a fresh implicit search.
- Verification:
  - `python -m pytest tests/unit/test_plan_remote_official_hydration.py` -> `3 passed`
  - `python -m py_compile scripts/plan_remote_official_hydration.py`

Official hydration artifact preservation:

- `scripts/run_remote_official_hydration.sh` now supports `PUBLISH_ARTIFACTS=1`.
- When enabled, it publishes a tarball of the official hydration report artifacts to B2 before auto-shutdown, including the train/val run report, objective status, post-validation test report, gate JSON, and test ledger if present.
- This is intentionally report-only; it avoids uploading the large hydrated data by default.
- Verification:
  - `python -m pytest tests/unit/test_run_remote_official_hydration.py tests/unit/test_plan_remote_official_hydration.py` -> `8 passed`
  - `bash -n scripts/run_remote_official_hydration.sh`

Canonical remote launcher artifact publishing:

- A fresh pinned Vast launch attempt for offer `36151271` was still rejected with `Your account lacks credit; see the billing page.`
- `scripts/plan_remote_official_hydration.py` now emits `PUBLISH_ARTIFACTS=1` in both generated launcher commands.
- The regenerated remote hydration plan keeps the same pinned offer, hardened downloader runtime, and guarded post-validation test chain, but now also enables report publication before auto-shutdown.
- This does not complete the literal objective; it removes a launch-command footgun so the next credit-unblocked official hydration run can preserve benchmark evidence automatically.

Remote artifact publishing bootstrap:

- `scripts/run_remote_official_hydration.sh` now installs `rclone` on apt-based hosts when `PUBLISH_ARTIFACTS=1`, `INSTALL_RCLONE=1`, and `rclone` is missing.
- This matches the practical remote execution path: a minimal Vast image should not complete validation/test and then fail final evidence upload only because `rclone` was absent.
- Tradeoff: non-apt hosts or `INSTALL_RCLONE=0` still fail closed and require a preinstalled `rclone`, which is safer than silently skipping report publication.

Train-only conditional transport preparation:

- The literal objective remains blocked until official hydrated train/val validation exists; the current local light-v1 train split is still underidentified for the validation regime.
- A quick research pass points toward symmetry/canonical/equivariant transport structure rather than further global constant-shift sweeps.
- `src/ups/data/convert_pdebench.py` now preserves source provenance in converted HDF5 files via `source_file_index`, `source_sample_index`, and `source_paths`.
- This is intentionally not a result claim. It prepares the official hydrated train/val artifacts for a benchmark-clean conditional transport probe once real data hydration runs, while preserving the no-test-before-validation policy.

Source-conditioned official validation gate:

- `scripts/run_source_conditioned_transport_shift_gate.py` now implements the provenance-conditioned probe.
- It fits shifts only on train rows grouped by `source_file_index`, validates that locked map on val rows, and blocks if val contains an unseen source regime.
- The official hydration plan now uses this gate for `validate_without_test`, still writes `official_hydrated_transport_shift_gate.json`, and still keeps `test_count=0` during train/val hydration.
- This makes the next official run more likely to answer the real question: whether train-only source-regime conditioning can pass the SOTA guard without touching held-out test data.

Post-validation held-out test consistency:

- `scripts/run_official_hydrated_post_validation_test.py` now invokes the same source-conditioned gate if validation reaches `literal_test_ready`.
- The ledger key changed from the old constant-shift estimator identity to `official_hydrated_source_conditioned_transport_shift`.
- This prevents a model switch between validation and the exactly-one held-out test stage.

Light-shard provenance propagation:

- `scripts/make_light_hdf5_shards.py` now carries root HDF5 attrs from converted source files into the light shards.
- This preserves `source_paths` next to the sample-aligned `source_file_index` and `source_sample_index` datasets, making source-conditioned gate reports auditable back to the official beta files.

Official hydrated achievement promotion:

- `scripts/audit_transport_objective_status.py` now treats a passed official hydrated train/val gate plus exactly one official hydrated held-out test result as `literal_achieved`.
- This removes the stale dependency on the older constant-shift goal audit for the source-conditioned official path.
- The audit still reports `literal_test_ready` before the held-out test and still blocks when the official hydrated gate is missing.

Sequential official hydration path:

- `scripts/hydrate_official_advection_source_sequential.py` now supports the lower-disk remote route: download one official Advection train file, append sampled rows to the hydrated source HDF5 with provenance, optionally delete the raw file, then continue.
- `scripts/run_remote_official_hydration.sh` can run this path with `SEQUENTIAL_HYDRATION=1` before the existing shard/validate/audit stages and guarded post-validation test.
- `scripts/plan_remote_official_hydration.py` now emits sequential launcher args by default; the regenerated plan requires `DISK_GB=32` instead of 120 GiB because it no longer needs all raw train files resident at once.
- The tradeoff is that execution remains network-bound and still needs a credit-unblocked remote or other large-data environment, but the literal benchmark policy is unchanged: train/val first, `literal_test_ready` before any held-out test, exactly one test result afterward.

Sequential Vast relaunch check:

- The literal audit still reports `literal_blocked`; the official hydrated gate artifact is still missing.
- A cheaper sequential RTX 4090 route is now available in the search results: offer `8936321` had 59 GiB disk at about `$0.401/hr`, enough for the 32 GiB sequential plan.
- The actual launch attempt used `DISK_GB=32`, `SEQUENTIAL_HYDRATION=1`, `SEQUENTIAL_CLEANUP_RAW=1`, `RUN_POST_VALIDATION_TEST=1`, `EXECUTE_TEST=1`, and `PUBLISH_ARTIFACTS=1`.
- Vast rejected the create request with `Your account lacks credit; see the billing page.`
- A later Vast instance-state refresh failed with DNS resolution for `console.vast.ai`, so do not treat this turn as a fresh proof that no instances exist. The prior successful check before the attempt showed no active instances.
- Next path remains external: restore Vast credit or provide another real-data execution environment, then run the generated sequential actual launcher and accept only the official hydrated train/val gate as validation evidence before the held-out test.

Sample-mode source-conditioned gate:

- `scripts/run_source_conditioned_transport_shift_gate.py` now supports `--fit-strategy aggregate|sample_mode`.
- `sample_mode` estimates each train trajectory's best periodic transport shift, then selects the modal train-supported shift per `source_file_index`; ties use the train metric, absolute shift size, then numeric shift for deterministic behavior.
- The official hydration plan now uses `--fit-strategy sample_mode` for `validate_without_test`.
- This is a stricter train-only canonicalization-style candidate than a single aggregate group fit. It should be more robust when a beta/source group contains mixed or noisy trajectories, while keeping the benchmark policy unchanged: no validation fitting, no test shard during train/val hydration, and exactly one held-out test only after `literal_test_ready`.

Train-only local shift refinement:

- `scripts/run_source_conditioned_transport_shift_gate.py` now supports `--refine-radius`.
- The official hydration plan uses `--fit-strategy sample_mode --refine-radius 4`, so the train fit starts from the coarse configured grid and then checks a local integer neighborhood around train-selected shifts.
- This reduces coarse-grid quantization risk for the official Advection gate without fitting on validation or reading held-out test data.
- The regenerated train/val plan still writes only `official_hydrated_transport_shift_gate.json`; the held-out test remains a separate post-validation step gated by `literal_test_ready`.

Post-validation estimator parity:

- `scripts/run_official_hydrated_post_validation_test.py` now passes the same estimator knobs to the guarded held-out test command: `--fit-strategy sample_mode --refine-radius 4`.
- The held-out test ledger key now includes `fit_strategy` and `refine_radius`, so exactly-one-test enforcement is scoped to the actual estimator configuration.
- The regenerated post-validation dry run remains blocked until `literal_test_ready`, but its command no longer differs from the official train/val validation estimator.

Sequential hydration provenance hardening:

- `scripts/hydrate_official_advection_source_sequential.py` now writes root `source_paths` before downloading/appending the first official raw file.
- It also writes `sequential_hydration_complete=False` while appends are in progress and flips it to `True` only after every planned official train file has been appended.
- This makes partial remote artifacts auditable after host/network failures without promoting them as benchmark evidence.

Incomplete hydration shard guard:

- `scripts/make_light_hdf5_shards.py` now refuses to build light shards from any HDF5 source marked `sequential_hydration_complete=False`.
- This fail-closed check protects both official train/val validation shard creation and the later held-out test shard creation from accidentally consuming a partial remote hydration artifact.

Remote git-ref pinning:

- `scripts/plan_remote_official_hydration.py` now emits `GIT_REF=codex/sota-learned-gate` in the generated Vast launcher commands.
- This keeps the next real-data run tied to the branch containing the sequential hydration completeness marker, incomplete-shard guard, source-conditioned gate, and official hydrated audit promotion.
- If the next run should use a different branch or immutable SHA, regenerate the remote plan with `--git-ref`; otherwise use the pinned generated `actual_launcher`.

Vast create DNS retry hardening:

- A live relaunch check found no active Vast instances, then found offer `35654867` as an eligible 32 GiB+ RTX 4090 route.
- The actual create request failed twice before instance creation with `Failed to resolve 'console.vast.ai'`, so no remote official hydration run started and no benchmark artifact was produced.
- `scripts/vast_launch.py` now supports bounded retries for transient Vast CLI DNS/connectivity failures, and `scripts/launch_remote_transport_shift_candidate_vast.sh` enables `LAUNCH_RETRIES=3` with `LAUNCH_RETRY_BACKOFF=10` by default.
- The generated remote hydration plan now includes those launch retry knobs explicitly. The next path is still the same pinned `actual_launcher`; the retry change only makes transient pre-instance Vast failures less likely to waste the run attempt.
- A fixed-path retry check did perform all four create attempts, but each failed on the same DNS resolution error before Vast created an instance; a follow-up instance check returned `[]`.

Fractional Fourier transport refinement:

- A follow-up live relaunch against current offer `9021757` again failed before instance creation: all four Vast create attempts hit `Failed to resolve 'console.vast.ai'`, and no benchmark artifact was produced.
- The local modeling path now moves from integer-only source shifts to train-only fractional Fourier shift refinement.
- `scripts/run_source_conditioned_transport_shift_gate.py` still starts from the train-only coarse/source-conditioned map, but when `--fractional-refine-step` is positive it evaluates fractional periodic shifts around the train-selected shift using a Fourier phase ramp.
- `scripts/plan_transport_official_hydration.py` now emits `--fractional-refine-step 0.5` for official train/val validation, and `scripts/run_official_hydrated_post_validation_test.py` passes the same estimator knob to the gated held-out test command and ledger key.
- This preserves the benchmark boundary: no validation fitting, no test shard in the train/val hydration plan, and no held-out test execution unless the objective audit reaches `literal_test_ready`.

Vast DNS preflight hardening:

- A later relaunch check found current RTX 4090 offers under `$1/hr`, including offer `35680432`, but the official launcher still failed resolving `console.vast.ai` before instance creation.
- Local sequential hydration is not currently feasible because the shared filesystem had only about 471 MiB free, far below the one-raw-file sequential hydration requirement.
- `scripts/vast_launch.py` now runs a DNS preflight for `console.vast.ai` before paid Vast launch/create requests; if the host cannot resolve, it exits before attempting instance creation.
- The preflight is deliberately bypassable with `--skip-launch-preflight`, but the default path avoids repeated create attempts when the local network state already proves the remote run cannot start.

Official execution readiness artifact:

- `scripts/check_official_execution_readiness.py` now records route-specific readiness for the two legitimate paths: pinned remote Vast execution and local sequential hydration.
- The live readiness artifact currently reports both routes blocked: `console.vast.ai` does not resolve for remote launch, `darus.uni-stuttgart.de` does not resolve for local download, and local free disk is about `553787392` bytes against a `9467911994` byte sequential requirement.
- Use `reports/research/sota_loop/official_execution_readiness.json` before another launch attempt so DNS/disk blockers are visible before any paid create request or large download is attempted.
- `scripts/run_official_transport_objective_status.sh` now includes this readiness artifact, so the canonical objective status surfaces those route-specific blockers while still requiring the official hydrated train/val gate before held-out test authorization.

Official Vast launcher readiness gate:

- `scripts/launch_remote_transport_shift_candidate_vast.sh` now runs `scripts/check_official_execution_readiness.py` before actual `scripts/run_remote_official_hydration.sh` launches.
- If `remote_launch_ready=false`, the wrapper exits before invoking `scripts/vast_launch.py launch`; dry runs and non-official remote scripts are unaffected.
- The live checkpoint still blocks both execution routes: `console.vast.ai` does not resolve, `darus.uni-stuttgart.de` does not resolve, and local free disk is about `257413120` bytes versus the `9467911994` byte sequential requirement.
- This is a fail-closed operational guard, not a benchmark substitute. The literal objective still requires `reports/research/sota_loop/official_hydrated_transport_shift_gate.json`, validation pass on official train/val, and exactly one held-out test only after `literal_test_ready`.

Fractional sample-mode refinement:

- The official estimator path already used `--fit-strategy sample_mode --refine-radius 4 --fractional-refine-step 0.5`.
- `scripts/run_source_conditioned_transport_shift_gate.py` now applies fractional periodic refinement inside each train-sample vote, not only as a final group-level refinement after integer sample voting.
- This better matches the shift-canonicalization direction from recent equivariant neural-operator work while remaining benchmark-clean: no validation fitting, no test-shard access during train/val hydration, and the locked source shift map is still train-derived.
- The current execution route is still blocked externally: live readiness reports unresolved `console.vast.ai` and `darus.uni-stuttgart.de`, plus about `3282300928` local free bytes versus the `9467911994` byte sequential requirement.

Sequential hydration preflight alignment:

- Local disk later improved to about `45878403072` free bytes, enough for the one-file sequential requirement but still below the all-files raw requirement.
- `scripts/preflight_transport_hydration.py` now supports `--mode all|sequential` and defaults to sequential mode because the active hydration path downloads/appends one official Advection file at a time.
- `scripts/recommend_transport_hydration_storage.py` uses the same mode, so storage recommendation now distinguishes full raw download storage from sequential hydration storage.
- `scripts/audit_transport_objective_status.py` no longer reports ready evidence statuses as blockers. The remaining live blockers are unresolved `darus.uni-stuttgart.de` for local official data access, unresolved `console.vast.ai` for remote launch, and the missing official hydrated train/val gate artifact.

Official data URL override path:

- A later live probe showed `curl` also failing to resolve `darus.uni-stuttgart.de`, so the local route is blocked on default-host access rather than disk.
- `scripts/download_pdebench_file.py` now accepts manifest-level `url`, `download_url`, or `source_url` fields and a `PDEBENCH_DATAFILE_URL_TEMPLATE` environment override.
- `scripts/plan_transport_official_hydration.py` preserves those optional URL fields into `remote_entries`, allowing the sequential hydrator to use a verified mirror or pre-signed official object URL while retaining the same logical path, size, and checksum checks.

Readiness URL override alignment:

- `scripts/check_official_execution_readiness.py` now derives official data DNS probes from the same URL rules as the downloader: manifest URL fields, then `PDEBENCH_DATAFILE_URL_TEMPLATE`, then the default Darus API URL.
- This means a reachable verified mirror or pre-signed URL can make the local sequential route ready without pretending Darus itself is reachable.
- With the current default manifest, live readiness is still blocked because no alternate URL is configured and `darus.uni-stuttgart.de` plus `console.vast.ai` still fail DNS resolution.

Staged raw sequential hydration path:

- `scripts/hydrate_official_advection_source_sequential.py` now supports `--use-existing-raw`.
- `scripts/run_remote_official_hydration.sh` exposes the same path via `SEQUENTIAL_USE_EXISTING_RAW=1`.
- Use this when official raw HDF5 files have been copied or mounted into the planned `raw_out` paths outside the downloader. The hydrator still appends only from those manifest paths, preserves `source_file_index` provenance, and leaves the held-out test gated behind `literal_test_ready`.

Staged raw readiness detection:

- `scripts/check_official_execution_readiness.py` now inspects planned raw files under `raw_out`.
- If every `remote_entries` file exists at the expected manifest size, local sequential hydration is ready even without Darus DNS.
- This makes the staged-raw path operationally visible in the canonical readiness artifact while preserving the same manifest path and size requirements.

Staged raw checksum guard:

- Staged raw readiness now validates MD5 checksums when the manifest provides checksum metadata.
- `scripts/hydrate_official_advection_source_sequential.py --use-existing-raw` blocks on missing, size-mismatched, or checksum-mismatched files before appending any samples.
- This lets staged raw files replace network access only when they match the official manifest evidence.

Vast instance wrap-up checkpoint:

- A live wrap-up check found no active Vast instances: `vastai show instances --raw` returned `[]`.
- Official execution readiness remains blocked on DNS for both default data access (`darus.uni-stuttgart.de`) and remote execution (`console.vast.ai`).
- Local sequential disk capacity is now sufficient for the active one-file-at-a-time hydration route, but the planned official raw HDF5 files are not staged under `data/pdebench/raw`.
- The literal objective remains blocked until the official hydrated train/val gate exists and passes; held-out test execution remains unauthorized.

Official raw staging instructions:

- `scripts/print_official_raw_staging_instructions.py` now provides the exact staged-raw checklist for the official path: local path, expected byte size, expected MD5, current completion state, and the next sequential hydration command.
- Running it against the live readiness state currently reports `status=needs_staging` with `0/8` official raw Advection files complete.
- This should be the next operator handoff when DNS remains broken: stage the listed files at the planned paths, rerun readiness, then run the printed `SEQUENTIAL_USE_EXISTING_RAW=1` hydration command.

Official raw download handoff:

- The readiness checker now reports `next_action=stage official raw files or restore official data DNS` in the current live state, because local sequential disk is sufficient but Darus and Vast DNS remain blocked.
- The staging instructions now include the exact source URL and resumable `curl` command for each official raw file, in addition to the planned path, size, MD5, and sequential hydration command.
- This makes the offline/staged route concrete without relaxing the official-data evidence requirements.

Dataverse redirect hydration attempt:

- A direct quoted `curl -I` to Darus briefly reached Dataverse and returned a pre-signed S3 redirect, but actual sequential hydration still failed because repeated Darus DNS lookups failed before any download or sample append completed.
- `scripts/download_pdebench_file.py` now supports resolving the Dataverse redirect once before ranged downloads via `PDEBENCH_DOWNLOAD_RESOLVE_REDIRECT=1`, with bounded retries for transient Darus DNS failures.
- `scripts/run_remote_official_hydration.sh` exports redirect resolution defaults for future official runs. The current live run remains blocked after 8 failed redirect probes; no official hydrated gate exists and held-out test execution remains unauthorized.

Resolved official URL plan path:

- `scripts/resolve_official_plan_urls.py` can now produce a derived official hydration plan whose `remote_entries` include pre-signed `source_url` values resolved from Dataverse.
- This is still benchmark-clean because it preserves official logical paths, byte sizes, and MD5 checksums; it only removes repeated dependence on Darus DNS after the redirects have been captured.
- The live resolver attempt failed after 3 Darus DNS failures on the first file, so no resolved-url plan is available yet.

S3 DNS fail-fast hardening:

- A live attempt with a captured pre-signed S3 URL for beta0.1 also failed because `s3.tik.uni-stuttgart.de` did not resolve during ranged download.
- `scripts/download_pdebench_file.py` now treats curl host-resolution failures as `NameResolutionError`, cancels pending ranged futures, and fails fast.
- The official route remains blocked until Darus/S3 DNS is stable enough to resolve all required URLs or the raw files are staged from another environment.

Official hydrated benchmark achieved:

- The local route ultimately succeeded using DNS-over-HTTPS backed `curl --resolve`, Dataverse redirect reuse, HTTP/1.1 ranged downloads, S3 A-record rotation, and `SEQUENTIAL_RESUME=1` for partial official hydration recovery.
- `reports/research/sota_loop/official_advection_sequential_hydration_run.json` is `status=executed`: all eight official Advection train files were appended with 48 samples each and raw staging was cleaned.
- `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` passed validation with `nrmse=0.0028383232393941124` against reference `0.30780652221851373`; the locked train-fitted source shift map is `{0: 1.0, 1: 2.0, 2: 4.0, 3: 7.0, 4: 10.0, 5: 20.5, 6: 41.0, 7: 71.5}`.
- Exactly one held-out test was then run after `literal_test_ready`; `reports/research/sota_loop/official_hydrated_post_validation_test_run.json` records `test_result_count=1` and test `nrmse=0.0017648902922571088`.
- `reports/research/sota_loop/transport_objective_status.json` now reports `status=literal_achieved` with no blockers. Do not rerun the held-out test unless a future plan explicitly changes the ledger policy and benchmark protocol.

Parameter-conditioned transport successor:

- `scripts/run_parameter_conditioned_transport_shift_gate.py` fits a train-only linear rule from parsed official Advection `beta` metadata to periodic shift. It uses `source_file_index` only to join rows to their source beta, not as the learned shift key.
- Fresh source-conditioned validation baseline on official `light-v1`: `nrmse=0.0028383232393941124`, no held-out test rerun.
- Parameter-conditioned validation result: `nrmse=0.001981674036057911`, fitted `shift = 10.236877359639507 * beta - 0.08098891730605368`, guard passed against reference `0.30780652221851373`.
- Exactly one held-out test was run for the locked beta-conditioned estimator and recorded in `reports/research/sota_loop/causal_transport_head/parameter_conditioned_test_ledger.json`: test `nrmse=0.001232006631009314`.
- Status: promote as the new best narrow official Advection transport result and the next cleaner G2.5 step. Do not call this universal SOTA; it is still parameterized Advection transport, not a general learned simulator across PDE families.

Inferred context transport successor:

- `scripts/run_inferred_transport_shift_gate.py` infers each sample's transport shift from early observed context and fits only a train-split linear calibration from inferred context shift to rollout shift.
- It does not use explicit `beta` metadata or `source_file_index` as the learned key. The benchmark tradeoff is that it uses observed context frames, so it is a causal online estimator rather than a zero-context simulator.
- Beta-conditioned baseline on this branch: validation `nrmse=0.001981674036057911`.
- Best train/val-only inferred setting: `context_transitions=8`, `refine_radius=4`, `fractional_refine_step=0.025`, validation `nrmse=0.00029621962142020844`.
- Exactly one held-out test was run for the locked inferred estimator and recorded in `reports/research/sota_loop/inferred_transport_head/inferred_transport_test_ledger.json`: test `nrmse=0.0001883979016384957`.
- Status: promote as the new best narrow official Advection transport result. The next universal-SOTA step is to test whether this context-inferred mechanism transfers beyond Advection or can be integrated into the broader UPS latent operator scorecard.

Inferred transport transfer scorecard:

- `scripts/run_inferred_transport_transfer_scorecard.py` now applies the inferred context transport gate to local train/validation splits only. It passes no held-out test split to task gates and records `held_out_policy=train/val only; no held-out test split is passed to task gates`.
- Live scorecard output at `reports/research/sota_loop/inferred_transfer_scorecard/scorecard.json` reports `status=partial_transfer_validated`, `evaluated_task_count=2`, `skipped_task_count=1`, and `mean_validation_nrmse=0.00303644300924271`.
- Advection transfer validation: `nrmse=0.0002474825485253347`, train `nrmse=0.000021722591109190475`, `test_touched=false`.
- Burgers transfer validation: `nrmse=0.0058254034699600854`, train `nrmse=0.062408372798664555`, `test_touched=false`.
- Darcy is skipped in the live local scorecard because `data/pdebench/darcy2d_train.h5` is absent. The scorecard also explicitly rejects non-`1d` tasks when splits exist, because this mechanism is a 1D transport gate rather than a static 2D operator.
- Status: this is the next credible transfer signal after the official Advection result, not a universal SOTA claim. It supports continuing toward a broader scorecard, but it does not yet demonstrate general PDE-family SOTA or foundation-model behavior.

Universal SOTA status audit:

- `scripts/audit_universal_sota_status.py` now produces `reports/research/sota_loop/universal_sota_status.json` from the light-v1 demo scorecard, official transport objective status, inferred transfer scorecard, and optional `reports/light_experiments_remote/ups_light*/summary.json` candidate summaries.
- Live status is `not_sota_ready`: official transport is achieved and transfer evidence is present, but the broader universal SOTA claim still fails closed.
- The best overall light-v1 row in the local ignored scorecard is `ups_light_observed_shift_estimator_test` with decoded rollout `nrmse=0.20177292896682064`; the audit records it as best overall but excludes diagnostic run fragments such as `gate_hook`, `residual_alpha`, `roll_shift`, `observed_shift`, `transport_gate`, `transport_horizon_gate`, and `transport_residual_gate` from claim eligibility.
- The best current claim-eligible light-v1 row is `ups_light_task_signature_trained_residual` with decoded rollout `nrmse=0.530536668470072`, only `0.06950056206815583` better than persistence `0.5701633411507036`; this fails the required `0.2` improvement gate.
- The next implementation gate must therefore improve the learned general PDE operator/refiner path itself, not just repackage diagnostic transport-sidecar wins.
- The audit also requires medium-or-larger split confirmation, a strong neural baseline comparison, W&B or artifact handles, and exact claim documentation before any SOTA-style claim is allowed.

Global residual gate calibration:

- `scripts/calibrate_residual_gate.py` now supports `--kind global`, a global per-horizon decoded residual schedule, and `--test-ledger-json` / `--allow-repeat-test` held-out test controls.
- Validation selected a global horizon schedule: horizon 1 uses residual alpha `0.4`; horizons 2-16 use `0.1`.
- Validation decoded rollout improved to `nrmse=0.3526528527726788`, a `0.011598261282886628` relative gain over clean reference `0.3567910081081011`, so the guarded held-out test was eligible.
- Exactly one held-out test was run and recorded in `reports/research/sota_loop/global_residual_gate/test_ledger.json` with measurement key `6cbf489a964e67ec50bc2f2ea44355cdeed0b90976e5d75d2f09e47819740762`.
- Held-out test decoded rollout was `nrmse=0.5383591367287355`, with per-task rollout `advection1d=0.7686066115389052`, `burgers1d=0.13556008026444324`, and `darcy2d=0.21900255465784269`.
- A repeat of the same guarded command failed before measurement with `held-out test measurement already recorded for this residual gate`, confirming the no-repeat guard.
- Status: do not promote this gate. It improved validation but regressed held-out test versus the current claim-eligible light row `0.530536668470072`, so it is evidence of validation overfit in the global residual schedule rather than universal SOTA progress.

Train-confirmed residual gate calibration:

- `scripts/calibrate_residual_gate.py` now supports `--selection-split`, so candidate gates can be selected on a training split and then independently confirmed on validation before a held-out test is authorized.
- `evaluation.skip_missing_tasks=true` is now honored by latent loader construction, decoded evaluation, and decoded grid-spec inference. This lets local train-selection probes use available train shards while preserving the full checkpoint task vocabulary; in the current local data, `darcy2d_train.h5` is absent and the train selector skips Darcy explicitly.
- Train selection on available `burgers1d` and `advection1d` train shards selected constant residual alpha `0.0` with train decoded rollout `nrmse=0.10786857080851395`.
- The frozen train-selected gate validated on the full local validation split at decoded rollout `nrmse=0.3685752310100123`, failing the guard against reference `0.3567910081081011` with relative improvement `-0.03302836291866644`.
- No held-out test was run and no test ledger was created for `reports/research/sota_loop/train_confirmed_residual_gate/`.
- Status: residual alpha schedules are now negative under stricter train-selection/validation-confirmation policy. The next universal-SOTA work should move to learned operator/refiner training or data/model capacity, not further scalar residual schedule search.

Local train substrate readiness audit:

- `scripts/check_demo_readiness.py --check-local-data` now audits manifest-required local source HDF5 shards under `source_root` before full-task train selection or learned-operator work proceeds.
- Initial live local audit output at `reports/research/sota_loop/local_light_data_readiness.json` reported `local_data.ok=false`, `expected_count=9`, `present_count=8`, `missing_count=1`, and missing `data/pdebench/darcy2d_train.h5` with `required_samples=128`.
- Local B2 hydration was not executable in this worktree: `.env` was absent and `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET`, `B2_BUCKET_NAME`, `B2_PREFIX`, `B2_S3_ENDPOINT`, and `B2_S3_REGION` were unset.
- The missing local Darcy train shard was hydrated from official PDEBench file `2D/DarcyFlow/2D_DarcyFlow_beta0.01_Train.hdf5` (`file_id=133217`, MD5 `d05c287d4c0b7d3178b0097084238251`). The local `data/pdebench/darcy2d_train.h5` uses raw `nu` rows `160:288`, avoiding the existing local Darcy test rows `0:32` and validation rows `128:160`.
- Post-hydration strict readiness output at `reports/research/sota_loop/local_light_data_readiness_after_darcy_train.json` reports `ready=true`, `local_data.ok=true`, `expected_count=9`, `present_count=9`, `missing_count=0`, and `short_count=0`.
- Full-task train-confirmed residual calibration without `evaluation.skip_missing_tasks=true` is recorded at `reports/research/sota_loop/train_confirmed_residual_gate_full_local/calibration.json`. Selection again chose constant alpha `0.0`; full train decoded rollout was `nrmse=0.11533330559043692`, validation confirmation stayed `nrmse=0.3685752310100123`, and the guard failed against reference `0.3567910081081011` with relative improvement `-0.03302836291866644`.
- No held-out test was run for the full local train-confirmed residual gate because the selected validation gate failed the test guard.
- Status: the local data-substrate blocker is removed, but scalar residual schedules remain negative under full-task train-selection/validation-confirmation policy. The next universal-SOTA work should move to learned operator/refiner training or data/model capacity, not further scalar residual schedule search.

Learned capacity validation probe:

- Local runner-native validation baselines are recorded under `reports/research/sota_loop/learned_capacity_gate/` using the current `ups_light_task_signature_trained_residual` checkpoint and full local validation split with `data.max_samples=32`, `decoded_rollout_steps=16`, and checkpoint-compatible conditioning `{"task_id":3,"equation_signature":15}`.
- Raw UPS rollout with implicit residual alpha `1.0` produced validation decoded rollout `nrmse=0.7970612206180094`, which confirms the learned operator alone is not competitive with persistence on this split.
- Explicit persistence alpha `0.0` reproduced validation decoded rollout `nrmse=0.3685752310100123`; explicit trained-residual alpha `0.25` produced `nrmse=0.374001436897959`.
- A small local joint codec/operator fine-tune from the trained-residual checkpoint, using `data.max_samples=32`, `stages.joint_codec_operator.epochs=2`, `rollout_steps=4`, `lambda_rollout=1.0`, `lambda_persistence_residual=0.5`, and `lambda_persistence_residual_spectral=0.05`, improved matched alpha `0.25` validation to `nrmse=0.363031039578185`.
- Validation-only alpha sweep on that fine-tuned checkpoint found the best result at transport-family alpha `0.18`: decoded rollout `nrmse=0.35376568397505964`, with task metrics `advection1d=0.4912554112347586`, `burgers1d=0.14738121412908425`, and `darcy2d=0.188979512124482`.
- A larger local probe with `data.max_samples=64` and `stages.joint_codec_operator.epochs=4` regressed to decoded rollout `nrmse=0.35417606099874893` at the same transport-family alpha `0.18`.
- Status: this is the first learned-capacity validation improvement over the clean transport reference `0.3567910081081011`, but the relative validation improvement is only about `0.00848`, below the strict `0.01` held-out-test authorization guard. No held-out test was run. The next run should scale the successful two-epoch joint fine-tune recipe remotely or add a true learned transport/refiner head; do not spend held-out test budget on the current local checkpoint.

Learned capacity remote execution wrap-up:

- The reproducible remote queue variant for `task_signature_rollout4_residual_ft2` is merged on `main`; it requires `--checkpoint-source`, forces `STAGES=joint_codec_operator`, and uses the validation-selected transport-family residual alpha `0.18`.
- Because the checkpoint source under `reports/light_experiments_remote/ups_light_task_signature_trained_residual` is ignored by git, a tarball copy was staged to B2 at `remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz` before Vast execution attempts.
- Vast contracts `37956109` and `37956442` both remained in `loading` with no Docker container or logs and were destroyed.
- Vast contract `37956640` used the on-start file route but stopped before container creation; logs reported `No such container: C.37956640`, and the contract was destroyed.
- Vast contract `37957089` used the corrected documented shape, `--ssh --direct --onstart --cancel-unavail`, but stopped before benchmark execution with status `docker_build() error writing dockerfile`; it was destroyed.
- A live cleanup check after the final destroy returned no active Vast instances: `vastai show instances --raw` returned `[]`.
- B2 does not contain the intended validation artifact `remote-runs/light/ups_capacity_light_task_signature_rollout4_residual_ft2_remote_val_20260526T1932Z.tar.gz`; no remote validation metric was produced.
- The ignored local Vast startup script now requires secrets via environment variables instead of hardcoding them.
- No held-out test was run. The learned-capacity path remains validation-only, and the strict `1%` guard is still not cleared.
- Status: current repo and CI are ready, but this specific Vast execution path is blocked at provider/container bootstrap. The next remote attempt should avoid Vast's Dockerfile-injection startup path, either by using a prebuilt image/template that already contains the runner bootstrap or by moving the exact queue to another GPU environment with ordinary Docker/SSH control.

Compact Vast bootstrap route:

- `scripts/vast_launch.py` now supports `--bootstrap-mode tracked-script`, which writes a small onstart shim that downloads `scripts/vast_remote_bootstrap.sh` from the requested Git ref and decodes the long remote script arguments from `UPS_SCRIPT_ARGS_B64`.
- `scripts/vast_remote_bootstrap.sh` owns the full remote setup: clone or GitHub zip fallback, rclone install, dependency install profile, optional prefetch, remote script execution, and auto-shutdown trapping.
- `scripts/launch_remote_smoke_vast.sh` and `scripts/launch_remote_transport_shift_candidate_vast.sh` expose this as `BOOTSTRAP_MODE=tracked-script`.
- A learned-capacity dry run with `--bootstrap-mode tracked-script`, `--git-ref codex/vast-tracked-bootstrap`, and the full `task_signature_rollout4_residual_ft2` validation command produced a `3223` byte `.vast/onstart.sh` with no literal B2 or W&B secrets.
- Verification:
  - `python -m pytest tests/unit/test_vast_launch.py tests/unit/test_launch_remote_smoke_vast.py tests/unit/test_launch_remote_transport_shift_candidate_vast.py -q` -> `17 passed`
  - `python -m black --check scripts/vast_launch.py tests/unit/test_vast_launch.py tests/unit/test_launch_remote_smoke_vast.py`
  - `bash -n scripts/vast_remote_bootstrap.sh scripts/launch_remote_smoke_vast.sh scripts/launch_remote_transport_shift_candidate_vast.sh`
- Status: this creates a smaller falsifiable Vast retry path, but it must be pushed to a branch before a live worker can fetch the tracked bootstrap script. No held-out test was run.

Compact Vast bootstrap live smoke:

- Branch `codex/vast-tracked-bootstrap` was pushed so Vast could fetch `scripts/vast_remote_bootstrap.sh` from GitHub raw content.
- Vast contract `37963841` launched with `--args-mode`, `--bootstrap-mode tracked-script`, `--git-ref codex/vast-tracked-bootstrap`, and a dry smoke dispatch: `DRY_RUN=1 PREP_SHARDS=0 RUN_EXPERIMENTS=0 CHECK_B2=0 PIPELINE_ROOT=reports/demo/remote_bootstrap_smoke`.
- The container reached user code, downloaded the branch by GitHub zip fallback because `git` was unavailable, installed the package with the `smoke` install profile, generated the smoke queue, printed `Remote smoke pipeline complete`, and exited with `REMOTE_BOOTSTRAP_EXIT_STATUS=0`.
- The contract was destroyed after log verification, and a cleanup check returned no active Vast instances: `vastai show instances --raw` returned `[]`.
- Status: the previous provider/container bootstrap blocker is cleared for the compact args-mode route. The next live step can use this same branch/ref and tracked bootstrap shape for the validation-only learned-capacity queue; held-out test execution remains unauthorized until the validation guard clears.
