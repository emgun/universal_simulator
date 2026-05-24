# Working Demo and SOTA Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working UPS demo with credible held-out results, then push the strongest track toward a narrow, defensible SOTA-style result.

**Architecture:** Treat the demo as a benchmark product: frozen data splits, reproducible runs, baseline comparisons, visual rollouts, and explicit promotion gates. Use the existing `scripts/run_light_experiment.py` and B2/Vast remote loop as the harness, then add only the highest-leverage model ideas that current physics-foundation-model work supports.

**Tech Stack:** Python, PyTorch, HDF5/PDEBench, B2 object storage, Vast.ai, W&B optional, existing UPS latent operator + any-point decoder + promotion-rule stack.

---

## Executive Direction

The fastest credible path is not "train a universal physics foundation model from scratch." It is:

1. Build a polished, reproducible demo on real held-out PDEBench shards.
2. Beat strong matched baselines on that demo under equal data and compute.
3. Scale only the winning track.
4. Attempt a narrow public-style SOTA claim only after the evaluation contract is frozen.
5. Keep a separate "foundation" track that measures transfer and sample efficiency, not just per-task error.

Current repo state supports this path:

- Branch: `codex/autowork-semigroup-foundation`.
- Current light loop: `scripts/run_light_experiment.py`.
- Current remote wrapper: `scripts/run_remote_light_promotion.sh`.
- Current best local heterogeneous config: `configs/train_multitask_heterogeneous_light_best.yaml`.
- Current real-data smoke: `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`.
- Known blocker for credible benchmark: small held-out B2 shards for `burgers1d`, `advection1d`, and `darcy2d` are not yet published.

## External Research Signals To Exploit

Use external work as design pressure, not as a reason to rewrite everything immediately.

- **Poseidon:** multiscale operator transformers, time-conditioned normalization, semigroup training, and data/model scaling are validated foundation-model ingredients.
- **DPOT:** autoregressive denoising pretraining is a credible high-upside addition once the real-data loop is stable.
- **PROSE-FD / PDEformer:** symbolic or PDE-graph conditioning is useful, but only if it improves held-out transfer. UPS's current `task_id + equation_signature` signal is the right small version to test first.
- **Walrus / cross-domain continuum models:** adaptive tokenization, patch jitter, tensor-law-aware augmentation, and 2D/3D mixed-domain training are longer-horizon foundation features.
- **RealPDEBench:** real-world measured data exposes sim-to-real gaps, so do not market simulated PDEBench success as a general physics foundation model without a separate real-data gate.

Primary references:

- PDEBench: https://github.com/pdebench/PDEBench
- Poseidon: https://arxiv.org/abs/2405.19101
- DPOT: https://arxiv.org/abs/2403.03542
- PROSE-FD: https://arxiv.org/abs/2409.09811
- PDEformer-2: https://arxiv.org/abs/2507.15409
- Walrus: https://huggingface.co/polymathic-ai/walrus
- RealPDEBench: https://arxiv.org/abs/2601.01829

## Demo Definition

The demo is successful when one command produces an artifact bundle that a reviewer can inspect without trusting our claims.

Required artifacts:

- `reports/demo/<run_id>/index.html`
- `reports/demo/<run_id>/summary.json`
- `reports/demo/<run_id>/metrics.tsv`
- `reports/demo/<run_id>/rollouts/*.png`
- `reports/demo/<run_id>/rollouts/*.gif` or `.mp4`
- `reports/demo/<run_id>/resolved_train.yaml`
- `reports/demo/<run_id>/resolved_eval.yaml`
- `reports/demo/<run_id>/cost.json`

Required views:

- Per-task prediction vs. ground truth rollouts for `burgers1d`, `advection1d`, `darcy2d`.
- Error-vs-horizon plot.
- Baseline comparison table.
- Speed and cost table.
- Failure cases, not only cherry-picked wins.

Required metrics:

- `decoded_rollout_nrmse`
- `decoded_step1_nrmse`
- `decoded_h4_nrmse` and `decoded_h16_nrmse` when available
- `mse`, `mae`, `rmse`
- per-task decoded rollout metrics
- per-family decoded rollout metrics
- tokens/sec or samples/sec
- GPU hours and estimated dollar cost

Minimum "good result" gate:

- UPS beats a matched baseline by at least 20% on held-out `decoded_rollout_nrmse` for at least two of three demo tasks.
- UPS does not regress more than 5% on the remaining task.
- UPS is faster than the baseline at inference, or the report explicitly explains the speed/accuracy tradeoff.

Minimum "SOTA-style result" gate:

- Same public split, preprocessing, rollout horizon, and metric definition as the comparison target.
- At least three seeds or a stability argument if compute makes three seeds impossible.
- Public or reproduced baseline table included.
- Exact configs, commit SHA, and data manifest included.

## Track Selection

Use three tracks with different credibility horizons.

### Track A: Fast Demo

Goal: get a high-quality demo in days.

Scope:

- Small real held-out shards.
- Current UPS architecture.
- Current semantic conditioning.
- Matched baselines.
- Visual report.

Primary success metric:

- held-out `decoded_rollout_nrmse`.

Promotion gate:

- better than baseline on held-out shards and visually plausible rollouts.

### Track B: Narrow SOTA Attempt

Goal: attempt a credible result on one or two PDEBench tasks.

Scope:

- Start with `burgers1d` and `advection1d`.
- Add `darcy2d` only after the grid/shape path is stable.
- Use full or medium real train/test splits.
- Compare against FNO/U-Net/DiT/CNO-style baselines where feasible.

Primary success metric:

- public-compatible nRMSE or relative RMSE at matching horizon.

Promotion gate:

- beats reproduced baselines under the same data/compute contract.

### Track C: Foundation Model Direction

Goal: show that UPS is more than a single-task solver.

Scope:

- Multitask pretraining.
- Held-out task transfer.
- Low-sample finetuning.
- Resolution transfer.
- Later: RealPDEBench or The Well schema.

Primary success metric:

- sample efficiency and transfer delta versus task-specific training.

Promotion gate:

- pretrained UPS reaches the same held-out error with at least 2x fewer target-task samples or steps.

## Phase 0: Freeze The Benchmark Contract

Done condition: no experiment result is discussed without a data manifest, config, commit, and metric contract.

Files:

- Create: `docs/demo_benchmark_contract.md`
- Modify: `docs/light_experiment_loop.md`
- Modify: `worklog.md`

Steps:

- [ ] Define the canonical demo splits: tasks, train count, val count, test count, source B2 keys, output B2 keys.
- [ ] Define the canonical metrics and which metric controls promotion.
- [ ] Define baseline families: `identity`, `persistence`, existing latent baselines, FNO/U-Net/DiT if available in repo.
- [ ] Define leakage rules: no test split tuning, no train-shard evaluation used for benchmark claims.
- [ ] Define cost tiers:
  - smoke: `data.max_samples<=8`, expected <$0.10 once data is hydrated.
  - light held-out: 32-128 train samples, expected <$1-$3.
  - medium: 512-2048 train samples, expected <$5-$20.
  - scale: full selected split, explicit approval required.
- [ ] Add exact result labels:
  - `smoke`: plumbing only.
  - `light`: cheap held-out evidence.
  - `medium`: candidate promotion.
  - `benchmark`: claim candidate.

Validation:

```bash
rg -n "decoded_rollout_nrmse|data.max_samples|leakage|cost" docs/demo_benchmark_contract.md docs/light_experiment_loop.md worklog.md
```

## Phase 1: Publish Small Held-Out B2 Shards

Done condition: B2 has versioned light shards for every demo task and split.

Files:

- Modify: `scripts/make_light_hdf5_shards.py`
- Create: `scripts/publish_light_hdf5_shards_b2.sh`
- Create: `docs/demo_data_manifest.yaml`
- Test: `tests/unit/test_make_light_hdf5_shards.py`

Tasks:

- [ ] Extend shard creation so each task can use its native source split when available.
- [ ] For `darcy2d`, derive `val` from the train source until a true `val` split exists, and label it clearly as `derived_val`.
- [ ] Add a manifest writer that records source files, offsets, sample counts, output hashes, and B2 destination keys.
- [ ] Add an upload wrapper that uses the existing `.env` without printing secrets.
- [ ] Add a dry-run mode that prints planned B2 keys and sizes.
- [ ] Publish `light-v1`:
  - `burgers1d`: train 128, val 32, test 32.
  - `advection1d`: train 128, val 32, test 32.
  - `darcy2d`: train 128, val 32 derived from train, test 32.
- [ ] Publish `smoke-v1`:
  - all tasks train 8, val 4, test 4.

Validation:

```bash
bash -n scripts/publish_light_hdf5_shards_b2.sh
python -m py_compile scripts/make_light_hdf5_shards.py
pytest tests/unit/test_make_light_hdf5_shards.py -q
```

Promotion gate:

- `DRY_RUN=1 bash scripts/run_remote_light_promotion.sh` resolves only the `light-v1` keys.
- No full 141 GiB hydration path is triggered.

## Phase 2: Build The Demo Scorecard

Done condition: every run produces a machine-readable and human-readable comparison.

Files:

- Create: `src/ups/eval/demo_scorecard.py`
- Create: `scripts/build_demo_report.py`
- Create: `scripts/collect_light_results.py`
- Create: `tests/unit/test_demo_scorecard.py`
- Modify: `scripts/run_light_experiment.py`

Tasks:

- [ ] Add a scorecard schema with:
  - run name
  - commit SHA
  - data manifest path
  - config path
  - train/eval sample caps
  - metrics
  - cost
  - promotion result
- [ ] Add aggregation from multiple `summary.json` files into `metrics.tsv`.
- [ ] Add promotion-rule support for:
  - max per-task decoded rollout error
  - mean decoded rollout error
  - worst-family decoded rollout error
  - minimum speedup
- [ ] Add report generation:
  - metrics table
  - per-task plots
  - rollout images
  - config and commit metadata
- [ ] Add a no-plot fallback so CI can validate report generation without GPU artifacts.

Validation:

```bash
python -m py_compile src/ups/eval/demo_scorecard.py scripts/build_demo_report.py scripts/collect_light_results.py
pytest tests/unit/test_demo_scorecard.py tests/unit/test_light_experiment_runner.py -q
```

Promotion gate:

- A synthetic-bootstrap run produces a complete `reports/demo/<run_id>/index.html`.

## Phase 3: Establish Matched Baselines

Done condition: UPS is compared against credible baselines before any claim.

Files:

- Modify: `scripts/train_baselines.py`
- Modify: `scripts/benchmark.py`
- Create: `configs/baselines/demo_fno.yaml`
- Create: `configs/baselines/demo_unet.yaml`
- Create: `configs/baselines/demo_persistence.yaml`
- Create: `tests/unit/test_demo_baselines.py`

Tasks:

- [ ] Verify which baseline implementations are actually functional on current HDF5 data.
- [ ] Add a persistence baseline that predicts the last observed state.
- [ ] Add a simple identity baseline for sanity checking.
- [ ] Add FNO/U-Net config presets if the existing baseline code supports them.
- [ ] If FNO/U-Net is not fully wired, do not block the demo; report `persistence` and `identity` immediately and put FNO/U-Net behind a separate task.
- [ ] Make all baselines write the same summary schema as UPS.

Validation:

```bash
python -m py_compile scripts/train_baselines.py scripts/benchmark.py
pytest tests/unit/test_demo_baselines.py tests/unit/test_pdebench_runner_eval.py -q
```

Promotion gate:

- At least one nontrivial baseline appears in the demo report.
- UPS beats persistence on held-out shards before moving to medium runs.

## Phase 4: Run The Fast Experiment Matrix

Done condition: the next model changes are driven by held-out real-data evidence.

Fixed harness:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
TASKS=burgers1d,advection1d,darcy2d \
TRAIN_CONFIG=configs/train_multitask_heterogeneous_light_best.yaml \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16" \
bash scripts/run_remote_light_promotion.sh
```

Initial variants:

- `current_best`: existing `configs/train_multitask_heterogeneous_light_best.yaml`.
- `no_conditioning`: remove explicit conditioning sources.
- `flat_full`: resolution, spatial dims, task id, family, traits, equation signature.
- `task_signature_only`: task id + equation signature.
- `semigroup0`: `training.lambda_semigroup=0.0`.
- `semigroup10`: `training.lambda_semigroup=0.1`.
- `joint16`: `stages.joint_codec_operator.epochs=16`.
- `joint48`: `stages.joint_codec_operator.epochs=48`.
- `rollout4`: `stages.joint_codec_operator.rollout_steps=4`.
- `time_stride1`: `training.time_stride=1`.

Experiment loop:

- [ ] Run `smoke-v1` for all variants first.
- [ ] Promote only variants that complete and produce valid summaries.
- [ ] Run `light-v1` for promoted variants.
- [ ] Aggregate results with `scripts/collect_light_results.py`.
- [ ] Pick at most two winners and one control.
- [ ] Record keep/discard decisions in `worklog.md`.

Promotion gate:

- Winner improves held-out mean `decoded_rollout_nrmse` by at least 10% versus `current_best` or has a clear per-task tradeoff worth scaling.

Stop condition:

- If all variants fail to beat persistence or current best, stop model search and inspect data/evaluation before adding architecture.

## Phase 5: Add Only High-Leverage Model Ideas

Done condition: each new idea has an ablation and is killed quickly if it does not help.

### Idea 1: Time-Conditioned Normalization

Rationale: Poseidon highlights time-conditioned layer norms and continuous-time evaluation. UPS already has conditioning infrastructure, so this should be a small, testable extension.

Files:

- Modify: `src/ups/core/conditioning.py`
- Modify: `src/ups/models/latent_operator.py`
- Create: `tests/unit/test_time_conditioning.py`

Experiment:

- Add `dt` or rollout horizon as an explicit conditioning source.
- Compare with and without time conditioning on `light-v1`.

Gate:

- Keep only if held-out h4/h16 improves without step1 regression.

### Idea 2: Autoregressive Denoising Pretraining

Rationale: DPOT's main signal is denoising pretraining for stable autoregressive PDE rollouts.

Files:

- Create: `src/ups/training/denoising_pretrain.py`
- Modify: `scripts/train.py`
- Create: `configs/train_multitask_denoising_light.yaml`
- Create: `tests/unit/test_denoising_pretrain.py`

Experiment:

- Corrupt latent sequence tokens with small Gaussian noise or masked spans.
- Train operator to predict the clean next latent state.
- Fine-tune with existing joint codec/operator stage.

Gate:

- Keep only if it improves held-out rollout stability at horizon 16 or improves low-sample training.

### Idea 3: Patch Jitter / Resolution Augmentation

Rationale: Walrus reports stability benefits from patch jitter and adaptive tokenization. UPS can test a small version by jittering patch starts or coordinate features.

Files:

- Modify: `src/ups/io/enc_grid.py`
- Modify: `src/ups/data/latent_pairs.py`
- Create: `tests/unit/test_patch_jitter.py`

Experiment:

- Enable only during training.
- Keep deterministic eval.

Gate:

- Keep only if it improves held-out test, not just synthetic bootstrap.

### Idea 4: Symbolic Conditioning Simplification

Rationale: PROSE-FD and PDEformer support symbolic inputs, but UPS's cheap signal favors narrow explicit signatures over broad metadata bundles.

Files:

- Modify: `src/ups/data/pdebench.py`
- Modify: `src/ups/data/latent_pairs.py`
- Create: `configs/train_multitask_symbolic_light.yaml`
- Create: `tests/unit/test_symbolic_conditioning.py`

Experiment:

- Use a minimal symbolic vector: task id, family, equation signature, boundary type if available.
- Avoid node-set complexity until it wins a held-out gate.

Gate:

- Keep only if it improves transfer or per-family worst-case error.

## Phase 6: Medium Runs

Done condition: one candidate is strong enough for the polished demo and one candidate is selected for the narrow SOTA attempt.

Medium settings:

- train samples: 512 to 2048 per task.
- eval samples: 128 to 512 per task.
- rollout horizon: 16 or matching public benchmark horizon.
- max runtime: 4 hours per run unless explicitly promoted.

Candidate matrix:

- best current UPS.
- best Phase 4/5 variant.
- best matched baseline.
- one ablation control.

Validation:

- Every medium run must produce:
  - `summary.json`
  - `metrics.tsv`
  - resolved configs
  - checkpoint list
  - W&B run link or local log
  - cost file

Promotion gate:

- Promote to demo if it beats baseline and has plausible visual rollouts.
- Promote to SOTA attempt only if it scales monotonically from `light-v1` to medium.

## Phase 7: Narrow SOTA Attempt

Done condition: a result can be described honestly as either "matched-baseline SOTA-style" or "public-compatible candidate."

Preferred narrow targets:

1. `burgers1d` full train/test.
2. `advection1d` full train/test.
3. `burgers1d + advection1d` multitask sample-efficiency benchmark.

Avoid initially:

- Full 3-task 141 GiB training unless medium runs are strong.
- RealPDEBench until the simulated demo and metrics contract are stable.
- 1B-parameter foundation-model scaling.

Required comparisons:

- Persistence baseline.
- Existing repo baseline.
- FNO/U-Net/DiT if wired and reproducible.
- Public numbers only if metric and split match exactly.

Claim tiers:

- Tier 1: "UPS demo beats matched in-repo baselines on held-out PDEBench shards."
- Tier 2: "UPS beats reproduced baselines on full public-compatible split."
- Tier 3: "UPS is SOTA on target public benchmark."

Do not use Tier 3 wording unless the public-compatible contract is proven.

## Phase 8: Demo Packaging

Done condition: the demo is runnable, inspectable, and persuasive.

Files:

- Create: `scripts/run_demo.py`
- Create: `docs/demo_runbook.md`
- Create: `reports/demo/README.md`
- Modify: `README.md`

Demo command:

```bash
python scripts/run_demo.py \
  --checkpoint <checkpoint> \
  --data-root data/pdebench_demo \
  --output reports/demo/latest \
  --tasks burgers1d advection1d darcy2d \
  --split test \
  --max-samples 8
```

Demo content:

- 60-second visual story:
  - what input looks like
  - what UPS predicts
  - how error grows over rollout horizon
  - how it compares to baseline
  - what it costs to run
- Static HTML report.
- Optional notebook only after CLI works.

Validation:

```bash
python -m py_compile scripts/run_demo.py scripts/build_demo_report.py
python scripts/run_demo.py --help
```

## Phase 9: Foundation Roadmap After Demo

Do this only after the demo works.

Next foundation milestones:

- Resolution transfer: train at one resolution, evaluate at another.
- Task transfer: hold out `darcy2d`, pretrain on 1D tasks, finetune with few Darcy samples.
- Domain transfer: add `navier_stokes2d`.
- Dataset transfer: map one The Well or RealPDEBench dataset into UPS schema.
- Pretrained initialization: compare from-scratch vs pretrained UPS at 1%, 5%, 10%, 100% target data.
- Inference-time correction: revisit TTC/reward-model-driven rollout only after base rollouts are strong.

Foundation success metric:

- Same target error with at least 2x fewer samples or training steps.
- Stable rollout on unseen physics without catastrophic per-task failure.

## Operating Loop

Use this loop every day during the push.

1. Freeze the harness for the day.
2. Pick three hypotheses maximum.
3. Run smoke checks.
4. Run cheap held-out checks.
5. Aggregate results.
6. Promote, revise, or kill each hypothesis.
7. Update `worklog.md`.
8. Commit code separately from result documentation.
9. Destroy paid instances and verify zero active instances.

Result decision rules:

- `keep`: improves held-out metric or fixes a clear failure without new regression.
- `revise`: promising but has one explainable failure.
- `discard`: improves synthetic only, regresses held-out, or increases cost without accuracy gain.
- `escalate`: requires full-data spend, external baseline integration, or changed claim scope.

## Cost Controls

Hard rules:

- Never launch default full-data hydration without `ALLOW_FULL_DATA=1`.
- Always set `data.max_samples` for light experiments.
- Always copy `summary.json` before destroying the instance.
- Always destroy Vast instances after result capture.
- Always record estimated GPU dollars in the run artifact.

Preferred spend ladder:

- Local synthetic: free sanity only.
- Remote smoke: <$0.10 after hydration.
- Light held-out: <$3 per full matrix cell target.
- Medium: <$20 per candidate.
- Full benchmark: explicit approval only.

## Immediate Next 10 Tasks

1. Write `docs/demo_benchmark_contract.md`.
2. Add manifest output to `scripts/make_light_hdf5_shards.py`.
3. Add `scripts/publish_light_hdf5_shards_b2.sh`.
4. Publish `smoke-v1` B2 shards.
5. Publish `light-v1` B2 shards.
6. Add `src/ups/eval/demo_scorecard.py`.
7. Add `scripts/collect_light_results.py`.
8. Add `scripts/build_demo_report.py`.
9. Run `current_best`, `no_conditioning`, and `semigroup0` on `smoke-v1`.
10. Run promoted variants on `light-v1` and select the first demo candidate.

## Expected Timeline

Fast path:

- Day 1: benchmark contract and shard publishing.
- Day 2: scorecard/report skeleton and baseline sanity.
- Day 3: smoke and light variant matrix.
- Day 4: medium run for top candidate and demo report.
- Day 5: polish demo, write claims, decide whether SOTA attempt is justified.

More realistic path:

- Week 1: working demo with credible held-out results.
- Week 2: medium-scale improved model and stronger baseline comparisons.
- Week 3+: narrow public-compatible SOTA attempt.

## Known Risks

- The current HDF5 loader materializes tensors, so full-data runs are memory risky until streaming is added.
- Current synthetic gains may not transfer to real held-out splits.
- Baseline code may be weaker than public baselines; claims must distinguish matched in-repo baselines from public SOTA.
- `darcy2d` lacks a B2 `val` split, so derived validation must be labeled.
- A polished demo can be achieved quickly; a general physics foundation model cannot be proven quickly.

## Recommended First Execution Slice

Start with Phase 0 and Phase 1. That creates the missing substrate for every later step. Do not add model architecture until `light-v1` held-out shards and a scorecard exist.

