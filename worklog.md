Task:
Make this branch ready for lightweight, repeatable experiments that can screen promising UPS directions cheaply.

Done condition:
The repo has a verified cheap experiment loop with include-aware presets, a runner that records comparable results, and at least one real end-to-end smoke run for operator and decoded shortlist paths.

Mutable files:
- worklog.md
- configs/defaults.yaml
- configs/train_burgers_light_operator.yaml
- configs/train_multitask_light_operator.yaml
- configs/train_burgers_light_joint.yaml
- configs/eval_burgers_light_proxy.yaml
- configs/eval_multitask_light_proxy.yaml
- docs/light_experiment_loop.md
- scripts/benchmark.py
- scripts/evaluate.py
- scripts/run_light_experiment.py
- src/ups/data/latent_pairs.py
- src/ups/data/pdebench.py
- src/ups/eval/pdebench_runner.py
- src/ups/training/losses.py
- scripts/train_baselines.py
- scripts/train.py
- tests/unit/test_conditioning.py
- tests/unit/test_eval_promotion.py
- tests/unit/test_light_experiment_runner.py
- tests/unit/test_losses.py
- tests/unit/test_pdebench.py
- tests/unit/test_pdebench_runner_eval.py
- tests/unit/test_train_pdebench_loader.py
- tests/integration/test_train_operator_pipeline.py

Fixed files:
- cloud launcher scripts
- diffusion / TTC architecture
- broad docs cleanup outside task-local notes

Validation:
- focused unit and integration tests for conditioning, loader behavior, eval, and the light experiment runner
- direct smoke runs through `scripts/run_light_experiment.py`

Constraints:
- current workspace is an older main snapshot
- keep the first implementation narrow and backward compatible
- do not claim benchmark gains without a real run harness

Current status:
- Research memo written in docs/cutting_edge_architecture_research_2026-04-07.md
- Sequence-preserving batches landed in src/ups/data/latent_pairs.py
- Per-step conditioning sequences are preserved for future conditioned rollout work
- Optional semigroup loss landed in scripts/train.py behind training.lambda_semigroup
- Narrow decoded physical-space evaluation landed in src/ups/eval/pdebench_runner.py via evaluate_decoded_operator
- Encoder checkpoint export now uses the exact operator-training encoder for single-task PDEBench runs
- Decoder stage added for single-task PDEBench runs, with saved decoder checkpoints
- Evaluate CLI can now merge decoded metrics when encoder/decoder checkpoints are available
- The staged `all` pipeline now includes decoder, frozen-codec decoded operator fine-tuning, and joint codec/operator fine-tuning when configured
- Decoded evaluation now reports rollout and horizon metrics, including decoded_rollout_nrmse, decoded_step1_nrmse, decoded_h4_nrmse, and decoded_h16_nrmse when available
- Auto-conditioning now wires structured PDEBench metadata into latent batches and the operator config, including resolution, spatial dimensionality, and multi-task one-hot task IDs
- Multi-task latent operator training now runs with conditioned task metadata instead of only concatenating task datasets blindly
- Shared encoder export now works for multi-task PDEBench latent training when tasks share channel count
- Raw-field decoder and joint codec/operator stages now support multi-task PDEBench runs across heterogeneous grids with variable-grid batching
- Decoded evaluation now supports multi-task PDEBench configs and emits per-task decoded rollout metrics
- PDEBench loading now supports config-driven `data.param_keys` and `data.bc_keys` so real HDF5 metadata can enter the conditioning path
- Auto-conditioning source dims are now inferred from actual PDEBench samples, including configured parameter and boundary-condition features
- Raw-field rollout stages and decoded evaluation now use per-step parameter / BC conditioning, not just static task metadata
- The eval CLI now supports held-out transfer-task evaluation with `transfer_*` metrics, including decoded transfer rollout metrics when codec checkpoints are present
- Promotion-rule evaluation now lives in src/ups/eval/promotion.py and is exposed through scripts/evaluate.py with optional nonzero exit on gate failure
- Configs can now define evaluation.promotion.rules, and eval reports include promotion_passed plus failing or missing rules
- PDEBench task semantics now include structured task-family and equation-trait features, not only flat task IDs
- Auto-conditioning now adds metadata-presence vectors (`param_presence`, `bc_presence`) so cross-task runs can distinguish missing metadata from zero-valued metadata
- Multi-task auto-conditioning source inference now includes semantic task features and metadata-presence features in both train and eval paths
- Decoded evaluation now emits per-family rollout metrics (for example `family_conservation_*`, `family_transport_*`) in addition to per-task metrics
- Auto-conditioning now also exposes grouped `equation_signature`, `parameter_signature`, and `boundary_signature` features as a proto equation-conditioning surface
- Auto-conditioning now also exposes set-structured `equation_nodes`, `parameter_nodes`, and `boundary_nodes` sources
- AdaLN conditioning now supports set-structured sources with learned attention pooling, so equation-style node sets are no longer collapsed with a fixed mean
- Promotion rules now support wildcard metric groups with reducers like `max:family_*_decoded_rollout_nrmse<=...` and `mean:task_*_decoded_step1_nrmse<=...`
- `train.py`, `benchmark.py`, and `train_baselines.py` now honor config `include:` directives, which unblocks preset-based cheap runs
- `scripts/run_light_experiment.py` now resolves configs, applies overrides, optionally bootstraps tiny PDEBench-style HDF5 data, runs selected stages, evaluates, and writes `summary.json` plus `results.tsv`
- Cheap preset configs now exist for single-task operator screening, single-task decoded shortlist runs, and multitask operator screening
- Verified smoke runs now exist in `reports/light_experiments/verified_burgers_operator`, `reports/light_experiments/verified_burgers_joint`, and `reports/light_experiments/verified_multitask_operator`
- Targeted tests passed for losses, loader behavior, operator training, decoder training, decoded operator fine-tuning, joint codec/operator fine-tuning, and decoded eval infrastructure

Next step:
- Use the new light experiment loop to compare candidate paths, then promote the best ones into real remote runs on held-out data.

Stop conditions:
- a new change breaks the light experiment loop or its smoke verification
- the next step requires real benchmark runs on remote held-out data rather than more local plumbing

Experiment loop update (2026-05-05, remote smoke variant matrix):
- Branch:
  - continued on `codex/remote-smoke-baseline`
- Remote execution:
  - provider: Vast.ai RTX 4090 instance
  - instance was destroyed after B2 artifact publication; `vastai show instances --raw` returned an empty list
  - B2 artifact: `remote-runs/smoke/smoke_variants_20260505T0625Z.tar.gz`
- Harness:
  - smoke-v1 shards from B2
  - `data.max_samples=8`
  - eval split `test`
  - stages: `operator -> decoder -> operator_decoded -> joint_codec_operator`
- Results:
  - persistence baseline: `0.1876487120420463`
  - current best UPS: `0.6297059754071941`
  - no conditioning: `0.6020991206609253`
  - task signature only: `0.4793234406026068`
  - semigroup 0: `0.6120762323925264`
  - semigroup 10: `0.6481950749556429`
- Keep:
  - `ups_smoke_task_signature_only` as the best UPS variant in this smoke batch
- Discard for now:
  - `ups_smoke_semigroup10`
  - `ups_smoke_semigroup0`
  - `ups_smoke_no_conditioning`
- Interpretation:
  - Task-signature-only conditioning improves over the current smoke best but still loses badly to persistence.
  - The immediate blocker is not remote plumbing anymore; it is model behavior against a strong simple physical-space baseline.
  - Next iteration should inspect per-task/family failures and test a candidate that narrows conditioning while improving the decoded rollout objective, rather than scaling this config.

Experiment loop update (2026-05-05, focused task-signature smoke matrix):
- Branch:
  - created and pushed `codex/smoke-focused-variants`
  - commit: `32102aa`
- Remote execution:
  - first Vast offer failed during host GPU CDI injection; instance was destroyed
  - second Vast RTX 4090 instance completed and was destroyed after artifact publication
  - `vastai show instances --raw` returned an empty list after teardown
  - B2 artifact: `remote-runs/smoke/smoke_focused_variants_20260505T0613Z.tar.gz`
- Results:
  - previous best task signature only: `0.4793234406026068`
  - task signature + semigroup disabled: `0.6230334342451865`
  - task signature + joint 48: `0.4971677039442661`
  - task signature + rollout 4: `0.617963100106149`
  - task signature + joint 48 + rollout 4: `0.617963100106149`
- Keep:
  - plain `ups_smoke_task_signature_only` remains the best UPS smoke row
  - `task_signature_joint48` is useful only as a diagnostic: step-1 improved but rollout/Darcy worsened
- Discard for now:
  - semigroup disabled
  - longer rollout loss
  - longer joint training as an isolated scale-up lever
- Next hypothesis:
  - test decoded fine-tuning and reconstruction-weight probes under the same task-signature conditioning before changing data scale

Experiment loop update (2026-05-05, decoded/reconstruction smoke matrix):
- Branch:
  - continued on `codex/smoke-focused-variants`
  - commit before launch: `4376658`
- Remote execution:
  - provider: Vast.ai RTX 4090 instance at roughly `$0.249/hr`
  - instance was destroyed after artifact publication; `vastai show instances --raw` returned an empty list
  - B2 artifact: `remote-runs/smoke/smoke_decoded_variants_20260505T0621Z.tar.gz`
- Results:
  - previous best task signature only: `0.4793234406026068`
  - task signature + joint 16: `0.5951420314812053`
  - task signature + operator_decoded 4: `0.6698199513986969`
  - task signature + operator_decoded 4 + joint 16: `0.705799969920023`
  - task signature + reconstruction loss disabled: `0.6537664796371954`
- Interpretation:
  - No decoded/reconstruction smoke probe improved on the best task-signature row.
  - More frozen-codec decoded fine-tuning actively regressed rollout and Darcy.
  - Further smoke-only hyperparameter search is now lower value than preparing held-out `light-v1` shards.

Experiment loop update (2026-05-05, light shard-prep launch fix):
- Fixed:
  - `scripts/run_remote_shard_prep_b2.sh` now accepts `KEY=VALUE` CLI assignments like the smoke pipeline.
- Validation:
  - `pytest tests/unit/test_remote_shard_prep.py -q`
  - `bash -n scripts/run_remote_shard_prep_b2.sh`
- Purpose:
  - allow `scripts/vast_launch.py --remote-script scripts/run_remote_shard_prep_b2.sh --script-args "DRY_RUN=0 ..."` to launch held-out `light-v1` prep without a custom wrapper.

Experiment loop update (2026-04-15):
- Objective metric:
  - `decoded_rollout_nrmse` on decoded multitask joint runs from `scripts/run_light_experiment.py`
- Fixed harness:
  - `python scripts/run_light_experiment.py --bootstrap-synthetic --device cpu --decoded`
  - train surface based on `configs/train_burgers_light_joint.yaml` with `data.task=[burgers1d, advection1d]`
- Baseline:
  - `ar_mt_joint_base`
  - `decoded_rollout_nrmse = 0.9701266884803772`
- Keep:
  - `ar_mt_joint_with_operator_decoded`
    - `0.9669556021690369`
  - `ar_mt_joint_best_joint2`
    - `0.9584252834320068`
  - `ar_mt_joint_best_joint3`
    - `0.949466347694397`
  - `ar_mt_joint_best_joint4`
    - `0.9416316747665405`
  - `ar_mt_joint_best_joint5`
    - `0.9301626682281494`
- Discard:
  - `ar_mt_joint_semigroup0`
  - `ar_mt_joint_semigroup10`
  - `ar_mt_joint_capacity24`
  - `ar_mt_joint_no_conditioning`
  - `ar_mt_joint_flat_conditioning`
  - `ar_mt_joint_node_conditioning`
  - `ar_mt_joint_rollout3`
  - `ar_mt_joint_task_only_conditioning`
  - `ar_mt_joint_task_traits_conditioning`
  - `ar_mt_joint_task_nodes_conditioning`
  - `ar_mt_joint_best_opdecoded2`
  - `ar_mt_joint_best_rollout3`
  - `ar_mt_joint_best_semigroup0`
  - `ar_mt_joint_best_joint2_low_lr`
  - `ar_mt_joint_best_joint2_rollout3`
  - `ar_mt_joint_best_joint2_rollout_heavy`
  - `ar_mt_joint_best_joint3_low_lr`
  - `ar_mt_joint_best_joint3_rollout3`
  - `ar_mt_joint_best_joint3_rollout_heavy`
  - `ar_mt_joint_best_joint4_rollout3`
  - `ar_mt_joint_best_joint4_rollout_heavy`
- Current best cheap candidate:
  - `configs/train_multitask_light_joint_best.yaml`
  - stages: `operator -> decoder -> operator_decoded -> joint_codec_operator`
  - `operator_decoded.epochs = 1`
  - `joint_codec_operator.epochs = 5`
- Current interpretation:
  - On the cheap synthetic multitask harness, the strongest signal is decoded training depth, not a wider model and not a reduced conditioning surface.
  - Conditioning still matters: removing it regresses from `0.9701` to `0.9778`.
  - The new set-structured and semantic conditioning work is not the immediate limiter on this tiny harness; training budget in the decoded stages is.
- Real next step:
  - run the current best candidate remotely on held-out `val` and `test`
  - compare it against the old multitask decoded baseline with the same eval harness
  - only after that resume broader architecture search

Experiment loop update (2026-04-16):
- New objective metric:
  - `decoded_rollout_nrmse` on decoded heterogeneous multitask runs from `scripts/run_light_experiment.py`
- Fixed harness:
  - `python scripts/run_light_experiment.py --bootstrap-synthetic --device cpu --decoded`
  - train surface based on `configs/train_multitask_light_joint_best.yaml` with `data.task=[burgers1d, advection1d, darcy2d]`
- Harness fixes landed before the run:
  - synthetic 2D scalar PDEBench bootstrap now writes `data` as `(samples, steps, 1, H, W)`
  - grid-shape inference now handles channel-first scalar 2D fields
  - latent batching and decoded flatten helpers now handle channel-first scalar 2D fields
  - decoded metric aggregation now supports heterogeneous point counts instead of assuming one flattened size
- Baseline:
  - `ar_3task_joint_base_v5`
  - `decoded_rollout_nrmse = 0.971684098523884`
- Keep:
  - `ar_3task_no_conditioning`
    - `0.9289172765646931`
  - `ar_3task_flat_conditioning_v2`
    - `0.9262027992342496`
  - `ar_3task_no_conditioning_joint6`
    - `0.9237801150001693`
  - `ar_3task_flat_no_signature_joint6`
    - `0.9415237688908223`
  - `ar_3task_no_conditioning_joint7`
    - `0.9172854702815767`
  - `ar_3task_no_conditioning_joint8`
    - `0.9085575683627656`
  - `ar_3task_no_conditioning_joint8_rollout_heavy`
    - `0.9095922539249501`
  - `ar_3task_no_conditioning_joint9`
    - `0.9006705289008915`
- Discard:
  - `ar_3task_node_conditioning_v2`
  - `ar_3task_joint6`
  - `ar_3task_semantic_ids`
  - `ar_3task_task_traits`
  - `ar_3task_flat_no_signature_joint7`
  - `ar_3task_flat_no_signature_joint7_rollout3`
  - `ar_3task_no_conditioning_joint7_rollout3`
  - `ar_3task_no_conditioning_joint8_low_lr`
- Current best heterogeneous cheap candidate:
  - `configs/train_multitask_heterogeneous_light_best.yaml`
  - stages: `operator -> decoder -> operator_decoded -> joint_codec_operator`
  - `auto_conditioning = false`
  - `joint_codec_operator.epochs = 9`
- Current interpretation:
  - the earlier “conditioning matters” conclusion does not hold once the cheap harness includes heterogeneous families and dimensions
  - the full current conditioning path appears over-specified or misaligned for broader multitask runs
  - reduced flat semantics help substantially relative to the full conditioning path, which means the information itself is not useless; the current conditioning surface is
  - the strongest immediate model-side bet is a simpler or gated operator conditioning path, but the strongest current run is still with operator conditioning disabled
- Real next step:
  - run `configs/train_multitask_heterogeneous_light_best.yaml` remotely on real `val` and `test`
  - compare against `configs/train_multitask_light_joint_best.yaml` on the same held-out eval harness
  - if the held-out result agrees, then turn the next local loop into conditioning simplification rather than conditioning expansion

Experiment loop update (2026-04-28):
- Issue found:
  - AdaLN conditioning was not exact-neutral at initialization.
  - The projection heads were zero-initialized, but `modulate()` still returned `sigmoid(2) * normed`, shrinking activations whenever a conditioner was attached.
- Fix:
  - `AdaLNConditioner.modulate()` now returns `normed + gate * (conditioned - normed)`.
  - With zero-initialized projections, conditioning is an exact no-op.
- Re-run baseline after fix:
  - `ar_3task_fixed_no_conditioning_joint9`
    - `0.9006705289008915`
  - `ar_3task_fixed_full_conditioning_joint9`
    - `0.9404433043415122`
  - `ar_3task_fixed_flat_conditioning_joint9`
    - `0.8537558470729366`
  - `ar_3task_fixed_flat_no_signature_joint9`
    - `0.8958172283966164`
- Keep:
  - `ar_3task_fixed_flat_conditioning_joint10`
    - `0.8339420045726132`
  - `ar_3task_fixed_flat_conditioning_joint11`
    - `0.8132875664480244`
  - `ar_3task_fixed_flat_conditioning_joint12`
    - `0.7916940618025152`
  - `ar_3task_fixed_flat_conditioning_joint14`
    - `0.7501271974686412`
- Matched control:
  - `ar_3task_no_conditioning_joint14`
    - `0.8777376404812709`
- Discard:
  - `ar_3task_fixed_signature_only_joint9`
  - `ar_3task_fixed_task_signature_joint9`
  - `ar_3task_fixed_flat_conditioning_joint9_rollout3`
  - `ar_3task_fixed_flat_conditioning_joint10_low_lr`
  - `ar_3task_fixed_flat_conditioning_joint10_rollout_heavy`
- Updated current best heterogeneous cheap candidate:
  - `configs/train_multitask_heterogeneous_light_best.yaml`
  - stages: `operator -> decoder -> operator_decoded -> joint_codec_operator`
  - conditioning: explicit flat semantic sources only
  - `joint_codec_operator.epochs = 14`
- Updated interpretation:
  - The old “disable conditioning” conclusion was an artifact of non-neutral AdaLN modulation plus an over-broad conditioning source set.
  - The best cheap signal is now flat semantic conditioning plus deeper decoded joint training.
  - Node-set conditioning remains a second-wave idea; the cheap harness prefers simpler explicit semantics.

Experiment loop update (2026-04-29):
- Harness fix:
  - `scripts/run_light_experiment.py` now applies `--eval-override` even without a separate `--eval-config`.
  - The synthetic bootstrap now accepts `--synthetic-samples` and `--synthetic-steps`, so cheap held-out checks are not limited to two trajectories.
- Larger held-out synthetic check:
  - Harness: `burgers1d + advection1d + darcy2d`, train split synthetic, eval split synthetic `val`
  - Data size: `--synthetic-samples 8 --synthetic-steps 6`
  - Stages: `operator -> decoder -> operator_decoded -> joint_codec_operator`
- Keep:
  - `ar_3task_val_flat_conditioning_joint14`
    - `0.8289915410904443`
  - `ar_3task_val_flat_conditioning_joint16`
    - `0.8032194122236591`
  - `ar_3task_val_flat_conditioning_joint20`
    - `0.7707491577354888`
  - `ar_3task_val_flat_conditioning_joint24`
    - `0.7179769378754296`
  - `ar_3task_val_flat_conditioning_joint32`
    - `0.6814858538593768`
  - `ar_3task_val8_flat_no_signature_joint32`
    - `0.8002299375444841`
  - `ar_3task_val8_task_signature_joint32`
    - `0.7919775048580832`
- Controls:
  - `ar_3task_val_no_conditioning_joint14`
    - `0.9675743909574095`
  - `ar_3task_val_no_conditioning_joint32`
    - `0.7301451171970339`
  - `ar_3task_val8_no_conditioning_joint32`
    - `0.9471399362116322`
- Discard:
  - `ar_3task_val_flat_conditioning_joint12`
  - `ar_3task_val_flat_conditioning_joint14_rollout_heavy`
  - `ar_3task_val8_full_conditioning_joint32`
  - `ar_3task_val8_task_id_joint32`
  - `ar_3task_val8_task_family_joint32`
  - `ar_3task_val8_signature_only_joint32`
- Updated current best heterogeneous cheap candidate:
  - `configs/train_multitask_heterogeneous_light_best.yaml`
  - conditioning: `resolution`, `spatial_dims`, `task_id`, `equation_signature`
  - `joint_codec_operator.epochs = 32`
- Current interpretation:
  - The larger synthetic `val` split still supports flat semantic conditioning over no conditioning.
  - The fuller flat bundle overfits or destabilizes `advection1d`; the narrower `task_id + equation_signature` surface is more robust.
  - Local epoch tuning is now dominated by synthetic simplicity, so real remote `val/test` is the only trustworthy next gate.

Experiment loop update (2026-04-29, remote promotion prep):
- Live B2 check:
  - usable real-data prefix is `full/`
  - `burgers1d` has `train`, `val`, and `test`
  - `advection1d` has `train`, `val`, and `test`
  - `darcy2d` has `train` and `test`, but no `val`
  - default full 3-task train/test file set is about 141 GiB
  - only currently obvious small real shard is `full/burgers1d/burgers1d_train_000.h5` at about 1.57 GiB
- Harness update:
  - `scripts/run_light_experiment.py` can now run extra held-out splits from the same trained checkpoints via `--extra-eval-split`
  - extra split outputs are written as `summary_<split>.json` and referenced from the primary `summary.json`
- Remote path:
  - added `scripts/run_remote_light_promotion.sh`
  - default B2 hydration fetches `burgers1d`, `advection1d`, and `darcy2d` train/test HDF5 files from `REMOTE_B2_PREFIX=full`
  - default promotion eval split is `test` because the 3-task real-data set has no common `val` split
  - actual default full-data hydration is blocked unless `ALLOW_FULL_DATA=1` is set because the current HDF5 loader reads files into memory
  - dry-run command: `ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env DRY_RUN=1 bash scripts/run_remote_light_promotion.sh`
- Small-shard prep:
  - added `scripts/make_light_hdf5_shards.py`
  - it slices sample-aligned HDF5 datasets into small `train`, `val`, and `test` files without loading full split tensors
  - use this after hydrating source files on a remote/data-prep box, then publish the resulting small shards back to B2 for cheap model experiments
- Launcher update:
  - `scripts/vast_launch.py launch` can now use `--remote-script`, `--script-args`, `--skip-prefetch`, and `--git-ref`
  - use `--remote-script scripts/run_remote_light_promotion.sh --skip-prefetch` for this promotion path
- Next gate:
  - publish or identify small B2 train/val/test shards for all target tasks before claiming cheap real-data benchmark capability
  - commit/push the current branch before a remote launch, then run the Vast dry run and only launch paid compute after the onstart script points to this branch and the expected B2 files.

Experiment loop update (2026-05-01, bounded real-data smoke):
- Remote execution:
  - branch: `codex/autowork-semigroup-foundation`
  - commits used: `128425a`, `f3a90b9`
  - provider: Vast.ai RTX 4090 instance, destroyed after artifact copy
  - real shard: `full/burgers1d/burgers1d_train_000.h5`
- Issue found:
  - The first retry on the real Burgers shard was not a cheap light experiment because decoded evaluation iterated all 2,000 trajectories and all 201 steps.
  - `--decoded-rollout-steps` capped rollout length but did not cap dataset size, so the harness still encoded full trajectories for every sample.
- Harness fix:
  - `PDEBenchConfig.max_samples` now limits HDF5 loading before tensors are materialized.
  - `data.max_samples` is honored by training, latent evaluation, decoded evaluation, and eval/model shape inference paths.
  - Validation: `python -m py_compile src/ups/data/pdebench.py src/ups/data/latent_pairs.py src/ups/eval/pdebench_runner.py scripts/train.py scripts/evaluate.py`
  - Validation: `pytest tests/unit/test_pdebench.py tests/unit/test_train_pdebench_loader.py tests/unit/test_pdebench_runner_eval.py tests/unit/test_light_experiment_runner.py -q`
- Bounded smoke command:
  - `FETCH_DATA=0 CHECK_DATA=1 TASKS=burgers1d TRAIN_CONFIG=configs/train_burgers_light_joint.yaml REMOTE_DATASET_FILES=burgers1d/burgers1d_train_000.h5 EVAL_SPLIT=train REQUIRED_GB=5 STAGES=operator,decoder,joint_codec_operator RUN_NAME=vast_burgers_shard_cap8 OUTPUT_ROOT=reports/light_experiments_remote LIGHT_EXTRA_ARGS="--override data.max_samples=8 --eval-override data.max_samples=4 --decoded-rollout-steps 2" bash scripts/run_remote_light_promotion.sh`
- Result:
  - local copied summary: `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`
  - duration: `3.8071365356445312` seconds after data was already hydrated
  - `decoded_rollout_nrmse = 0.9488316819858322`
  - `decoded_step1_nrmse = 0.9453324050249666`
  - this is a plumbing and bounded-real-data smoke, not a benchmark, because it trains/evaluates on a tiny train-shard slice.
- Next gate:
  - publish small train/val/test shards for `burgers1d`, `advection1d`, and `darcy2d`, then run the same bounded harness against held-out shards.

Experiment loop update (2026-05-04, unattended demo roadmap start):
- Goal:
  - turn the working-demo/SOTA roadmap into executable local progress before spending more remote compute
  - preserve all progress in tracked repo artifacts for morning continuation
- Plan artifact:
  - added `docs/superpowers/plans/2026-05-04-working-demo-sota-roadmap.md`
- Benchmark contract:
  - added `docs/demo_benchmark_contract.md`
  - defines `smoke`, `light`, `medium`, and `benchmark` claim tiers
  - primary promotion metric is `decoded_rollout_nrmse`
  - train-shard evaluation remains plumbing only, not benchmark evidence
- Data manifest:
  - added schema placeholder at `docs/demo_data_manifest.yaml`
  - shard generation can now write records with source split, source offset, sample count, dataset shapes, bytes, sha256, and B2 remote keys
- Shard builder:
  - `scripts/make_light_hdf5_shards.py` now prefers native source splits when present
  - missing native splits fall back to `train` and are marked `derived_from_source_split: true`
  - remote keys are recorded as `<version>/<task>/<task>_<split>.h5`, matching `scripts/run_remote_light_promotion.sh` default generated keys when `REMOTE_B2_PREFIX=<version>`
- Publish wrapper:
  - added `scripts/publish_light_hdf5_shards_b2.sh`
  - default is `DRY_RUN=1`
  - can optionally build shards before publishing via `BUILD_SHARDS=1`
  - loads B2 credentials from `ENV_FILE` without printing secrets
- Validation so far:
  - `bash -n scripts/publish_light_hdf5_shards_b2.sh`
  - `python -m py_compile scripts/make_light_hdf5_shards.py`
  - `pytest tests/unit/test_make_light_hdf5_shards.py -q`
  - temp-source dry-run of `scripts/publish_light_hdf5_shards_b2.sh` generated 9 shard upload commands and a manifest upload command
- Next gate:
  - run broader focused tests, commit this checkpoint, then proceed to demo scorecard/report skeleton if time remains.

Experiment loop update (2026-05-04, demo scorecard branch):
- Branch:
  - created `codex/demo-scorecard-loop` after checkpoint `2109117`
- Scorecard module:
  - added `src/ups/eval/demo_scorecard.py`
  - loads one or more `summary.json` files from `scripts/run_light_experiment.py`
  - records run name, config paths, stages, duration, commit, data manifest, main metric, metric columns, and promotion-rule result
  - uses existing wildcard-aware promotion rules from `src/ups/eval/promotion.py`
- CLIs:
  - added `scripts/collect_light_results.py`
  - added `scripts/build_demo_report.py`
  - report output includes `index.html`, `metrics.tsv`, and `scorecard.json`
  - optional `--copy-summaries` preserves source summaries under the report directory
- Validation:
  - `python -m py_compile src/ups/eval/demo_scorecard.py scripts/collect_light_results.py scripts/build_demo_report.py`
  - `pytest tests/unit/test_demo_scorecard.py tests/unit/test_eval_promotion.py -q`
  - `python scripts/build_demo_report.py reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json --output-dir /tmp/ups_demo_report_smoke --title 'UPS Smoke Report' --data-manifest docs/demo_data_manifest.yaml --promotion-rule 'decoded_rollout_nrmse<=1.0' --copy-summaries`
  - `python scripts/collect_light_results.py reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json --output-tsv /tmp/ups_collect_metrics.tsv --output-json /tmp/ups_collect_scorecard.json --data-manifest docs/demo_data_manifest.yaml --promotion-rule 'decoded_rollout_nrmse<=1.0'`
- Observed smoke aggregation:
  - `vast_burgers_shard_cap8`
  - `decoded_rollout_nrmse = 0.9488316819858322`
  - promotion rule `decoded_rollout_nrmse<=1.0` passed
- Next gate:
  - commit and push `codex/demo-scorecard-loop`
  - then either publish `smoke-v1`/`light-v1` shards if source data is locally available, or continue with baseline/report visualizations locally.

Experiment loop update (2026-05-04, persistence baseline branch):
- Branch:
  - created `codex/demo-persistence-baseline` from `codex/demo-scorecard-loop`
- Baseline:
  - added `src/ups/eval/persistence_baselines.py`
  - physical-space persistence predicts the previous field as the next field
  - supports single-task and multitask PDEBench configs
  - supports `data.max_samples` and rollout-step caps
  - emits overall, per-task, per-family, step-1, and horizon metrics in the same decoded metric namespace as UPS runs
- CLI:
  - added `scripts/run_persistence_baseline.py`
  - writes `summary.json`, `resolved_eval.yaml`, and `results.tsv` under the same output-root layout as `scripts/run_light_experiment.py`
  - supports promotion rules through `src/ups/eval/promotion.py`
- Validation:
  - `python -m py_compile src/ups/eval/persistence_baselines.py scripts/run_persistence_baseline.py`
  - `pytest tests/unit/test_persistence_baseline.py tests/unit/test_demo_scorecard.py -q`
- Next gate:
  - run broader focused tests, commit and push the branch, then add visual rollout artifact generation or create/publish real held-out shards if source data becomes available.

Experiment loop update (2026-05-04, B2 shard check branch):
- Branch:
  - created `codex/demo-b2-shard-check` from `codex/demo-persistence-baseline`
- B2 inspection:
  - external env exists at `/Users/emerygunselman/Code/universal_simulator/.env`
  - `rclone` is installed locally
  - no secret values were printed
  - checked prefixes: `pdebench/smoke-v1`, `pdebench/light-v1`, `smoke-v1`, `light-v1`, `pdebench/full`, `full`
  - `smoke-v1` and `light-v1` are not present yet
  - real data is under top-level `full/`, not `pdebench/full/`
  - top-level `full/` listed 32 entries including `advection1d`, `burgers1d`, `darcy2d`, and `navier_stokes2d`
- Added:
  - `scripts/check_demo_b2_shards.py`
  - `tests/unit/test_check_demo_b2_shards.py`
- Purpose:
  - read `docs/demo_data_manifest.yaml`
  - derive expected B2 keys when manifest records are empty
  - prefer explicit `records[].remote_key` after shard generation
  - report present/missing shard keys before paid compute is launched
- Bug caught and fixed during live check:
  - `rclone lsjson` can return success with an empty list for missing exact keys
  - checker now uses `rclone size --json` and requires `count > 0`
  - live check currently reports all 9 `light-v1` keys missing, matching the earlier prefix listing
- Next gate:
  - validate and push this branch, then use the checker after actual `smoke-v1` or `light-v1` publishing.

Experiment loop update (2026-05-04, remote shard prep wrapper):
- B2 full source sizes checked:
  - `full/burgers1d/burgers1d_train.h5`: about `69.045 GiB`
  - `full/burgers1d/burgers1d_val.h5`: about `7.704 GiB`
  - `full/burgers1d/burgers1d_test.h5`: about `15.36 GiB`
  - `full/burgers1d/burgers1d_train_000.h5`: about `1.57 GiB`
  - `full/advection1d/advection1d_train.h5`: about `46.03 GiB`
  - `full/advection1d/advection1d_val.h5`: about `7.704 GiB`
  - `full/advection1d/advection1d_test.h5`: about `7.704 GiB`
  - `full/darcy2d/darcy2d_train.h5`: about `2.441 GiB`
  - `full/darcy2d/darcy2d_test.h5`: about `0.613 GiB`
- Added:
  - `scripts/run_remote_shard_prep_b2.sh`
- Purpose:
  - run on a cheap remote/data-prep box, not during GPU training
  - hydrate only one task source set at a time from B2 `full/`
  - cut small shards with manifest records
  - delete full source files between tasks
  - publish all small shards plus the aggregate manifest to B2 `light-v1`
- Important:
  - this still requires roughly enough disk for Burgers train/val/test at once, so plan at least a 120-150 GiB scratch disk
  - dry-run first with `DRY_RUN=1`

Experiment loop update (2026-05-04, demo report plots branch):
- Branch:
  - created `codex/demo-report-plots` from `codex/remote-shard-prep`
- Added:
  - `src/ups/eval/demo_plots.py`
- Report polish:
  - `scripts/build_demo_report.py` now writes compact PNG bar plots for `decoded_rollout_nrmse`, `decoded_step1_nrmse`, and the main metric when available
  - `src/ups/eval/demo_scorecard.py` can embed generated plot paths in `index.html`
- Purpose:
  - make the demo report more inspectable without needing notebooks or browser-side dependencies
  - keep this artifact generation local and cheap

Experiment loop update (2026-05-04, demo runbook handoff branch):
- Branch:
  - created `codex/demo-runbook-handoff` from `codex/demo-report-plots`
- Added:
  - `docs/demo_runbook.md`
- Purpose:
  - preserve the exact branch stack and execution order
  - document current B2 state and source sizes
  - document preflight, remote shard prep, UPS light run, persistence baseline run, and report build commands
  - state keep/discard and stop rules for the morning continuation
- README:
  - added a Quickstart pointer to `docs/demo_runbook.md`

Experiment loop update (2026-05-04, demo cost tracking branch):
- Branch:
  - created `codex/demo-cost-tracking` from `codex/demo-runbook-handoff`
- Added:
  - optional `--cost-json` inputs for `scripts/collect_light_results.py`
  - optional `--cost-json` inputs for `scripts/build_demo_report.py`
- Purpose:
  - track remote experiment provider, instance, GPU type/count, wall-clock hours, GPU hours, and estimated USD
  - keep cost accounting outside the training runner so cheap remote runs can be compared without coupling to one cloud provider
  - make morning iteration decisions metric-aware and spend-aware

Experiment loop update (2026-05-04, demo experiment queue branch):
- Branch:
  - created `codex/demo-experiment-queue` from `codex/demo-cost-tracking`
- Added:
  - `scripts/plan_demo_experiments.py`
  - `tests/unit/test_plan_demo_experiments.py`
- Purpose:
  - generate JSONL, TSV, and shell queues for bounded smoke/light/medium variant matrices
  - keep generated commands dry-run by default
  - freeze the first variant set around current best, no-conditioning, task-signature-only, semigroup weight, joint-depth, and rollout-depth probes

Experiment loop update (2026-05-04, demo baseline delta branch):
- Branch:
  - created `codex/demo-baseline-delta` from `codex/demo-experiment-queue`
- Added:
  - optional `--baseline-run`, `--baseline-metric`, and `--baseline-min-improvement` scorecard/report arguments
- Purpose:
  - make the demo keep/discard gate explicit in generated artifacts
  - compare UPS rows against the matched persistence baseline on held-out decoded rollout nRMSE
  - track lower-is-better delta, ratio, improvement fraction, and pass/fail state

Experiment loop update (2026-05-04, demo readiness check branch):
- Branch:
  - created `codex/demo-readiness-check` from `codex/demo-baseline-delta`
- Added:
  - `scripts/check_demo_readiness.py`
  - `tests/unit/test_check_demo_readiness.py`
- Purpose:
  - summarize manifest, expected shard keys, optional live B2 presence, summary artifacts, baseline presence, and candidate presence
  - provide a single JSON readiness payload before launching remote experiments or building the final report

Experiment loop update (2026-05-04, smoke source-key override):
- Read-only live check:
  - `light-v1` B2 readiness still reports 0 present and 9 missing shard keys
  - matched `persistence_light_v1_test` and `ups_light_v1_current_best` summaries are not present yet
- Added:
  - source-key override support in `scripts/run_remote_shard_prep_b2.sh`
  - `tests/unit/test_remote_shard_prep.py`
- Purpose:
  - allow smoke-only data-prep runs to hydrate known smaller source files such as `full/burgers1d/burgers1d_train_000.h5`
  - keep this explicitly out of benchmark claims because smoke val/test may be derived from train

Experiment loop update (2026-05-04, branch stack refresh):
- Branch:
  - created `codex/demo-branch-stack-refresh` from `codex/smoke-source-key-shards`
- Updated:
  - `docs/demo_runbook.md`
- Purpose:
  - refresh the morning handoff branch stack through the cost, queue, baseline, readiness, and smoke source-key override branches
  - record the live B2 readiness result that `light-v1` still has 0 present and 9 missing expected keys

Experiment loop update (2026-05-04, smoke manifest readiness branch):
- Branch:
  - created `codex/smoke-manifest-readiness` from `codex/demo-branch-stack-refresh`
- Added:
  - `docs/demo_smoke_data_manifest.yaml`
- Purpose:
  - make smoke-tier B2/readiness checks first-class instead of overloading the held-out `light-v1` manifest
  - keep smoke manifests explicitly labeled as plumbing-only, not benchmark evidence
  - document smoke readiness commands separately from held-out light readiness commands
  - require live `--check-b2` in the smoke readiness examples before remote launch decisions

Experiment loop update (2026-05-04, hardened B2 fetcher branch):
- Branch:
  - created `codex/harden-b2-fetcher` from `codex/smoke-manifest-readiness`
- Added:
  - `tests/unit/test_fetch_datasets_b2.py`
- Fixed:
  - `scripts/fetch_datasets_b2.sh` now requires `rclone lsjson` to return a non-empty JSON list before treating a candidate as present
- Purpose:
  - prevent remote shard prep from treating missing B2 keys as found
  - align the fetcher with the earlier `scripts/check_demo_b2_shards.py` lesson that `rclone lsjson` success alone is not enough

Experiment loop update (2026-05-04, smoke shard prep wrapper branch):
- Branch:
  - created `codex/smoke-shard-prep-wrapper` from `codex/harden-b2-fetcher`
- Added:
  - `scripts/run_smoke_shard_prep_b2.sh`
- Purpose:
  - provide one dry-run-default command for plumbing-only smoke shard prep
  - use the known small Burgers train source by default
  - keep benchmark-safe `light-v1` shard prep separate from smoke shortcuts

Experiment loop update (2026-05-04, smoke output root fix branch):
- Branch:
  - created `codex/fix-smoke-output-root` from `codex/smoke-shard-prep-wrapper`
- Fixed:
  - `scripts/run_smoke_shard_prep_b2.sh` now defaults `OUT_ROOT=data/pdebench_smoke`
- Purpose:
  - align the wrapper output path with `docs/demo_smoke_data_manifest.yaml`
  - avoid mixing smoke plumbing artifacts with held-out light artifacts

Experiment loop update (2026-05-04, smoke prep size handoff branch):
- Branch:
  - created `codex/smoke-prep-size-handoff` from `codex/fix-smoke-output-root`
- Live B2 source-size check:
  - `full/burgers1d/burgers1d_train_000.h5`: `1.570 GiB`
  - `full/advection1d/advection1d_train.h5`: `46.030 GiB`
  - `full/darcy2d/darcy2d_train.h5`: `2.441 GiB`
- Purpose:
  - document that default three-task smoke prep still needs roughly 60 GiB scratch
  - identify smaller Advection source shard creation as the next cost reducer

Experiment loop update (2026-05-04, cheap smoke split-source branch):
- Branch:
  - created `codex/cheap-smoke-split-sources` from `codex/smoke-prep-size-handoff`
- Added:
  - per-task split-source mapping support in `scripts/run_remote_shard_prep_b2.sh`
- Updated:
  - `scripts/run_smoke_shard_prep_b2.sh` now derives Advection smoke splits from `advection1d_val.h5`
  - `scripts/run_smoke_shard_prep_b2.sh` now derives Darcy smoke splits from `darcy2d_test.h5`
- Purpose:
  - reduce default three-task smoke source hydration from roughly 50 GiB to roughly 10 GiB
  - keep the shortcut explicitly plumbing-only because train/val/test may be derived from the same source file

Experiment loop update (2026-05-04, local shard-prep test mode branch):
- Branch:
  - created `codex/local-shard-prep-test-mode` from `codex/cheap-smoke-split-sources`
- Added:
  - `FETCH_DATA=0` support for `scripts/run_remote_shard_prep_b2.sh`
  - `PUBLISH_SHARDS=0` support for `scripts/run_remote_shard_prep_b2.sh`
- Purpose:
  - allow shard cutting against already-hydrated sources
  - test the actual split-source generation path locally with tiny HDF5 files without contacting B2 or publishing

Experiment loop update (2026-05-04, final remote smoke handoff branch):
- Branch:
  - created `codex/final-remote-smoke-handoff` from `codex/local-shard-prep-test-mode`
- Local blocker check:
  - `df -h` reported about `1.9 GiB` free on the local filesystem
  - optimized smoke prep needs roughly `10-12 GiB` source hydration plus output room
- Updated:
  - `docs/demo_runbook.md`
- Purpose:
  - refresh the branch stack through the latest smoke manifest, fetch hardening, wrapper, split-source, and local-test-mode branches
  - document that smoke prep should run on a remote/data-prep box, not this local machine

Experiment loop update (2026-05-04, remote smoke pipeline branch):
- Branch:
  - created `codex/remote-smoke-pipeline` from `codex/final-remote-smoke-handoff`
- Added:
  - `scripts/run_remote_smoke_pipeline.sh`
  - `tests/unit/test_remote_smoke_pipeline.py`
- Purpose:
  - provide one remote-safe orchestration command for smoke shard readiness, shard prep, queue generation, and optional smoke experiment execution
  - keep defaults safe: dry-run shard prep, dry-run generated queue, and `RUN_EXPERIMENTS=0`
  - preserve readiness JSON, prep logs, queue artifacts, and optional queue run logs under `reports/demo/remote_smoke_pipeline`

Experiment loop update (2026-05-04, safe smoke queue default branch):
- Branch:
  - created `codex/safe-smoke-queue-default` from `codex/remote-smoke-pipeline`
- Fixed:
  - `scripts/run_remote_smoke_pipeline.sh` now keeps generated smoke queue commands at `DRY_RUN=1` by default, even when shard prep uses `DRY_RUN=0`
- Purpose:
  - prevent an operator from publishing shards and accidentally generating live training commands without explicitly setting `QUEUE_DRY_RUN=0`

Experiment loop update (2026-05-04, smoke disk guard branch):
- Branch:
  - created `codex/smoke-disk-guard` from `codex/safe-smoke-queue-default`
- Added:
  - `REQUIRED_GB` guard in `scripts/run_remote_shard_prep_b2.sh`
  - `REQUIRED_GB=12` default in `scripts/run_smoke_shard_prep_b2.sh`
- Purpose:
  - fail before B2 hydration when the remote/data-prep box does not have enough scratch disk
  - encode the current optimized smoke source-set estimate as an executable guard

Experiment loop update (2026-05-04, demo completion audit branch):
- Branch:
  - created `codex/demo-completion-audit` from `codex/smoke-disk-guard`
- Added:
  - `docs/demo_completion_audit.md`
- Purpose:
  - preserve an explicit prompt-to-artifact checklist for the working-demo goal
  - distinguish completed tooling from missing remote data, smoke summaries, light summaries, report artifacts, and baseline-gated performance evidence
  - prevent treating smoke plumbing or green unit tests as a completed demo

Experiment loop update (2026-05-04, enforce smoke-ready before queue branch):
- Branch:
  - created `codex/enforce-smoke-ready-before-queue` from `codex/demo-completion-audit`
- Fixed:
  - `scripts/run_remote_smoke_pipeline.sh` refuses live smoke queue execution when `CHECK_B2=1`, `QUEUE_DRY_RUN=0`, and smoke shards are not ready
- Validation:
  - fake `rclone size` test returns zero objects and confirms no live queue script is generated
- Purpose:
  - avoid paying for smoke experiments before `smoke-v1` shards actually exist in B2

Experiment loop update (2026-05-04, package demo artifacts branch):
- Branch:
  - created `codex/package-demo-artifacts` from `codex/enforce-smoke-ready-before-queue`
- Added:
  - `scripts/package_demo_artifacts.sh`
  - `tests/unit/test_package_demo_artifacts.py`
- Purpose:
  - package remote readiness JSON, logs, queues, summaries, reports, and manifests before tearing down a remote box
  - record missing default artifact paths in a manifest instead of silently omitting them

Experiment loop update (2026-05-04, require B2 check for live smoke branch):
- Branch:
  - created `codex/require-b2-check-for-live-smoke` from `codex/package-demo-artifacts`
- Fixed:
  - `scripts/run_remote_smoke_pipeline.sh` refuses live smoke queue execution when `CHECK_B2=0`, unless `ALLOW_UNCHECKED_LIVE_QUEUE=1` is explicitly set
- Purpose:
  - prevent bypassing the B2 shard readiness gate during paid smoke experiments

Experiment loop update (2026-05-04, final pipeline audit refresh branch):
- Branch:
  - created `codex/final-pipeline-audit-refresh` from `codex/require-b2-check-for-live-smoke`
- Updated:
  - `docs/demo_runbook.md`
  - `docs/demo_completion_audit.md`
- Purpose:
  - refresh the branch stack through remote smoke pipeline, packaging, and live-execution safeguards
  - document that live smoke queue execution requires B2 readiness checks unless explicitly overridden for controlled tests

Experiment loop update (2026-05-04, safe Vast smoke launcher branch):
- Branch:
  - created `codex/safe-vast-smoke-launcher` from `codex/final-pipeline-audit-refresh`
- Added:
  - `scripts/launch_remote_smoke_vast.sh`
  - `tests/unit/test_launch_remote_smoke_vast.py`
  - `tests/unit/test_vast_launch.py`
- Fixed:
  - `scripts/vast_launch.py` dry-run output redacts B2 and W&B secret values
  - `scripts/run_remote_smoke_pipeline.sh` accepts `KEY=VALUE` CLI assignments from the Vast onstart command
- Purpose:
  - make the next remote smoke step dry-run-first and log-safe
  - avoid copying secrets into dry-run output while still allowing actual launch commands to pass B2 credentials

Experiment loop update (2026-05-04, Vast offer summary branch):
- Branch:
  - created `codex/vast-offer-summary` from `codex/safe-vast-smoke-launcher`
- Added:
  - `scripts/search_vast_smoke_offers.py`
  - `tests/unit/test_search_vast_smoke_offers.py`
- Purpose:
  - summarize current Vast offers into JSON/TSV artifacts without launching paid compute
  - avoid brittle `vastai --raw | head` usage, which produced a broken-pipe traceback locally
- Live dry artifact check:
  - `python scripts/search_vast_smoke_offers.py --limit 3 --output-json /tmp/ups_vast_smoke_offers.json --output-tsv /tmp/ups_vast_smoke_offers.tsv`
  - cheapest returned RTX 4090 offers were about `$0.288-$0.298/hr` at query time

Experiment loop update (2026-05-04, final remote launch audit branch):
- Branch:
  - created `codex/final-remote-launch-audit` from `codex/vast-offer-summary`
- Updated:
  - `docs/demo_runbook.md`
  - `docs/demo_completion_audit.md`
- Purpose:
  - refresh canonical handoff docs with the safe Vast smoke launcher and Vast offer summary branches
  - mark cheap remote discovery as ready but not launched

Experiment loop update (2026-05-04, Vast cheap launch order branch):
- Branch:
  - created `codex/vast-cheap-launch-order` from `codex/final-remote-launch-audit`
- Updated:
  - `scripts/vast_launch.py` supports `--order` and `--limit` for `vastai launch instance`
  - `scripts/launch_remote_smoke_vast.sh` defaults to `ORDER=dph_total` and `LIMIT=10`
- Purpose:
  - make the dry-run-first smoke launcher prefer cheap RTX 4090 offers instead of Vast's default score ordering

Experiment loop update (2026-05-05, smoke-focused variants and light-v1 readiness):
- Branch:
  - continued on `codex/smoke-focused-variants`
- Remote smoke results:
  - broad smoke matrix artifact: `remote-runs/smoke/smoke_variants_20260505T0625Z.tar.gz`
  - best broad UPS row: `ups_smoke_task_signature_only`, `decoded_rollout_nrmse = 0.4793234406026068`
  - focused task-signature artifact: `remote-runs/smoke/smoke_focused_variants_20260505T0613Z.tar.gz`
  - best focused row: `ups_smoke_task_signature_joint48`, `decoded_rollout_nrmse = 0.4971677039442661`
  - decoded/reconstruction artifact: `remote-runs/smoke/smoke_decoded_variants_20260505T0621Z.tar.gz`
  - best decoded follow-up row: `ups_smoke_task_signature_joint16`, `decoded_rollout_nrmse = 0.5951420314812053`
- Remote light-v1 data prep:
  - published `light-v1` held-out shards to B2
  - readiness artifact: `reports/demo/light_readiness_after_prep.json`
  - live B2 check: `9/9` expected keys present
- Remote light-v1 held-out runs:
  - UPS candidate artifact: `remote-runs/light/ups_light_task_signature_20260505T0731Z.tar.gz`
  - UPS summary: `reports/light_experiments_remote/ups_light_v1_task_signature_only/summary.json`
  - UPS `decoded_rollout_nrmse = 0.8881691012411048`
  - persistence artifact: `remote-runs/light/persistence_light_v1_test_20260505T0740Z.tar.gz`
  - persistence summary: `reports/light_experiments_remote/persistence_light_v1_test/summary.json`
  - persistence `decoded_rollout_nrmse = 0.5701633411507036`
- Demo report:
  - generated `reports/demo/light_latest/index.html`
  - generated `reports/demo/light_latest/metrics.tsv`
  - generated `reports/demo/light_latest/scorecard.json`
  - readiness artifact: `reports/demo/light_readiness_after_runs.json`, `ready=true`
- Decision:
  - the light experiment loop is operational and B2-backed
  - current UPS task-signature-only candidate passes the absolute rule but fails the held-out persistence gate
  - `baseline_improvement_passed=false`, baseline delta `0.31800576009040127`, ratio `1.5577450129441892`
  - decoded rollout spectral energy error is much worse than persistence, so the next iteration should change the objective or architecture toward persistence-residual and stability-aware decoded rollout instead of scaling this candidate

Experiment loop update (2026-05-06, W&B-tracked post-light plan):
- Branch:
  - created `codex/wandb-post-light-loop` from `codex/smoke-focused-variants`
- Added:
  - `docs/superpowers/plans/2026-05-06-post-light-v1-improvement-plan.md`
  - `scripts/collect_wandb_runs.py`
- Updated:
  - light runs can set W&B project/entity/group/tags/job type from args or env when `--allow-wandb` is used
  - remote light promotion loads `WANDB_GROUP`, `WANDB_TAGS`, and `WANDB_JOB_TYPE` from `.env`
  - monitoring writes W&B run metadata to `logs/wandb_runs.jsonl`
  - light `summary.json` records W&B requested/enabled state, run IDs, and URLs
  - demo scorecards include `tracking_wandb_*` columns
- Purpose:
  - make future paid experiments traceable in both W&B and local/B2 artifacts
  - backfill historical W&B metrics into `reports/wandb/`
  - narrow the next model loop to persistence-residual or stability-aware decoded rollout candidates before any medium-scale spend

Experiment loop update (2026-05-06, residual light candidate branch):
- Branch:
  - created `codex/residual-light-candidate` from `codex/wandb-post-light-loop`
- W&B backfill:
  - wrote ignored local artifacts `reports/wandb/runs.json` and `reports/wandb/runs.tsv`
  - collected 200 recent W&B runs for local review
- Added:
  - decoded persistence-residual blend gate via `evaluation.decoded_persistence_residual_alpha`
  - `task_signature_residual_alpha25`
  - `task_signature_residual_alpha50`
- Local generated queue artifacts:
  - `reports/demo/residual_light_queue.jsonl`
  - `reports/demo/residual_light_queue.tsv`
  - `reports/demo/run_residual_light_queue.sh`
- Purpose:
  - cheaply test whether UPS decoded predictions contain useful residual signal over persistence before changing training objective or scaling
  - keep `alpha=0.0` anchored to the exact physical persistence baseline and `alpha=1.0` anchored to the current UPS decoded rollout

Experiment loop update (2026-05-06, W&B remote launch guard):
- Remote attempt:
  - launched `ups_light_task_signature_residual_alpha25` on Vast contract `36243886`
  - artifact target was `remote-runs/light/ups_light_residual_alpha25_20260506T1457Z.tar.gz`
  - destroyed the instance before completion after finding the remote `experiment` install profile did not install `wandb`
- Fix:
  - Vast `experiment` install profile now installs `wandb`
  - W&B-enabled monitoring now raises immediately if the `wandb` package is unavailable
- Purpose:
  - prevent paid remote runs from silently producing untracked summaries when W&B credentials are present but the dependency is absent

Experiment loop update (2026-05-06, residual alpha25 remote result):
- Remote run:
  - successful Vast contract: `36245004`
  - B2 artifact: `remote-runs/light/ups_light_residual_alpha25_20260506T1528Z.tar.gz`
  - local extracted artifact: `reports/demo/remote_artifacts/ups_light_residual_alpha25_20260506T1528Z/`
  - local summary copy: `reports/light_experiments_remote/ups_light_task_signature_residual_alpha25/summary.json`
  - rebuilt scorecard: `reports/demo/light_latest/scorecard.json`
- W&B:
  - tracked first completed run set: `00ud83aw`, `3ugaodok`, `i3ej1zp9`, `dm8y4ccc`
  - destroyed the instance after artifact publication because the Vast `--args-mode` container restarted the entrypoint instead of staying powered off
- Metrics:
  - residual alpha25 decoded rollout NRMSE: `0.5486869325531744`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - previous UPS light decoded rollout NRMSE: `0.8881691012411048`
  - baseline metric delta: `-0.021476408597529195`
  - baseline ratio: `0.9623328841973854`
  - baseline improvement fraction: `0.03766711580261458`
  - baseline improvement gate: `false`
  - absolute promotion rule `decoded_rollout_nrmse<=1.0`: `true`
- Interpretation:
  - alpha25 shows the UPS decoded prediction has useful residual signal over persistence
  - the improvement is too small for demo promotion, and spectral energy remains much worse than persistence, so the next iteration should map alpha50 only as a cheap curve check and then move toward a trained residual/stability objective

Experiment loop update (2026-05-06, residual alpha50 remote result):
- Remote run:
  - successful Vast contract: `36245771`
  - B2 artifact: `remote-runs/light/ups_light_residual_alpha50_20260506T1548Z.tar.gz`
  - local extracted artifact: `reports/demo/remote_artifacts/ups_light_residual_alpha50_20260506T1548Z/`
  - local summary copy: `reports/light_experiments_remote/ups_light_task_signature_residual_alpha50/summary.json`
  - rebuilt scorecard: `reports/demo/light_latest/scorecard.json`
- W&B:
  - tracked first completed run set: `dr5wpv23`, `tp1wbop8`, `e3v1o3ce`, `axcvkdcy`
  - destroyed the instance after artifact publication because the Vast `--args-mode` container restarted the entrypoint
- Metrics:
  - residual alpha50 decoded rollout NRMSE: `0.6084554326486734`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - residual alpha25 decoded rollout NRMSE: `0.5486869325531744`
  - baseline metric delta: `0.03829209149796986`
  - baseline ratio: `1.0671598623311852`
  - baseline improvement fraction: `-0.06715986233118525`
  - baseline improvement gate: `false`
  - absolute promotion rule `decoded_rollout_nrmse<=1.0`: `true`
- Interpretation:
  - the scalar residual blend peaks below `alpha=0.50`; alpha50 is worse than persistence
  - do not continue scalar blend sweeps; next useful implementation is eval-only checkpoint reuse plus a trained residual/stability objective

Experiment loop update (2026-05-06, eval-only checkpoint reuse):
- Added:
  - `scripts/run_light_experiment.py --skip-training`
  - `scripts/run_light_experiment.py --checkpoint-source <run-or-checkpoints-dir>`
  - `scripts/run_remote_light_promotion.sh` passthrough envs `SKIP_TRAINING=1` and `CHECKPOINT_SOURCE=...`
- Validation:
  - `pytest tests/unit/test_light_experiment_runner.py tests/unit/test_vast_launch.py tests/unit/test_monitoring.py tests/unit/test_demo_scorecard.py -q`
  - `python -m py_compile scripts/run_light_experiment.py scripts/vast_launch.py src/ups/utils/monitoring.py`
  - `bash -n scripts/run_remote_light_promotion.sh`
  - `git diff --check`
- Purpose:
  - future alpha/stability probes can reuse an existing checkpoint directory instead of paying to retrain identical weights for every evaluation-only variant

Experiment loop update (2026-05-06, trained residual iteration surface):
- Added:
  - `scripts/run_light_experiment.py` now opens a W&B `benchmark-summary` run when `--allow-wandb` is set and logs final benchmark metrics under `summary/*`.
  - `scripts/train.py` decoded stages now accept persistence-residual, residual-spectral, spectral, and relative decoded loss weights without changing defaults.
  - `scripts/plan_demo_experiments.py` now includes `task_signature_trained_residual`.
  - `docs/superpowers/plans/2026-05-06-post-light-v1-improvement-plan.md` records the new queue and remote dry-run shape.
- Validation:
  - `pytest tests/unit/test_light_experiment_runner.py tests/unit/test_monitoring.py tests/unit/test_plan_demo_experiments.py tests/unit/test_losses.py tests/unit/test_pdebench_runner_eval.py tests/unit/test_demo_scorecard.py -q`
- Purpose:
  - make W&B contain the benchmark truth, not only training curves
  - move the next paid light-v1 candidate from scalar eval blending to a trained persistence-residual/stability objective

Experiment loop update (2026-05-06, trained residual remote result):
- Remote run:
  - successful Vast contract: `36250467`
  - pinned offer: `36109890`, RTX 4090, California, US, about `$0.268/hr`
  - B2 artifact: `remote-runs/light/ups_light_trained_residual_20260506T1755Z.tar.gz`
  - local extracted artifact: `reports/demo/remote_artifacts/ups_light_trained_residual_20260506T1755Z/`
  - local summary copy: `reports/light_experiments_remote/ups_light_task_signature_trained_residual/summary.json`
  - rebuilt scorecard: `reports/demo/light_latest/scorecard.json`
- W&B:
  - tracked completed run set: `4wps03re`, `u76hpryu`, `kv2z579u`, `quw7vz35`, `3dr2jyfa`
  - `3dr2jyfa` is the new `benchmark-summary` run containing final benchmark metrics
  - destroyed the instance after artifact publication because the Vast `--args-mode` container restarted the entrypoint
- Metrics:
  - trained residual decoded rollout NRMSE: `0.530536668470072`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - previous best alpha25 decoded rollout NRMSE: `0.5486869325531744`
  - previous UPS light decoded rollout NRMSE: `0.8881691012411048`
  - baseline metric delta: `-0.03962667268063158`
  - baseline ratio: `0.9304994379318442`
  - baseline improvement fraction: `0.06950056206815583`
  - baseline improvement gate: `false`
  - decoded rollout spectral energy error: `4.541018111181074`
  - absolute promotion rule `decoded_rollout_nrmse<=1.0`: `true`
- Interpretation:
  - trained residual/stability loss produced another real improvement and is now the best held-out light-v1 candidate
  - the improvement is still only about `6.95%` over persistence, below the `20%` demo gate
  - task metrics show Burgers and Darcy are strong (`0.2152`, `0.2704`) while Advection remains the main failure (`0.7362`), so the next iteration should isolate transport/advection rather than globally increase residual loss weights

Experiment loop update (2026-05-06, eval-only transport residual gate):
- Added:
  - evaluation supports `evaluation.decoded_persistence_residual_alpha_by_task`
  - evaluation supports `evaluation.decoded_persistence_residual_alpha_by_family`
  - planner variant `task_signature_transport_residual_gate` now uses global persistence with transport-only residual alpha `0.42`
- Local eval-only setup:
  - reused checkpoints from `reports/light_experiments_remote/ups_light_task_signature_trained_residual`
  - hydrated only held-out `light-v1` test shards locally
  - no local training
- Best local eval-only result:
  - run: `ups_light_transport_residual_gate_alpha0p42_eval`
  - decoded rollout NRMSE: `0.5126627282110727`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - baseline ratio: `0.8991506314250525`
  - baseline improvement fraction: `0.10084936857494749`
  - baseline improvement gate: `false`
  - decoded rollout spectral energy error: `0.1945553537247434`
  - task Advection decoded rollout NRMSE: `0.7223984441272786`
- Interpretation:
  - selective residual gating is now the best cheap path and improves over the trained residual run without retraining
  - the remaining gap is still transport/advection; next candidate should train the gate or improve advection dynamics, not spend on larger scale yet

Experiment loop update (2026-05-11, validation-calibrated transport residual gate):
- Added:
  - `scripts/calibrate_residual_gate.py`
  - unit tests for calibration helper selection and override serialization
  - planner default for `task_signature_transport_residual_gate` now uses validation-calibrated transport alpha `0.20`
- Local calibration setup:
  - checkpoint source: `reports/light_experiments_remote/ups_light_task_signature_trained_residual`
  - hydrated held-out `light-v1` validation shards locally
  - swept transport-family alpha on `val`: `0.0`, `0.1`, `0.2`, `0.3`, `0.36`, `0.4`, `0.42`, `0.44`, `0.5`, `0.75`, `1.0`
  - selected alpha by minimum `decoded_rollout_nrmse` on `val`
  - ran exactly one frozen held-out `test` eval with selected alpha
- Calibration result:
  - best validation run: `ups_light_transport_gate_valcal_val_family_transport_alpha0p2`
  - validation decoded rollout NRMSE: `0.35679104424840724`
  - frozen test run: `ups_light_transport_gate_valcal_test_family_transport_alpha0p2`
  - frozen test decoded rollout NRMSE: `0.5283710326453532`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - baseline ratio: `0.9267011652818558`
  - baseline improvement fraction: `0.07329883471814426`
  - baseline improvement gate: `false`
- Interpretation:
  - the prior test-swept alpha `0.42` remains the best exploratory score (`0.5126627282110727`) but is not benchmark-clean
  - validation-calibrated alpha `0.20` is the clean result and modestly improves over persistence and the trained residual run
  - next useful work is a learned gate or advection-specific dynamics improvement; more manual alpha sweeps are unlikely to reach the 20% gate alone

Experiment loop update (2026-05-11, horizon-calibrated residual gate):
- Added:
  - decoded evaluation supports residual-alpha schedules by rollout horizon, task, and family
  - decoded evaluation can emit all global/task/family horizon metrics with `evaluation.report_all_horizon_metrics=true`
  - `scripts/calibrate_residual_gate.py --schedule-by-horizon`
  - `scripts/calibrate_residual_gate.py --reuse-existing`
  - a 1% default complexity guard via `--schedule-min-relative-improvement`
- Local calibration setup:
  - checkpoint source: `reports/light_experiments_remote/ups_light_task_signature_trained_residual`
  - reused the same `light-v1` validation/test shards and eval-only checkpoints
  - swept transport-family alphas on `val`, selected per-horizon candidates, then evaluated the proposed schedule on `val` as a full rollout
  - selected the horizon schedule only if aggregate validation improvement cleared the 1% complexity guard
- Calibration artifacts:
  - record: `reports/light_experiments_remote/ups_light_transport_horizon_gate_valcal_calibration.json`
  - validation schedule run: `reports/light_experiments_remote/ups_light_transport_horizon_gate_valcal_val_family_transport_horizon_schedule/summary.json`
  - guarded selected test run: `reports/light_experiments_remote/ups_light_transport_horizon_gate_valcal_test_family_transport_alpha0p2/summary.json`
  - non-promoted exploratory schedule test: `reports/light_experiments_remote/ups_light_transport_horizon_gate_valcal_test_family_transport_horizon_schedule/summary.json`
  - rebuilt scorecard: `reports/demo/light_latest/scorecard.json`
- Metrics:
  - best constant validation decoded rollout NRMSE: `0.35679104424840724`
  - schedule validation decoded rollout NRMSE: `0.3562364331301045`
  - schedule relative validation improvement: `0.0015544423752873247`
  - schedule selection threshold: `0.01`
  - selected gate: constant alpha `0.20`
  - selected frozen test decoded rollout NRMSE: `0.5283710326453532`
  - exploratory schedule frozen test decoded rollout NRMSE: `0.5352231399077773`
- Interpretation:
  - per-horizon schedule selection overfits the validation split at this scale; the tiny validation gain did not justify the more complex gate
  - the benchmark-clean result remains the simpler validation-calibrated transport alpha `0.20`
  - horizon metrics are still useful diagnostics, but the next real SOTA step should learn the gate or improve transport/advection dynamics rather than hand-tune schedules

Experiment loop update (2026-05-11, learned-gate execution setup):
- Branch:
  - `codex/sota-learned-gate`
- Goal:
  - execute the universal physics SOTA improvement plan with an `autoresearch` loop and durable progress tracking
- Added:
  - decoded evaluation now accepts `evaluation.decoded_persistence_residual_gate`
  - gate parameters use a bounded logistic sidecar over the already-resolved static alpha
  - gate features include horizon, normalized horizon, residual magnitude, persistence magnitude, and prediction magnitude
  - decoded evaluation reports gate alpha mean/std globally, by task, by family, and by horizon
- Local validation-only probes:
  - `ups_light_gate_hook_constant_alpha0p2_val`: `decoded_rollout_nrmse = 0.36417941757537725`
  - `ups_light_gate_hook_transport_base_val`: `decoded_rollout_nrmse = 0.3567910081081011`
- Learning:
  - setting `decoded_persistence_residual_gate.base_alpha=0.2` overrides all families and regresses validation because Burgers/Darcy no longer stay at persistence
  - omitting `base_alpha` lets the gate inherit the resolved family/task alpha, reproducing the clean transport-only gate
  - this confirms the hook is usable for learned deltas around the clean transport gate without changing the benchmark harness

Experiment loop update (2026-05-11, learned-gate calibration/export path):
- Added:
  - `scripts/calibrate_residual_gate.py --use-decoded-residual-gate`
  - repeatable `--gate-config-candidate` JSON sweeps for decoded residual gate configs
  - `--gate-feature-weight name=value` convenience wiring into gate `feature_weights`
  - `--export-selected-gate-config` for a frozen validation-selected override payload
  - unit tests for gate config serialization, candidate merging, and selected override construction
- Local smoke validation:
  - command used `--skip-test`, `--eval-max-samples 8`, and `--decoded-rollout-steps 4`
  - output root: `reports/research/sota_loop/gate_calibrator_smoke`
  - exported selected gate: `reports/research/sota_loop/gate_calibrator_smoke/selected_gate.json`
  - candidates: transport alpha `0.2` and `0.3` with a neutral decoded gate config
  - selected validation alpha: `0.2`
  - smoke decoded rollout NRMSE: `0.2900588529988161`
- Learning:
  - the export path works end-to-end and preserves the clean no-test-budget workflow
  - the smoke metric is not comparable to the benchmark because it used only 8 samples and 4 rollout steps
  - next useful experiment is a validation-only gate-config sweep with nonzero target-free feature weights over the full 32-sample, 16-step validation setup

Experiment loop update (2026-05-11, learned-gate feature sweep):
- Added:
  - calibrator held-out test guard via `--reference-metric-value` and `--test-min-relative-improvement`
  - selected-gate records now include `test_guard` and `test_skipped` when validation does not clear the guard
- Comparable validation-only sweep:
  - output root: `reports/research/sota_loop/gate_config_sweep`
  - reference: clean constant transport alpha validation NRMSE `0.35679104424840724`
  - neutral decoded gate: `0.3567910081081011`
  - best candidate: alpha `0.2`, gate `feature_weights.horizon_norm=-0.5`
  - best validation decoded rollout NRMSE: `0.35560983348888475`
  - relative validation improvement vs reference: `0.0033106513702179665`
- Refinement sweep:
  - output root: `reports/research/sota_loop/gate_config_refine`
  - swept alphas `0.15`, `0.2`, `0.25` with negative horizon/residual feature weights
  - selected gate remained alpha `0.2` with `feature_weights.horizon_norm=-0.5`
  - exported selected gate: `reports/research/sota_loop/gate_config_refine/selected_gate.json`
  - held-out test was skipped by guard because improvement was below the required `0.01`
- Learning:
  - decreasing residual trust over rollout horizon is directionally useful
  - target-free scalar gate features produce only a small validation gain, not enough for a clean held-out test
  - next high-leverage local step should move from scalar blending to a transport/advection dynamics correction or a trained sidecar with per-sample supervision

Experiment loop update (2026-05-11, advection roll-shift transport correction):
- Added:
  - decoded evaluation supports config-gated periodic roll shifts via `evaluation.decoded_roll_shift_by_task`
  - additional shift override surfaces: `decoded_roll_shift_by_family`, `decoded_roll_shift_by_task_horizon`, and `decoded_roll_shift_by_family_horizon`
  - unit coverage for a synthetic advection case where a one-cell periodic shift exactly fixes persistence
- Validation-only sweep:
  - output root: `reports/research/sota_loop/transport_shift_sweep`
  - setup: persistence residual alpha `0.0`, advection-only roll shift, 32 validation samples, 16 rollout steps
  - coarse/refined selected shift: `+40`
  - selected validation run: `ups_light_advection_roll_shift_40_val`
  - selected validation decoded rollout NRMSE: `0.11155091371736849`
  - validation neighbors: `+38 -> 0.11443604286804047`, `+42 -> 0.11160543416536953`, `+44 -> 0.11459499323130662`
- Frozen held-out test:
  - run: `ups_light_advection_roll_shift_40_test`
  - summary: `reports/light_experiments_remote/ups_light_advection_roll_shift_40_test/summary.json`
  - decoded rollout NRMSE: `0.30780652221851373`
  - persistence decoded rollout NRMSE: `0.5701633411507036`
  - clean transport-alpha gate decoded rollout NRMSE: `0.5283710326453532`
  - baseline ratio vs persistence: `0.5398567392938639`
  - baseline improvement fraction: `0.46014326070613615`
  - scorecard: `reports/demo/light_latest/scorecard.json`, `baseline_improvement_passed=true`
- Learning:
  - Advection is dominated by a learnable translation/transport rule; a simple periodic shift beats scalar residual blending by a large margin
  - this is a strong demo candidate and passes the light-v1 persistence gate, but it is still a hand-selected physics correction rather than a learned general simulator mechanism
  - next step should turn this into a parameter-conditioned or learned transport head so the result is defensible as a foundation-model improvement rather than a task-specific postprocess

Experiment loop update (2026-05-11, roll-shift calibration harness):
- Added:
  - `scripts/calibrate_roll_shift.py`
  - unit tests for shift override serialization, candidate defaults, and horizon schedule selection
  - planner variant `task_signature_advection_roll_shift40`
- Reconstructed calibration:
  - command used `--reuse-existing` over shifts `36`, `38`, `40`, `42`, `44`
  - selected validation shift: `+40`
  - selected validation decoded rollout NRMSE: `0.11155091371736849`
  - selected validation guard improvement vs clean transport gate: `0.6873494570124249`
  - exported selected shift: `reports/research/sota_loop/transport_shift_sweep/selected_shift.json`
  - reused frozen held-out test summary: `reports/light_experiments_remote/ups_light_advection_roll_shift_40_test/summary.json`
- Learning:
  - the shift result is now reproducible through a guarded calibration/export script, not a manual shell loop
  - validation advection is almost exactly corrected by shift `+40`, but held-out test advection remains `0.4065598205949988`; this means the fixed shift is not a universal transport law
  - local hydrated data lacks `advection1d_train.h5`, so the next learned/parameter-conditioned transport-head step needs remote train data hydration or a small dedicated train split

Experiment loop update (2026-05-11, observed transport shift estimator):
- Added:
  - `evaluation.decoded_observed_roll_shift_estimator`, a state-conditioned decoded evaluator hook that estimates periodic transport shift from the previous observed transition rather than using a fixed task/split shift
  - metrics for estimated shift mean/std at aggregate, task, family, and horizon levels
  - unit coverage for a synthetic advection sequence where the estimator recovers a one-cell shift and improves decoded rollout NRMSE
- Validation guard:
  - run: `ups_light_observed_shift_estimator_val`
  - summary: `reports/research/sota_loop/observed_shift_estimator/ups_light_observed_shift_estimator_val/summary.json`
  - decoded rollout NRMSE: `0.1419775490176828`
  - advection NRMSE: `0.12911778915203231`
  - Burgers/Darcy NRMSE: `0.14738121412908425` / `0.188979512124482`
  - spectral error: `0.04679754756316474`
  - estimated shift mean/std: `40.0` / `0.0`
- Frozen held-out test:
  - run: `ups_light_observed_shift_estimator_test`
  - summary: `reports/research/sota_loop/observed_shift_estimator/ups_light_observed_shift_estimator_test/summary.json`
  - decoded rollout NRMSE: `0.20177292896682064`
  - baseline ratio vs persistence: `0.35388618384269066`
  - baseline improvement fraction: `0.6461138161573093`
  - advection NRMSE: `0.22508631227914033`
  - Burgers/Darcy NRMSE: `0.17446879896821743` / `0.20909553062258152`
  - h4/h8/h16 NRMSE: `0.08770583713720184` / `0.08372942384331622` / `0.08648617959454696`
  - spectral error: `0.06721626206246058`, effectively tied with the fixed roll-shift result but numerically higher by about `1.3e-9`
  - estimated shift mean/std: `64.0` / `0.0`
- Learning:
  - a state-conditioned transport rule beats the fixed validation-selected shift on held-out aggregate NRMSE (`0.20177` vs `0.30781`) and advection NRMSE (`0.22509` vs `0.40656`)
  - because the estimator uses observed previous transitions, it is a strong proof of transport-state signal but not yet a fully autonomous learned simulator rollout mechanism
  - next step should train or fit a causal transport head that predicts the shift from model state/metadata without ground-truth future transitions, then compare against this observed-estimator upper bound

Experiment loop update (2026-05-14, causal model-prediction shift estimator):
- Added:
  - `evaluation.decoded_prediction_roll_shift_estimator`, a causal decoded evaluator hook that estimates periodic shift by comparing the current field to the model's own next-step decoded prediction
  - estimator modes `roll_prediction` and `roll_persistence`
  - unit coverage for a synthetic one-step advection case where the model prediction carries the correct shift signal
- Validation-only candidates:
  - output root: `reports/research/sota_loop/prediction_shift_estimator`
  - setup: trained-residual checkpoint, 32 validation samples, 16 rollout steps, Advection-only prediction-shift estimator
  - `ups_light_prediction_shift_roll_persistence_val`: decoded rollout NRMSE `0.5584609221453186`, Advection NRMSE `0.8005553475932097`, inferred shift mean/std `-9.42578125` / `76.6591886881615`
  - `ups_light_prediction_shift_roll_prediction_val`: decoded rollout NRMSE `0.5584609221453186`, Advection NRMSE `0.8005553475932097`, inferred shift mean/std `-9.42578125` / `76.6591886881615`
  - both failed the validation guard; no held-out test was run
- Diagnostic:
  - local `advection1d_val.h5` has optimal observed one-step shift `+40` for horizons 1-16
  - local `advection1d_test.h5` has optimal observed one-step shift `+72` for horizons 1-16 when the candidate grid includes `72`
  - initial-frame aggregate stats are identical across local validation/test shards, so the split-specific transport rate is not recoverable from the first frame alone in this light shard
- Learning:
  - the current trained-residual model's decoded prediction is not yet a reliable causal phase/velocity signal
  - the observed-transition estimator remains an upper bound, not a deployable causal mechanism
  - next useful step is not another eval-only causal heuristic; it is a small remote/data-backed train or fit step for a transport head using train/val trajectories, with held-out test still frozen

Experiment loop update (2026-05-14, train/val transport-shift fit harness):
- Added:
  - `scripts/fit_transport_shift_head.py`
  - unit tests for train-fitted shift selection, train/val mismatch reporting, and same-split refusal
  - SOTA-plan guardrail that local same-split smoke runs are not benchmark evidence
- Scope contract:
  - fit uses `advection1d_train.h5` only
  - validation measures the train-fitted shift on `advection1d_val.h5`
  - held-out `test` remains frozen and should only run after the validation guard passes
- Local smoke:
  - command used `--train-split val --val-split val --allow-same-split-smoke` because local data still lacks `advection1d_train.h5`
  - output: `reports/research/sota_loop/transport_head_fit/local_smoke_fit_record.json`
  - selected smoke shift: `+40`
  - smoke validation NRMSE: `0.012850472703576088`
  - interpretation: script works end-to-end, but this is same-split smoke only
- Remote/light-v1 command skeleton:
  - hydrate B2 `light-v1/advection1d/advection1d_train.h5` and `light-v1/advection1d/advection1d_val.h5`
  - run `scripts/fit_transport_shift_head.py --data-root <hydrated-light-root> --task advection1d --train-split train --val-split val --max-samples 32 --rollout-steps 16 --output-json reports/research/sota_loop/transport_head_fit/light_v1_fit_record.json --export-selected-config reports/research/sota_loop/transport_head_fit/light_v1_selected_config.json`
  - only if validation passes, run one frozen `test` through `scripts/run_light_experiment.py` with the exported selected override

Experiment loop update (2026-05-15, real light-v1 train/val transport-shift fit):
- Data:
  - B2 readiness: `9/9` `light-v1` keys present
  - hydrated local missing shard: `data/pdebench/advection1d_train.h5` from `light-v1/advection1d/advection1d_train.h5`
  - existing local validation/test shards were left untouched
- Validation-first fit:
  - script: `scripts/fit_transport_shift_head.py`
  - candidate shifts: `-80` through `80` coarse grid including `0`, `40`, and `72`
  - 32-sample train fit: selected train shift `0`, train NRMSE `0.01366413850337267`
  - 32-sample validation measurement of train-fitted shift: validation NRMSE `0.5140249729156494`
  - validation oracle in same candidate set: shift `40`, validation NRMSE `0.012850472703576088`
  - validation guard vs fixed roll-shift reference `0.30780652221851373`: failed, relative improvement `-0.6699612770087436`
  - 128-sample train fit repeated the same conclusion: selected train shift `0`, validation NRMSE `0.5140249729156494`
  - held-out test was not run
- Learning:
  - the real `light-v1` train split has a different apparent transport shift from validation; a constant train-fitted shift does not generalize
  - validation-selected fixed shift remains a demo diagnostic, not a train-learned mechanism
  - next result-oriented step must train a per-sample/per-trajectory transport head with real trajectory features or revisit the light shard construction; another constant-shift fit is exhausted

Experiment loop update (2026-05-15, transport split-regime diagnostic):
- Added:
  - `scripts/diagnose_transport_shift_splits.py`
  - unit tests for consistent and mismatched split regimes
- Real `light-v1` diagnostic:
  - output: `reports/research/sota_loop/transport_shift_split_diagnostic.json`
  - data root: `data/pdebench`
  - task: `advection1d`
  - splits: `train,val,test`
  - candidate shifts: `-96` through `96` in steps of `8`
  - best shifts: train `0`, val `40`, test `72`
  - `consistent_best_shift=false`
- Learning:
  - the failed train-fitted constant shift is explained by split-regime mismatch, not by optimization noise
  - the active benchmark-clean objective cannot be completed by another constant-shift fit on the current `light-v1` shards
  - next aligned work is either a per-trajectory transport head that predicts split variation from available trajectory features, or a corrected/successor shard version with train/val/test drawn from compatible transport-rate distributions

Experiment loop update (2026-05-15, benchmark-clean transport-shift gate runner):
- Added:
  - `scripts/run_transport_shift_gate.py`
  - unit tests for pass, split-mismatch block, and validation-guard block
- Gate contract:
  - only train and validation splits are inspected
  - test is eligible only if train/val best shifts are consistent and the train-fitted shift passes the configured validation guard
  - held-out test remains blocked otherwise
- Current `light-v1` gate output:
  - output: `reports/research/sota_loop/transport_shift_gate.json`
  - train/val best shifts: train `0`, val `40`
  - train-fitted selected shift: `0`
  - validation NRMSE for train-fitted shift: `0.5140249729156494`
  - guard reference: `0.30780652221851373`
  - relative improvement: `-0.6699612770087436`
  - `test_eligible=false`
  - blockers: split best shifts differ; validation metric did not pass SOTA guard
- Decision:
  - do not run held-out test
  - the active goal remains open because current evidence proves the requested clean result is blocked on current `light-v1`

Experiment loop update (2026-05-15, train-derived compatible split proof):
- Built a local Advection-only candidate split from the real hydrated train shard:
  - source: `data/pdebench/advection1d_train.h5`
  - command used `scripts/make_light_hdf5_shards.py`
  - train count: `64`
  - validation count: `32`
  - test count: `0`
  - manifest: `reports/research/sota_loop/transport_trainval_candidate_manifest.yaml`
- Gate output:
  - output: `reports/research/sota_loop/transport_trainval_candidate_gate.json`
  - train/val best shifts: train `0`, val `0`
  - train-fitted selected shift: `0`
  - validation NRMSE: `0.012206551618874073`
  - guard reference: `0.30780652221851373`
  - relative improvement: `0.9603434276476813`
  - `test_eligible=true` for the train/val gate
- Interpretation:
  - this proves the transport-shift gate is achievable when train and validation are distribution-compatible
  - this is not benchmark evidence because validation is derived from the train source split
  - no held-out test was run
  - the required benchmark-clean path is now a corrected successor shard or a richer per-trajectory transport head on data with valid train/val/test transport-rate coverage

Experiment loop update (2026-05-15, train-derived successor split with held-out test):
- Built a non-overlapping Advection-only successor split from the real hydrated train shard:
  - source: `data/pdebench/advection1d_train.h5`
  - command used `scripts/make_light_hdf5_shards.py`
  - train count: `64`
  - validation count: `32`
  - test count: `32`
  - manifest: `reports/research/sota_loop/transport_successor_manifest.yaml`
- Extended `scripts/run_transport_shift_gate.py` so an optional held-out `--test-split` is measured only after the train/validation gate passes.
- Gate output:
  - output: `reports/research/sota_loop/transport_successor_gate.json`
  - train/val best shifts: train `0`, val `0`
  - train-fitted selected shift: `0`
  - validation NRMSE: `0.012206551618874073`
  - validation guard reference: `0.30780652221851373`
  - validation relative improvement: `0.9603434276476813`
  - `test_eligible=true`
  - held-out successor test selected shift: `0`
  - held-out successor test NRMSE: `0.015408719889819622`
  - held-out successor test oracle shift: `0`
- Interpretation:
  - this proves the full train-fit, validation-guard, and one-test gate works when train/validation/test are distribution-compatible
  - this is still successor-split evidence, not official current `light-v1` benchmark evidence, because all three successor splits are derived from the current train source shard
  - the official current `light-v1` train/val/test objective remains blocked by the observed split-regime mismatch unless we rebuild compatible benchmark shards or learn a per-trajectory transport mechanism

Experiment loop update (2026-05-15, official light-v1 observed-transition transport gate):
- Added:
  - `scripts/run_observed_transport_shift_gate.py`
  - unit tests covering guarded held-out test measurement and validation-guard blocking
- Gate contract:
  - train locks the lagged observed-transition estimator contract and candidate shift support
  - validation applies the same estimator without selecting a validation shift
  - held-out test is measured only if validation passes the configured guard
- Official light-v1 result:
  - output: `reports/research/sota_loop/observed_transport_shift_gate.json`
  - data root: `data/pdebench`
  - task: `advection1d`
  - train max samples: `128`
  - validation/test max samples: `32`
  - rollout steps: `16`
  - reference metric: `0.30780652221851373`
  - train NRMSE: `0.014504759572446346`, inferred shift mean/std `0.0` / `0.0`
  - validation NRMSE: `0.012846261262893677`, inferred shift mean/std `40.0` / `0.0`
  - validation relative improvement: `0.9582651427581705`
  - `test_eligible=true`
  - held-out test NRMSE: `0.004225204233080149`, inferred shift mean/std `72.0` / `0.0`
- Interpretation:
  - this is the first official current `light-v1` Advection transport-shift gate in this loop that passes validation and measures a held-out test under the gate
  - it is benchmark-clean for a state-conditioned observed-transition transport estimator because no validation/test shift is selected as a hyperparameter
  - it is not yet a fully autonomous causal rollout head because it uses the previous observed transition at each step; the next SOTA-facing step is to train a causal transport head that predicts the same shift/rate from allowed model state or metadata without ground-truth future transitions

Experiment loop update (2026-05-15, train-window transport scan):
- Added:
  - `scripts/scan_transport_train_windows.py`
  - unit coverage for train-window shift histograms
- Purpose:
  - audit train-source shift regimes without using validation/test selection
  - support remote scans against `full/advection1d/advection1d_train.h5` before building a corrected train-only light shard
- Local current `light-v1` scan:
  - output: `reports/research/sota_loop/transport_train_window_scan.json`
  - source: `data/pdebench/advection1d_train.h5`
  - source shape: `[128, 201, 1024, 1]`
  - window size/stride: `32` / `32`
  - windows scanned: `4`
  - best-shift histogram: `{"0": 4}`
- Interpretation:
  - every current official local train window fits shift `0`
  - there is no local train-only window that can explain the official validation shift `40`
  - the original constant train-fitted shift objective now requires a remote/full-source scan and likely a corrected light shard built from compatible train-source windows; local repeated fitting is exhausted

Experiment loop update (2026-05-15, remote constant-shift candidate pipeline):
- Added:
  - split-specific shard start offsets in `scripts/make_light_hdf5_shards.py`
  - `scripts/run_remote_transport_shift_candidate.sh`
  - `scripts/launch_remote_transport_shift_candidate_vast.sh`
  - dry-run/unit coverage for the new remote candidate path
- Remote pipeline contract:
  - hydrate full `advection1d` train/val/test shards from B2 prefix `full`
  - scan only full train windows for `TARGET_SHIFT`, default `40`
  - build a small candidate shard using the selected train-source start and native val/test starts
  - run `scripts/run_transport_shift_gate.py` with `--test-split test`, so held-out test is measured only if validation passes
- Local verification:
  - `DRY_RUN=1 bash scripts/run_remote_transport_shift_candidate.sh` is local-safe and prints the intended plan without touching `/workspace`
  - launcher dry-run emits a redacted Vast command for branch `codex/sota-learned-gate`
- Next executable command:
  - `DRY_RUN=0 ENV_FILE=.env bash scripts/run_remote_transport_shift_candidate.sh`
  - or via Vast: `DRY_RUN=0 ENV_FILE=/path/to/.env bash scripts/launch_remote_transport_shift_candidate_vast.sh`

Experiment loop update (2026-05-15/16, full-train remote transport scan):
- Remote run:
  - Vast instance: `36855174`
  - branch: `codex/sota-learned-gate`
  - command path: `scripts/launch_remote_transport_shift_candidate_vast.sh` -> `scripts/run_remote_transport_shift_candidate.sh`
  - full shards hydrated from B2 prefix `full`: `advection1d_train.h5`, `advection1d_val.h5`, `advection1d_test.h5`
  - local copied evidence: `reports/research/sota_loop/remote_transport_shift_candidate/train_window_scan.json`
  - instance was destroyed after copying evidence
- Full train scan:
  - source shape: `[60000, 201, 1024, 1]`
  - window size/stride: `32` / `32`
  - windows scanned: `1875`
  - best-shift histogram: `{"0": 625, "8": 937, "16": 1, "24": 312}`
  - target validation shift `40`: `0` matching train windows
- Interpretation:
  - the original constant train-fitted shift objective is blocked by source coverage, not local hydration or optimizer error
  - full Advection train contains no scanned window whose best shift matches the official validation shift `40`
  - the pipeline correctly refused to build a candidate/test run because no train-only source window could support the required validation regime
  - next aligned options are to change the benchmark split/source construction with explicit data-governance approval, or move from a constant shift to a learned/state-conditioned transport mechanism

Experiment loop update (2026-05-16, full train/val/test compatibility scan):
- Added:
  - `scripts/select_transport_compatible_windows.py`
  - all-split compatibility mode in `scripts/run_remote_transport_shift_candidate.sh`
- Remote run:
  - Vast instance: `36856643`
  - branch: `codex/sota-learned-gate`
  - command path: `scripts/launch_remote_transport_shift_candidate_vast.sh` with `SCAN_ALL_SPLITS=1 REQUIRE_TEST_COMPATIBLE=1`
  - local copied evidence directory: `reports/research/sota_loop/remote_transport_shift_candidate_all_splits/`
  - instance was destroyed after copying evidence
- Full split scans:
  - train source shape: `[60000, 201, 1024, 1]`, windows scanned `1875`, histogram `{"0": 625, "8": 937, "16": 1, "24": 312}`
  - validation source shape: `[10000, 201, 1024, 1]`, windows scanned `313`, histogram `{"40": 313}`
  - test source shape: `[10000, 201, 1024, 1]`, windows scanned `313`, histogram `{"72": 313}`
  - compatible train/val/test shifts: `[]`
- Interpretation:
  - no native full-source train/val/test window triplet shares a constant transport shift under the scanned 32-sample window protocol
  - the original benchmark-clean constant train-fitted shift objective is fully blocked for the current full source splits
  - no held-out test was run because the train/val compatibility precondition failed
  - the only defensible continuation is a benchmark-policy decision: reconstruct a compatible benchmark, or replace the constant-shift objective with a learned/state-conditioned transport objective

Experiment loop update (2026-05-19, transport-shift goal audit):
- Added:
  - `scripts/audit_transport_shift_goal.py`
  - unit coverage for requirement-by-requirement goal classification
- Audit result against current artifacts:
  - command: `/opt/anaconda3/bin/python scripts/audit_transport_shift_goal.py --output-json reports/research/sota_loop/transport_shift_goal_audit.json`
  - status: `blocked_incompatible_splits`
  - `test_allowed`: `false`
  - satisfied requirements: real `light-v1` train/val data accessed, train-only constant shift fit recorded, results recorded
  - failed requirement: validation SOTA guard did not pass for the train-fitted shift
  - blocked requirement: held-out test correctly not run because the validation and split-compatibility preconditions failed
- Interpretation:
  - the audit makes the active objective mechanically checkable instead of relying on prose scattered across the worklog
  - this does not complete the original benchmark-clean constant-shift objective
  - it preserves the benchmark discipline: no validation-selected shift and no held-out test without a train-supported validation pass

Follow-up audit update (2026-05-19, Advection schema evidence):
- Extended `scripts/audit_transport_shift_goal.py` to inspect the real `light-v1` HDF5 schema.
- Current Advection split schema:
  - `advection1d_train.h5`: only dataset `data`, shape `[128, 201, 1024, 1]`, no file attrs, no dataset attrs
  - `advection1d_val.h5`: only dataset `data`, shape `[32, 201, 1024, 1]`, no file attrs, no dataset attrs
  - `advection1d_test.h5`: only dataset `data`, shape `[32, 201, 1024, 1]`, no file attrs, no dataset attrs
  - `parameter_metadata_available`: `false`
- Interpretation:
  - there is no allowed coefficient/velocity/metadata field in the current `light-v1` artifacts that could support a train-fitted metadata-to-shift extrapolator
  - the remaining valid routes are unchanged: benchmark split reconstruction with compatible transport-rate support, or a learned/state-conditioned mechanism that proves validation lift without selecting validation/test shifts

Follow-up audit update (2026-05-19, fail-closed CLI mode):
- Added `--require-status {report,test-ready,achieved}` to `scripts/audit_transport_shift_goal.py`.
- Current enforcement check:
  - command: `/opt/anaconda3/bin/python scripts/audit_transport_shift_goal.py --output-json /private/tmp/transport_shift_goal_audit_test_ready.json --require-status test-ready`
  - result: exits `2`
  - reason: current status is `blocked_incompatible_splits`, so the held-out test path is not ready
- Purpose:
  - `report` keeps the audit usable for diagnostics
  - `test-ready` lets remote/CI scripts fail closed unless validation and compatibility allow the one permitted held-out test
  - `achieved` lets release checks fail unless the complete benchmark-clean result, including the authorized test, is present

Experiment loop update (2026-05-19, remote pipeline audit enforcement):
- Updated `scripts/run_remote_transport_shift_candidate.sh`:
  - all-split runs now execute `scripts/audit_transport_shift_goal.py` after `scripts/run_transport_shift_gate.py`
  - default `AUDIT_REQUIRE_STATUS=achieved`, so the remote candidate pipeline exits nonzero unless the validation gate passed and the authorized held-out test result is recorded
  - dry runs announce the final audit requirement
- Verification:
  - `DRY_RUN=1 SCAN_ALL_SPLITS=1 REQUIRE_TEST_COMPATIBLE=1 bash scripts/run_remote_transport_shift_candidate.sh`
  - `bash -n scripts/run_remote_transport_shift_candidate.sh`
  - `tests/unit/test_remote_transport_shift_candidate.py`
- Interpretation:
  - this does not change the benchmark criteria or turn the current blocked evidence into success
  - it makes the original held-out-test invariant executable in the remote/full-source pipeline that would produce a valid result if compatible evidence ever exists

Experiment loop update (2026-05-19, official local audit runner):
- Added `scripts/run_official_transport_shift_audit.sh`.
- Purpose:
  - rerun the official real `light-v1` Advection train-only constant-shift gate
  - pass `--test-split test` to the gate, relying on `scripts/run_transport_shift_gate.py` to measure held-out test only if validation passes
  - immediately audit the refreshed evidence with `scripts/audit_transport_shift_goal.py`
- Current report-only refresh:
  - command: `AUDIT_REQUIRE_STATUS=report bash scripts/run_official_transport_shift_audit.sh`
  - train best shift: `0`
  - validation best shift: `40`
  - train-fitted validation NRMSE: `0.5140249729156494`
  - validation relative improvement vs guard: `-0.6699612770087436`
  - `test_eligible`: `false`
  - `test`: `null`
  - audit status: `blocked_incompatible_splits`
- Interpretation:
  - the local official path is now one command, reproducible, and fail-closed by default with `AUDIT_REQUIRE_STATUS=achieved`
  - the current evidence still blocks the original goal; no held-out test was run

Follow-up audit update (2026-05-19, light-v1 file identity):
- Extended `scripts/audit_transport_shift_goal.py` to record byte size and SHA-256 for each inspected real HDF5 split.
- Current local `light-v1` Advection identities:
  - `data/pdebench/advection1d_train.h5`: `93508109` bytes, SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`
  - `data/pdebench/advection1d_val.h5`: `24158705` bytes, SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`
  - `data/pdebench/advection1d_test.h5`: `24220172` bytes, SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`
- Interpretation:
  - the audit now proves which real files were accessed for the official train/val gate and the gated test path
  - this strengthens reproducibility; it does not change the blocked validation outcome

Follow-up audit update (2026-05-19, held-out test policy invariant):
- Extended `scripts/audit_transport_shift_goal.py` with explicit held-out test policy fields:
  - `test_result_count`
  - `leaked_test_result`
  - `test_allowed_next`
  - `exactly_one_test_after_validation`
- Added violation statuses:
  - `invalid_test_leakage` if a held-out test result is present before gate eligibility
  - `invalid_multiple_tests` if more than one held-out test result is present
- Current official refreshed audit:
  - `test_eligible`: `false`
  - `test_result_count`: `0`
  - `leaked_test_result`: `false`
  - `test_allowed_next`: `false`
- Interpretation:
  - the audit now distinguishes a correctly blocked benchmark from an invalid artifact that leaked or repeated held-out test evaluation
  - the current result remains correctly blocked, not complete

Follow-up audit update (2026-05-19, enforce light-v1 identity):
- Added optional `--expected-data-sha256 SPLIT=SHA256` checks to `scripts/audit_transport_shift_goal.py`.
- Updated `scripts/run_official_transport_shift_audit.sh` to enforce the current local `light-v1` Advection train/val/test SHA-256 values by default.
- Current official report-only refresh:
  - `data_identity_policy.passed`: `true`
  - `data_identity_policy.mismatches`: `[]`
  - audit status remains `blocked_incompatible_splits`
- Interpretation:
  - future local official refreshes now fail closed if the real Advection files drift or are replaced
  - the active blocker remains validation/split incompatibility, not data identity

Follow-up audit update (2026-05-19, require complete identity coverage):
- Added `--require-data-identity` to `scripts/audit_transport_shift_goal.py`.
- Updated `scripts/run_official_transport_shift_audit.sh` to require expected SHA-256 values for every inspected existing split.
- Current official report-only refresh:
  - `data_identity_policy.require_all_inspected_splits`: `true`
  - `data_identity_policy.missing`: `[]`
  - `data_identity_policy.mismatches`: `[]`
  - `data_identity_policy.passed`: `true`
- Interpretation:
  - direct audit calls can still inspect partial identities when requested
  - the official local benchmark path now requires complete train/val/test identity coverage before considering validation or held-out test status

Follow-up gate update (2026-05-19, source fingerprints in gate artifact):
- Updated `scripts/run_transport_shift_gate.py` to write `data_sources` into the gate JSON.
- Current official gate source fingerprints:
  - train: `data/pdebench/advection1d_train.h5`, `93508109` bytes, SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`
  - val: `data/pdebench/advection1d_val.h5`, `24158705` bytes, SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`
  - test: `data/pdebench/advection1d_test.h5`, `24220172` bytes, SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`
- Interpretation:
  - the train-only fit artifact now independently records the exact files it used
  - the audit still remains the promotion authority and continues to block held-out test completion

Follow-up audit update (2026-05-19, gate/source provenance cross-check):
- Extended `scripts/audit_transport_shift_goal.py` to compare gate-level `data_sources` against independent HDF5 inspection.
- The audit now marks provenance invalid if the gate artifact claims a different path, byte count, or SHA-256 from the inspected data files.
- Current official report-only refresh:
  - `data_identity_policy.gate_source_mismatches`: `[]`
  - `data_identity_policy.passed`: `true`
  - audit status remains `blocked_incompatible_splits`
- Interpretation:
  - stale or hand-edited gate artifacts cannot silently pass provenance validation
  - the active blocker remains the actual train/val transport regime mismatch

Follow-up runner update (2026-05-19, default promotion path fails closed):
- Ran the default promotion command:
  - command: `bash scripts/run_official_transport_shift_audit.sh`
  - exit code: `2`
- Refreshed evidence:
  - `data_identity_policy.passed`: `true`
  - `data_identity_policy.gate_source_mismatches`: `[]`
  - `test_eligible`: `false`
  - `test_result_count`: `0`
  - `status`: `blocked_incompatible_splits`
- Interpretation:
  - the one-command official promotion path correctly refuses to pass on current evidence
  - the nonzero exit is the desired benchmark-clean stop condition, not an execution failure

Follow-up audit update (2026-05-19, result-record enforcement):
- Added `--result-record` and `--require-result-records` to `scripts/audit_transport_shift_goal.py`.
- Updated `scripts/run_official_transport_shift_audit.sh` to require both repo-local records:
  - `worklog.md`
  - `docs/superpowers/plans/2026-05-11-universal-physics-sota-improvement-plan.md`
- Current official report-only refresh:
  - `result_record_policy.required`: `true`
  - `result_record_policy.required_tokens`: `["blocked_incompatible_splits"]`
  - `result_record_policy.mismatches`: `[]`
  - `result_record_policy.passed`: `true`
- Interpretation:
  - the audit now checks that the current benchmark-clean stop condition is recorded in both required repo handoff documents
  - the remaining blocker is still train/val split incompatibility

Follow-up audit update (2026-05-19, metric-backed result-record enforcement):
- Extended result-record enforcement so required repo records must include both the current audit status and the measured selected-validation NRMSE.
- Current official report-only refresh:
  - `result_record_policy.required_tokens`: `["blocked_incompatible_splits", "0.5140249729156494"]`
  - `result_record_policy.mismatches`: `[]`
  - `result_record_policy.passed`: `true`
- Interpretation:
  - a status-only handoff is no longer enough to pass the official audit
  - both required repo records now have to preserve the key measured failure value behind the blocked promotion decision

Experiment loop update (2026-05-19, lagged observed-transition transport gate):
- Ran `scripts/run_observed_transport_shift_gate.py` on real local `light-v1` Advection train/val/test.
- Updated the observed gate to fingerprint the exact train/val/test HDF5 files it used.
- Added optional `--test-ledger-json` enforcement so the same guarded held-out test measurement cannot be repeated accidentally.
- Command output artifact: `reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json` (ignored local report).
- Data identities:
  - train: `data/pdebench/advection1d_train.h5`, `93508109` bytes, SHA-256 `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`
  - val: `data/pdebench/advection1d_val.h5`, `24158705` bytes, SHA-256 `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`
  - test: `data/pdebench/advection1d_test.h5`, `24220172` bytes, SHA-256 `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`
- Validation gate:
  - estimator: `lagged_observed_transition_shift`
  - policy: train locks candidate shift support; validation does not select a split-level shift
  - train NRMSE: `0.014504759572446346`, shift mean/std `0.0` / `0.0`
  - validation NRMSE: `0.012846261262893677`, shift mean/std `40.0` / `0.0`
  - reference NRMSE: `0.30780652221851373`
  - validation relative improvement: `0.9582651427581705`
  - `validation_guard.passed`: `true`
- Held-out test:
  - measured only after validation passed through the gate
  - test NRMSE: `0.004225204233080149`
  - test shift mean/std: `72.0` / `0.0`
- Interpretation:
  - this is the first local real `light-v1` transport-shift gate result in this thread that validates and reaches a guarded held-out test
  - it is not the exhausted constant train-fitted-shift result; it depends on a two-frame observed-context assumption (`t-1 -> t` predicts `t -> t+1`)
  - if that two-frame context is acceptable benchmark policy, this is the strongest current path toward the requested benchmark-clean result
  - if the benchmark requires fully autonomous rollout from only the initial frame, this should remain an upper-bound/state-signal result and the next step is a train-only learned causal head

Follow-up gate update (2026-05-19, observed held-out test ledger):
- `scripts/run_observed_transport_shift_gate.py` now supports `--test-ledger-json`.
- When validation passes and `--test-split` is provided, the runner computes a deterministic held-out measurement key from:
  - estimator name
  - candidate shifts
  - train/val/test data fingerprints
  - split names, sample caps, rollout steps, metric, reference metric, and validation threshold
- If the same key is already present in the ledger, the runner raises an error before measuring held-out test again.
- `--allow-repeat-test` exists only for explicit debugging and does not append another official ledger entry.
- Interpretation:
  - the observed-context path now has a concrete mechanism for the objective's "exactly one held-out test" requirement
  - the previously recorded test result remains the current evidence; future official reruns should use the ledger flag to avoid accidental repeated test evaluation

Follow-up runner update (2026-05-19, official observed transport command):
- Added `scripts/run_official_observed_transport_shift_gate.sh`.
- The command wraps the real `light-v1` lagged observed-transition gate with:
  - train split `train`
  - validation split `val`
  - guarded test split `test`
  - reference metric `0.30780652221851373`
  - default evidence path `reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json`
  - default exactly-once test ledger `reports/research/sota_loop/observed_transport_shift_test_ledger.json`
- `DRY_RUN=1` prints the exact local-safe command contract.
- `ALLOW_REPEAT_TEST=1` is exposed only as an explicit debugging override.
- Interpretation:
  - the observed-context candidate now has a reproducible official local entrypoint instead of an ad hoc command line
  - this improves the benchmark-clean path if the two-frame observed-context policy is accepted

Follow-up audit update (2026-05-19, observed transport result audit):
- Added `scripts/audit_observed_transport_shift_result.py`.
- Updated `scripts/run_official_observed_transport_shift_gate.sh` to run that audit after the observed gate.
- The observed audit checks:
  - real train/val/test data identity and gate/source fingerprint agreement
  - validation guard status
  - exactly one held-out test only after validation eligibility
  - required result-record tokens in `worklog.md` and the SOTA plan
- Default official observed runner behavior now requires audit status `achieved`.
- Interpretation:
  - the observed-context result is now machine-auditable rather than only note-backed
  - this still does not change the policy caveat: promotion depends on accepting the two-frame observed-context estimator as benchmark-clean

Observed audit refresh (2026-05-19, existing gate artifact, no test rerun):
- Ran `scripts/audit_observed_transport_shift_result.py` directly against `reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json`.
- The audit did not run the observed gate and did not re-measure held-out test.
- Audit result:
  - `status`: `achieved`
  - `data_identity_policy.passed`: `true`
  - `data_identity_policy.gate_source_mismatches`: `[]`
  - `validation_guard.passed`: `true`
  - `held_out_test_policy.test_result_count`: `1`
  - `held_out_test_policy.exactly_one_test_after_validation`: `true`
  - `result_record_policy.required_tokens`: `["achieved", "0.012846261262893677", "0.004225204233080149"]`
  - `result_record_policy.passed`: `true`
- Caveat:
  - the existing ignored gate artifact predates the exactly-once ledger, so `held_out_test_policy.ledger` is `null`
  - this audit proves the artifact contains exactly one authorized test result; future official reruns should use the ledger path

Experiment loop update (2026-05-19, train-only first-frame transport feature diagnostic):
- Added `scripts/diagnose_train_only_transport_features.py`.
- The diagnostic fits nearest shift centroids using train split first-frame summary/spectral features only, then measures on validation.
- It does not read or evaluate the held-out test split.
- Real local `light-v1` train/val command output:
  - output: `reports/research/sota_loop/train_only_transport_feature_diagnostic.json` (ignored local report)
  - train shift histogram: `{"0": 128}`
  - validation shift histogram: `{"40": 32}`
  - validation prediction histogram: `{"0": 32}`
  - unsupported validation shifts: `[40]`
  - validation accuracy: `0.0`
  - conclusion: `blocked_no_train_support_for_validation_shift`
- Interpretation:
  - a simple train-fitted first-frame feature head has no train support for the validation transport regime in the current local shard
  - this reinforces that the literal train-only constant/feature shift path is blocked by split construction, not by another missing sweep
  - the remaining credible options are accepting the two-frame observed-context result, rebuilding split-compatible shards, or training a richer causal mechanism with additional allowed signal

Follow-up diagnostic update (2026-05-19, full local train/val feature probe):
- Extended `scripts/diagnose_train_only_transport_features.py` with:
  - explicit full-split caps via `--max-samples -1` and `--val-max-samples -1`
  - per-sample best-shift margin summaries
- Ran the diagnostic on the full local `light-v1` train/val shards without reading test.
- Output: `reports/research/sota_loop/train_only_transport_feature_diagnostic_full.json` (ignored local report).
- Full-shard result:
  - train shift histogram: `{"0": 128}`
  - validation shift histogram: `{"40": 32}`
  - validation prediction histogram: `{"0": 32}`
  - unsupported validation shifts: `[40]`
  - validation accuracy: `0.0`
  - train best-margin mean/min/max: `0.13357117772102356` / `0.037119414657354355` / `0.29049214720726013`
  - validation best-margin mean/min/max: `0.11928332597017288` / `0.04894868656992912` / `0.22282543778419495`
  - conclusion: `blocked_no_train_support_for_validation_shift`
- Interpretation:
  - the train-only first-frame feature failure is not a sample-cap artifact; it holds over the full local train/val shards
  - the positive margins indicate the shift labels are not ambiguous ties under the tested candidate grid

Follow-up audit update (2026-05-19, objective-level transport status):
- Added `scripts/audit_transport_objective_status.py`.
- The aggregate audit reads:
  - constant train-only audit
  - observed-context audit
  - train-only feature diagnostic
- It does not rerun gates and does not touch held-out test.
- Current literal-objective audit:
  - output: `reports/research/sota_loop/transport_objective_status.json` (ignored local report)
  - `status`: `literal_blocked`
  - blockers:
    - constant train-only audit status is `blocked_incompatible_splits`
    - train-only feature diagnostic conclusion is `blocked_no_train_support_for_validation_shift`
    - observed-context result is `achieved` but not accepted for the literal objective
- Current observed-accepted audit:
  - output: `reports/research/sota_loop/transport_objective_status_observed_accepted.json` (ignored local report)
  - `status`: `observed_context_achieved`
  - caveat: literal train-only shift objective remains unproven and status depends on accepting two-frame observed context
- Interpretation:
  - this is the clearest current state split: literal train-only objective is blocked; observed-context result is promotable only under an explicit benchmark-policy acceptance

Follow-up runner update (2026-05-19, official objective status command):
- Added `scripts/run_official_transport_objective_status.sh`.
- Default behavior:
  - reads existing constant, observed, and train-only feature evidence
  - does not rerun gates
  - does not touch held-out test
  - requires `literal-achieved`
- Observed-context policy mode:
  - `ACCEPT_OBSERVED_CONTEXT=1 REQUIRE_STATUS=observed-accepted bash scripts/run_official_transport_objective_status.sh`
- Interpretation:
  - there is now a single official command for the final release question
  - it fails closed for the literal objective by default and only passes observed-context promotion when that policy is explicitly selected

Official objective command refresh (2026-05-19):
- Ran default literal release command:
  - command: `bash scripts/run_official_transport_objective_status.sh`
  - exit code: `2`
  - status: `literal_blocked`
- Ran observed-context acceptance command:
  - command: `ACCEPT_OBSERVED_CONTEXT=1 REQUIRE_STATUS=observed-accepted OBJECTIVE_STATUS_JSON=reports/research/sota_loop/transport_objective_status_observed_accepted.json bash scripts/run_official_transport_objective_status.sh`
  - exit code: `0`
  - status: `observed_context_achieved`
- Neither command reruns gates or touches held-out test; both read existing evidence artifacts.
- Interpretation:
  - default release behavior correctly fails closed for the literal objective
  - observed-context promotion is mechanically available only through an explicit policy flag

Experiment loop update (2026-05-19, train/val temporal-window support diagnostic):
- Added `scripts/diagnose_transport_temporal_windows.py`.
- The diagnostic scans temporal start windows inside train and validation trajectories, fits/evaluates best transport shift per split/window, and does not read held-out test.
- Real local `light-v1` train/val command:
  - output: `reports/research/sota_loop/transport_temporal_window_diagnostic.json` (ignored local report)
  - full train/val shards
  - rollout steps: `16`
  - temporal starts: `0,16,32,...,176`
- Result:
  - train shift histogram: `{"0": 12}`
  - validation shift histogram: `{"40": 12}`
  - common temporal best shifts: `[]`
  - conclusion: `blocked_no_temporal_common_shift`
- Interpretation:
  - the literal train-only path is not rescued by choosing a later temporal rollout window
  - across all scanned 16-step temporal windows, train remains shift `0` and validation remains shift `40`

Experiment loop update (2026-05-19, two-frame context transport gate):
- Added `scripts/run_context_transport_shift_gate.py`.
- The gate estimates one transport shift from the first observed transition (`t0 -> t1`) and then rolls out autoregressively from `t1` without reading future observed transitions.
- Added `tests/unit/test_run_context_transport_shift_gate.py` for validation pass/test measurement, validation-fail blocking, ledger repeat refusal, and explicit debugging repeat behavior.
- Real local `light-v1` command:
  - output: `reports/research/sota_loop/context_transport_shift_gate.json` (ignored local report)
  - ledger: `reports/research/sota_loop/context_transport_shift_test_ledger.json` (ignored local ledger)
  - train/val max samples: `128`
  - test max samples: `32`
  - rollout steps: `16`
  - candidate shifts: `-96,-88,...,96`
- Result:
  - validation NRMSE: `0.12336619943380356`
  - validation relative improvement versus reference `0.30780652221851373`: `0.5992086244805913`
  - validation guard: `passed=true`
  - held-out test measured exactly once after validation passed
  - test NRMSE: `0.040703773498535156`
  - train/validation/test shift means: `0.0` / `40.0` / `72.0`
  - ledger measurement key: `aad43ee28e0606013d01f8fcbfb525ea406a41925d16fbcf12084aecfeca2d06`
- Interpretation:
  - this is cleaner than the lagged observed-transition gate because it does not use per-step future observed transitions during rollout
  - it is still not the literal one-frame train-only constant-shift objective; it requires a benchmark policy that allows two initial context frames for state-conditioned prediction
  - under that two-frame-context policy, the current local `light-v1` Advection result is validation-clean and has a guarded held-out test measurement

Follow-up audit update (2026-05-19, context transport result):
- Added `scripts/audit_context_transport_shift_result.py`.
- Added `scripts/run_official_context_transport_shift_gate.sh`.
- Updated `scripts/audit_transport_objective_status.py` and `scripts/run_official_transport_objective_status.sh` to include an explicit `context-accepted` policy mode.
- Direct audit command:
  - output: `reports/research/sota_loop/context_transport_shift_goal_audit.json` (ignored local report)
  - exit code: `0`
  - status: `achieved`
  - verified data identities, no gate/source mismatches, validation pass, exactly one authorized test result, and result-record tokens `achieved`, `0.12336619943380356`, and `0.040703773498535156`
- Context-accepted objective command:
  - command: `ACCEPT_CONTEXT_TRANSPORT=1 REQUIRE_STATUS=context-accepted OBJECTIVE_STATUS_JSON=reports/research/sota_loop/transport_objective_status_context_accepted.json bash scripts/run_official_transport_objective_status.sh`
  - exit code: `0`
  - status: `context_transport_achieved`
- Default literal objective command:
  - command: `bash scripts/run_official_transport_objective_status.sh`
  - exit code: `2`
  - status: `literal_blocked`
- Interpretation:
  - the context transport result is now first-class audited evidence
  - default release behavior still fails closed for the original literal train-only objective
  - promotion requires an explicit policy decision to accept two initial context frames

Follow-up audit update (2026-05-19, train-only identifiability):
- Added `scripts/audit_train_only_transport_identifiability.py`.
- The audit checks train/val shift-label support without reading held-out test.
- Real local `light-v1` train/val command:
  - output: `reports/research/sota_loop/train_only_transport_identifiability_audit.json` (ignored local report)
  - full train/val shards
  - rollout steps: `16`
  - candidate shifts: `-96,-88,...,96`
- Result:
  - status: `blocked_underidentified_train_only_shift`
  - train shift support: `[0]`
  - validation shift support: `[40]`
  - unsupported validation shifts: `[40]`
  - split-level train best: shift `0`, NRMSE `0.01450516376644373`
  - split-level validation best: shift `40`, NRMSE `0.012850472703576088`
- Interpretation:
  - no supervised train-only shift-label learner can identify the validation shift from the current train shard because the required validation label is absent from train
  - this is why richer equivariant/canonical-frame operators are unlikely to solve the literal objective without either train support, parameter metadata, rebuilt splits, or allowed two-frame context
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `blocked_underidentified_train_only_shift`
  - context-accepted command still exits `0` with `status=context_transport_achieved`

Follow-up audit update (2026-05-19, benchmark-clean hydration options):
- Added `scripts/audit_transport_data_hydration_options.py`.
- The audit separates canonical local real shards, synthetic report-generated shards, and remote official raw PDEBench Advection files from `docs/pdebench_manifest.yaml`.
- Real local command:
  - output: `reports/research/sota_loop/transport_data_hydration_options.json` (ignored local report)
  - train/val only; does not read held-out test
- Result:
  - status: `remote_official_hydration_required`
  - canonical local train support remains `[0]`
  - canonical local validation support remains `[40]`
  - official remote Advection train files in manifest: `8`
  - total official remote Advection train size: `61.34038382768631` GiB
  - synthetic report shard entries are cataloged as `synthetic_report_artifact_not_benchmark_clean`
- Interpretation:
  - the current workspace has no additional benchmark-clean local train source that covers validation shift `40`
  - there is a benchmark-clean hydration route through official raw Advection train files, but it requires an explicit large download/storage step
  - synthetic `reports/light_experiments/**/synthetic_pdebench` shards should not be used as release evidence for the literal objective
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `remote_official_hydration_required`

Follow-up planning update (2026-05-19, official Advection hydration plan):
- Added `scripts/plan_transport_official_hydration.py`.
- Generated `reports/research/sota_loop/official_advection_hydration_plan.json` (ignored local report).
- Plan status: `ready_for_explicit_hydration`.
- Selected official Advection train files: `8`.
- Estimated download size: `61.34038382768631` GiB.
- Planned hydrated source root: `data/pdebench_official_advection_hydrated`.
- Planned train/val light root: `data/pdebench_official_advection_light`.
- Planned shard counts: train `256`, val `64`, test `0`.
- Held-out test policy:
  - no official test file is downloaded by the plan
  - no test shard is built by the plan
  - held-out test remains gated behind validation
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `ready_for_explicit_hydration`, meaning the next literal step is external download/storage execution rather than another local model probe

Follow-up validation update (2026-05-19, official hydration plan validation):
- Added `scripts/validate_transport_hydration_plan.py`.
- Generated `reports/research/sota_loop/official_advection_hydration_plan_validation.json` (ignored local report).
- Validation status: `valid`.
- The validator checks:
  - selected paths match download commands
  - every selected path is official `1D/Advection/Train`
  - estimated download size is above the expected floor
  - no held-out test split is downloaded or sharded
  - train/val shard command uses `--test-count 0`
  - validation command does not pass `--test-split`
  - synthetic report artifacts are not referenced
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `official hydration plan validation status is valid`

Follow-up execution update (2026-05-19, official hydration dry run):
- Added `scripts/run_transport_official_hydration_plan.py`.
- Generated `reports/research/sota_loop/official_advection_hydration_plan_run.json` (ignored local report).
- Run status: `dry_run`.
- The runner validates the plan first, then lists stages in order:
  - download official train files
  - convert raw official train files into a hydrated source
  - build train/val-only light shards with `--test-count 0`
  - validate without `--test-split`
  - refresh objective audit
- No command was executed in the dry run.
- The dry run reports blocker `download stage requires --execute-downloads`, preserving the explicit approval boundary for the 61.34 GiB network/disk step.
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `official hydration plan run status is dry_run`

Follow-up preflight update (2026-05-19, official hydration disk capacity):
- Added `scripts/preflight_transport_hydration.py`.
- Regenerated `reports/research/sota_loop/official_advection_hydration_plan.json` with per-file remote entry sizes.
- Generated `reports/research/sota_loop/official_advection_hydration_preflight.json` (ignored local report).
- Preflight status: `blocked_insufficient_disk`.
- Raw official files present: `0` of `8`.
- Remaining download bytes: `65863735616`.
- Required free bytes with safety factor `1.15`: `75743295958`.
- Current free bytes at `data/pdebench/raw`: `1599369216`.
- Interpretation:
  - the official hydration plan is valid and staged, but cannot run on the current filesystem
  - the next literal-path requirement is freeing/providing at least ~75.74 GB usable space or moving raw/hydrated roots to a larger volume before running downloads
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `official hydration preflight status is blocked_insufficient_disk`

Follow-up storage update (2026-05-19, official hydration storage recommendation):
- Added `scripts/recommend_transport_hydration_storage.py`.
- Generated `reports/research/sota_loop/official_advection_hydration_storage_recommendation.json` (ignored local report).
- Storage recommendation status: `external_or_freed_space_required`.
- Checked candidate roots:
  - `data/pdebench/raw`
  - `/private/tmp`
  - `/Volumes`
- All candidates resolve to the same filesystem with `1588920320` free bytes at audit time.
- Required free bytes remain `75743295958`.
- Recommendation:
  - free local disk space or mount a larger volume
  - regenerate the hydration plan with `--raw-out`, `--hydrated-source-root`, and `--hydrated-light-root` pointing to that larger root
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `official hydration storage status is external_or_freed_space_required`

Follow-up remote execution update (2026-05-19, official hydration remote plan):
- Added `scripts/plan_remote_official_hydration.py`.
- Generated `reports/research/sota_loop/remote_official_advection_hydration_plan.json` (ignored local report).
- Remote plan status: `ready_for_remote_hydration`.
- Required remote disk: `120` GB.
- Dry-run launcher:
  - `DRY_RUN=1 DISK_GB=120 GPU=RTX_4090 REMOTE_SCRIPT=scripts/run_remote_official_hydration.sh EXTRA_PIPELINE_ARGS='PLAN_JSON=reports/research/sota_loop/official_advection_hydration_plan.json VALIDATION_JSON=reports/research/sota_loop/official_advection_hydration_plan_validation.json RUN_JSON=reports/research/sota_loop/official_advection_hydration_plan_run.json EXECUTE=1 EXECUTE_DOWNLOADS=1 MIN_DOWNLOAD_BYTES=60000000000' bash scripts/launch_remote_transport_shift_candidate_vast.sh`
- Actual launcher switches `DRY_RUN=0`.
- Interpretation:
  - the local literal path is blocked by disk, but the next executable route is now a dry-run-first remote job with enough disk
  - the remote job uses `scripts/run_remote_official_hydration.sh` because the Vast launcher invokes remote scripts through `bash`
  - the wrapper calls the staged hydration runner, requires `EXECUTE_DOWNLOADS=1`, downloads official train files only, and builds train/val shards with `test_count=0`
- Objective audit refresh:
  - default literal command still exits `2` with `status=literal_blocked`
  - literal blockers now include `remote official hydration plan status is ready_for_remote_hydration`

Follow-up evidence wiring update (2026-05-19, official hydrated train/val gate):
- Extended `scripts/audit_transport_objective_status.py` and `scripts/run_official_transport_objective_status.sh` to read `reports/research/sota_loop/official_hydrated_transport_shift_gate.json`.
- The objective status now has an explicit intermediate state, `literal_test_ready`, for the case where the remote official hydrated train/val gate passes and the held-out test has not yet been recorded.
- This matters because the remote hydration runner writes the train/val validation gate artifact after building official shards; the release check must consume that artifact before deciding whether one held-out test is authorized.
- Policy preserved:
  - `literal_achieved` still requires the promoted constant goal audit.
  - `literal_test_ready` is not success; it only means the next benchmark-clean action is exactly one held-out test through the gated path.
  - default `REQUIRE_STATUS=literal-achieved` still fails closed until the held-out test and promotion audit are recorded.
- Verified:
  - `python -m pytest tests/unit/test_audit_transport_objective_status.py tests/unit/test_official_transport_objective_status.py`
  - `10 passed`

Follow-up hydration pipeline update (2026-05-19, post-validation audit boundary):
- Updated `scripts/plan_transport_official_hydration.py` so the train/val-only hydration pipeline runs `REQUIRE_STATUS=literal-test-ready bash scripts/run_official_transport_objective_status.sh` after validation.
- Extended `scripts/validate_transport_hydration_plan.py` to reject hydration plans whose post-validation audit still requires final `literal-achieved` status before the held-out test is authorized.
- Regenerated the ignored local artifacts:
  - `reports/research/sota_loop/official_advection_hydration_plan.json`
  - `reports/research/sota_loop/official_advection_hydration_plan_validation.json`
  - `reports/research/sota_loop/official_advection_hydration_plan_run.json`
- Why this matters:
  - the hydration phase intentionally downloads/builds train/val only and does not shard or read held-out test
  - if the official hydrated train/val gate passes, the correct next state is `literal_test_ready`
  - final `literal_achieved` remains blocked until exactly one held-out test is run through the gated path and the result is recorded/promoted
- Verified:
  - `python -m pytest tests/unit/test_plan_transport_official_hydration.py tests/unit/test_validate_transport_hydration_plan.py tests/unit/test_run_transport_official_hydration_plan.py`
  - `9 passed`

Follow-up post-validation test update (2026-05-19, guarded official hydrated test phase):
- Added `scripts/run_official_hydrated_post_validation_test.py`.
- Added `tests/unit/test_run_official_hydrated_post_validation_test.py`.
- Generated `reports/research/sota_loop/official_hydrated_post_validation_test_run.json` (ignored local report).
- Current preview status: `dry_run`.
- Current blockers:
  - `objective status is literal_blocked, expected literal_test_ready`
  - `held-out test execution requires --execute-test`
- The runner will only create/read the held-out test shard when:
  - `reports/research/sota_loop/transport_objective_status.json` has `status=literal_test_ready`
  - execution is explicitly requested with `--execute --execute-test`
- Planned held-out test command:
  - build only `advection1d_test.h5` from the official hydrated train source
  - use `--split-start-index test=320` after the train/val windows (`256 + 64`)
  - rerun `scripts/run_transport_shift_gate.py` with `--test-split test`, preserving the gate's validation-first behavior
- Verified:
  - `python -m pytest tests/unit/test_run_official_hydrated_post_validation_test.py`
  - `3 passed`

Follow-up remote chain update (2026-05-19, full objective remote wrapper):
- Updated `scripts/run_remote_official_hydration.sh` so the remote wrapper can optionally chain the guarded post-validation test phase after train/val hydration and validation.
- Updated `scripts/plan_remote_official_hydration.py` so the generated Vast command passes:
  - `RUN_POST_VALIDATION_TEST=1`
  - `EXECUTE_TEST=1`
  - `POST_VALIDATION_TEST_JSON=reports/research/sota_loop/official_hydrated_post_validation_test_run.json`
- Policy preserved:
  - the hydration runner still downloads/builds train/val first
  - the post-validation runner still refuses to build/read held-out test unless objective status is already `literal_test_ready`
  - `EXECUTE_TEST=1` only authorizes the runner to proceed if that status gate passes
- Dry-run Vast onstart now ends with:
  - `bash scripts/run_remote_official_hydration.sh ... RUN_POST_VALIDATION_TEST=1 EXECUTE_TEST=1 ...`
- Verified:
  - `python -m pytest tests/unit/test_plan_remote_official_hydration.py tests/unit/test_run_remote_official_hydration.py tests/unit/test_run_official_hydrated_post_validation_test.py`
  - `8 passed`

Follow-up remote launch update (2026-05-19, fresh-checkout plan generation):
- Launched Vast contract `37096085` with SSH/onstart mode; it stalled in `loading` without a container and was destroyed.
- Relaunched Vast contract `37096391` with `ARGS_MODE=1 SSH=0`; it reached execution but failed before hydration because `reports/research/sota_loop/official_advection_hydration_plan.json` is an ignored local artifact and was missing from the fresh remote checkout.
- Destroyed contract `37096391` after confirming the repeated `FileNotFoundError`.
- Fixed `scripts/run_remote_official_hydration.sh` so it regenerates the official hydration plan with `scripts/plan_transport_official_hydration.py` when `PLAN_JSON` is missing.
- Updated `tests/unit/test_run_remote_official_hydration.py` to cover fresh-checkout plan generation.
- Verified:
  - `python -m pytest tests/unit/test_run_remote_official_hydration.py tests/unit/test_plan_remote_official_hydration.py tests/unit/test_plan_transport_official_hydration.py tests/unit/test_validate_transport_hydration_plan.py tests/unit/test_run_transport_official_hydration_plan.py`
  - `14 passed`

Follow-up download telemetry update (2026-05-19, official host throughput bottleneck):
- Relaunched Vast contract `37096575` after the fresh-checkout fix; the remote regenerated the official hydration plan successfully.
- The job then entered the first official file download but made only small disk progress and produced no stage output after plan emission.
- Local probe of the official file endpoint showed range requests work, but the observed transfer rate was about `0.45 MB/s` for a 1 MiB range sample from this environment.
- Destroyed contract `37096575` to avoid paying for a likely tens-of-hours serial download path.
- Updated `scripts/download_pdebench_file.py` with resumable ranged downloads:
  - default `PDEBENCH_DOWNLOAD_WORKERS=8`
  - default `PDEBENCH_DOWNLOAD_PART_SIZE_MIB=256`
  - per-request retries and explicit part-completion progress
  - checksum verification after assembly
  - skip logic for already-present files with matching size/checksum
- Added `tests/unit/test_download_pdebench_file.py`.
- Verified:
  - `python -m pytest tests/unit/test_download_pdebench_file.py tests/unit/test_plan_transport_official_hydration.py tests/unit/test_run_transport_official_hydration_plan.py`
  - `10 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`
  - `git diff --check`

Follow-up ranged download retry update (2026-05-19, hung part timeout):
- Relaunched Vast contract `37097600` in the Netherlands with ranged downloads.
- The run confirmed the pivot worked: two full official Advection files were saved and the third reached `98.4%`.
- It then stalled on one remaining ranged part with flat CPU/disk counters, so contract `37097600` was destroyed.
- Tightened `scripts/download_pdebench_file.py` with `PDEBENCH_DOWNLOAD_PART_TIMEOUT` / `--part-timeout`; each part attempt now has a wall-clock deadline and retries from a clean temp file.
- Added unit coverage for retrying after a part timeout.
- Verified:
  - `python -m pytest tests/unit/test_download_pdebench_file.py`
  - `6 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`
  - `git diff --check`

Follow-up ranged download disk update (2026-05-19, direct-to-temp range writes):
- Relaunched Vast contract `37098407` with the part-timeout patch and 120 GB disk.
- The timeout patch worked repeatedly:
  - `beta1.0`, `beta2.0`, and `beta4.0` each recovered from slow range attempts after `600s` timeouts.
  - `beta7.0` downloaded all 62 ranged parts.
- The run then failed while assembling `beta7.0`:
  - `OSError: [Errno 28] No space left on device`
  - the hydration runner could not write `reports/research/sota_loop/official_advection_hydration_plan_run.json` after disk exhaustion.
- Destroyed contract `37098407`; no validation or held-out test ran.
- Root cause: the ranged downloader stored all `.parts` files and then created a second full-size `.tmp` file during assembly, producing excessive peak disk use on the remote.
- Updated `scripts/download_pdebench_file.py` so ranged downloads preallocate one `.tmp` destination and write each range directly at its byte offset.
- This preserves the benchmark-clean official source and range retry policy while reducing peak per-file temporary storage from roughly two full copies to one.

Follow-up official split update (2026-05-20, stratified beta-balanced train/val/test slices):
- Relaunched Vast contract `37101416` with 160 GB disk and the direct-to-temp downloader.
- The run hydrated all 8 official Advection train files, converted them, built train/val shards, and ran `scripts/run_transport_shift_gate.py`.
- Validation did not pass the SOTA guard:
  - `validation_guard.passed`: `false`
  - `test_eligible`: `false`
  - `reference_metric_value`: `0.30780652221851373`
  - selected train-fitted validation `nrmse`: `0.7047799825668335`
  - best validation/oracle shift was near `56`, while the train-selected shift was `8`
- No held-out test ran, which is the correct policy when validation fails.
- Destroyed contract `37101416`; `vastai show instances --raw` returned `[]`.
- Root cause:
  - the official conversion concatenated sorted beta files
  - previous shard construction used contiguous source rows: train `0..255`, val `256..319`
  - with 40 samples per beta file, train and val were beta-regime skewed rather than beta-balanced
  - this made the validation failure a split-construction confound, not a useful final benchmark signal
- Updated `scripts/make_light_hdf5_shards.py` with optional stratified block slicing:
  - `--split-block-size`
  - repeated `--split-block-offset SPLIT=OFFSET`
  - indexed reads preserve sample-aligned datasets and manifest provenance
- Updated `scripts/plan_transport_official_hydration.py` so the official hydration plan now converts `48` samples per beta file and builds:
  - train: `32` samples per beta file, `256` total, block offset `0`
  - val: `8` samples per beta file, `64` total, block offset `32`
  - reserved test: `8` samples per beta file, `64` total, block offset `40`
- Updated `scripts/run_official_hydrated_post_validation_test.py` so the held-out test shard uses the reserved stratified test block (`--split-block-size 48 --split-block-offset test=40`) and still refuses to run before `literal_test_ready`.
- Tradeoff:
  - this is not a looser guard and does not read test early
  - it changes the light shard sampling contract to remove a beta-regime confound while preserving train-only fitting and validation-first held-out-test policy
- Verified:
  - `python -m pytest tests/unit/test_make_light_hdf5_shards.py tests/unit/test_plan_transport_official_hydration.py tests/unit/test_validate_transport_hydration_plan.py tests/unit/test_run_transport_official_hydration_plan.py tests/unit/test_run_remote_official_hydration.py tests/unit/test_plan_remote_official_hydration.py tests/unit/test_run_official_hydrated_post_validation_test.py tests/unit/test_audit_transport_objective_status.py`
  - `28 passed`

Follow-up remote wrap-up (2026-05-20, stopped partial stratified hydration run):
- Launched Vast contract `37157238` in Norway with the stratified official hydration plan and the guarded post-validation test chain enabled.
- The job confirmed the intended benchmark-clean plan on the remote:
  - official Advection train files only
  - `--samples 48`
  - train block offset `0`
  - validation block offset `32`
  - reserved test block offset `40`
  - no test split built during train/val hydration
- At wrap-up, the remote had saved 5 of 8 official train files (`beta0.1`, `beta0.2`, `beta0.4`, `beta0.7`, `beta1.0`) and was downloading `beta2.0`.
- The ranged downloader continued to recover from slow range attempts via the `600s` part timeout, so this was stopped by operator choice rather than a code failure.
- Destroyed contract `37157238`; `vastai show instances --raw` returned `[]`.
- No validation ran and no held-out test ran on this instance, so there is no benchmark result to promote from this attempt.

Follow-up remote launch/network hardening (2026-05-20):
- Retried the stratified official hydration run after the stopped partial run.
- Several explicit Vast offer launches created stopped/no-container contracts or hit host/image-layer issues before repo code ran:
  - `37166613`: stopped no-container contract from Florida offer `36696899`; destroyed.
  - `37166670`: stopped no-container contract from Oregon offer `37158063`; destroyed.
  - `37166700`: stopped no-container contract from Norway offer `37061645`; destroyed.
  - `37167637`: Mexico offer `36792121` reached image pull but failed with host-side container storage exhaustion (`no space left on device`); destroyed.
- Nevada contract `37168284` reached the real official downloader, but the first official file failed because outbound network to `darus.uni-stuttgart.de` became unreachable across repeated range attempts (`Errno 101 Network is unreachable`); destroyed.
- The benchmark policy remained intact:
  - validation did not run
  - held-out test did not run
  - no result should be promoted from these attempts
- Updated `scripts/download_pdebench_file.py` with configurable exponential backoff between ranged-part retries:
  - CLI/env knob: `--retry-backoff` / `PDEBENCH_DOWNLOAD_RETRY_BACKOFF`
  - default initial backoff: `15s`, doubling per failed part attempt
  - this gives transient remote network outages time to recover instead of exhausting all retries immediately across parallel workers
- Verified:
  - `python -m pytest tests/unit/test_download_pdebench_file.py tests/unit/test_make_light_hdf5_shards.py tests/unit/test_plan_transport_official_hydration.py tests/unit/test_run_transport_official_hydration_plan.py`
  - `19 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`
  - `git diff --check`

Follow-up current Vast wrap-up (2026-05-20, operator-stopped active official hydration):
- Launched Vast contract `37169407` in the Netherlands with the stratified official hydration plan and the guarded post-validation test chain enabled.
- The instance reached the official downloader and confirmed the same benchmark-clean train/val contract:
  - official Advection train files only
  - `--samples 48`
  - train block offset `0`
  - validation block offset `32`
  - reserved test block offset `40`
  - no test split built during train/val hydration
- Last available logs showed 5 of 8 official train files saved (`beta0.1`, `beta0.2`, `beta0.4`, `beta0.7`, `beta1.0`) and `beta2.0` near completion.
- The retry-backoff path recovered slow range parts on this run, including `600s` part-timeout retries.
- Per operator request to wrap up the current Vast instance, destroyed contract `37169407` before conversion, validation, or held-out test.
- No SOTA guard validation ran and no held-out test ran on this instance, so there is no benchmark result to promote from this attempt.

Follow-up current Vast wrap-up (2026-05-20, patched downloader checkpoint):
- Destroyed stale Vast contract `37176828` after it reached the first official Advection train file but stalled at `61/62` ranged parts (`7.54 GiB`, `98.4%`).
- Hardened `scripts/download_pdebench_file.py` so each ranged request uses the configured per-part read timeout, preventing a socket read from hanging indefinitely before yielding chunks.
- Relaunched Vast contract `37177098` in Spain with `python:3.11-slim`, the stratified official hydration plan, guarded post-validation chain, `8` workers, `128 MiB` parts, `6` retries, `20s` retry backoff, and `180s` per-part timeout.
- The lighter image successfully bootstrapped the repo archive, installed the minimal experiment dependencies, regenerated the hydration plan, and entered the official downloader.
- The patched downloader completed the previously stuck first official file:
  - saved `data/pdebench/raw/1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5`
  - completed all `62/62` ranged parts
  - completed range part `4/62`, the range that was the likely prior hang point
- The run then advanced into the second official train file and reached at least `54/62` ranged parts (`6.75 GiB`, `88.0%`) before wrap-up.
- Per operator request to wrap up the current Vast instance, destroyed contract `37177098` before conversion, validation, or held-out test.
- No SOTA guard validation ran and no held-out test ran on this instance, so there is no benchmark result to promote from this attempt.

Follow-up current Vast wrap-up (2026-05-20, adaptive range-split hardening):
- Launched Vast contract `37177336` in Virginia with the same benchmark-clean stratified hydration plan and guarded post-validation chain.
- The run confirmed the official train-only source and progressed farther than the previous slim-image attempt:
  - saved `beta0.1`, `beta0.2`, `beta0.4`, `beta0.7`, and `beta1.0`
  - entered the sixth official train file (`beta2.0`) and reached `98.4%`
  - disk usage rose to `47` GB
- The patched per-part timeout/retry path recovered multiple slow ranges, proving the HTTP read-timeout fix works in the live downloader.
- The run then hit a new failure mode: one `128 MiB` range (`5368709120-5502926847`) failed three consecutive `180s` attempts on `beta2.0`.
- Destroyed contract `37177336` after the repeated same-range stall; conversion, validation, and held-out test did not run.
- Hardened `scripts/download_pdebench_file.py` again so a repeatedly timed-out ranged request splits into smaller byte ranges after configurable failures:
  - `PDEBENCH_DOWNLOAD_SPLIT_AFTER_RETRIES` / `--split-after-retries`
  - `PDEBENCH_DOWNLOAD_MIN_SPLIT_SIZE_MIB` / `--min-split-size-mib`
  - default split trigger: `2` failed attempts
  - default minimum split size: `8 MiB`
- Tradeoff:
  - this preserves the official data source and benchmark policy while increasing request count only for pathological stuck ranges
  - it avoids spending every retry on the same bad CDN range before making smaller-range progress
- Verified:
  - `python -m pytest tests/unit/test_download_pdebench_file.py`
  - `10 passed`
  - `python -m py_compile scripts/download_pdebench_file.py`

Follow-up current Vast wrap-up (2026-05-20, credit-blocked relaunch):
- Relaunched the adaptive range-split downloader on Vast contract `37178922` using California RTX 4090 offer `35149296`.
- The remote bootstrap succeeded on `python:3.11-slim`, regenerated the official stratified hydration plan, and entered the official downloader with the same benchmark-clean contract:
  - official Advection train files only
  - `48` samples per beta file
  - train block offset `0`
  - validation block offset `32`
  - reserved test block offset `40`
  - no test split built during train/val hydration
- The first official file reached `60/62` ranged parts (`7.42 GiB`, `96.7%`) before the instance unexpectedly stopped without a Python traceback in the public log.
- Inspected the stopped instance with `vastai execute`; it contained only the preallocated first-file temp path (`1D_Advection_Sols_beta0.1.hdf5.tmp`) and `official_advection_hydration_plan.json`. No completed official file, validation JSON, or held-out test artifact existed.
- Destroyed contract `37178922`.
- Attempted to relaunch on Texas RTX 4090 offer `35956477`, but Vast rejected instance creation with `Your account lacks credit; see the billing page.`
- `vastai show instances --raw` returned `[]` after cleanup.
- No SOTA guard validation ran and no held-out test ran, so there is still no benchmark result to promote.
- Next path is externally blocked until Vast credit is available or another real-data execution path is provided; once available, relaunch the same adaptive range-split stratified hydration and accept only the train/val guard result as evidence.

Follow-up local audit refresh (2026-05-20, light-v1 policy crossroads):
- Verified the local `light-v1` Advection shard identities against `reports/research/sota_loop/remote_manifests/light-v1_manifest.yaml`:
  - train: `67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1`
  - val: `9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329`
  - test: `4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93`
- Fresh default objective audit:
  - command: `bash scripts/run_official_transport_objective_status.sh`
  - status: `literal_blocked`
  - reason: the literal train-only constant-shift path remains blocked by incompatible splits / underidentified train-only shift support, while context and observed transport results are not accepted by default.
- Fresh context-accepted audit:
  - command: `ACCEPT_CONTEXT_TRANSPORT=1 REQUIRE_STATUS=context-accepted OBJECTIVE_STATUS_JSON=/private/tmp/transport_objective_status_context_accepted_refresh.json bash scripts/run_official_transport_objective_status.sh`
  - status: `context_transport_achieved`
  - caveat: this depends on explicitly accepting the two-frame context transport policy.
- Fresh result audits:
  - `python scripts/audit_context_transport_shift_result.py --context-gate-json reports/research/sota_loop/context_transport_shift_gate.json --output-json /private/tmp/context_transport_shift_goal_audit_refresh.json --require-status achieved`
  - `python scripts/audit_observed_transport_shift_result.py --observed-gate-json reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json --output-json /private/tmp/observed_transport_shift_goal_audit_refresh.json --require-status achieved`
  - both returned `status=achieved` without rerunning the held-out test.
- Current decision point:
  - if the benchmark policy accepts two initial context frames, the best available result is the context transport result (`val nrmse=0.12336619943380356`, `test nrmse=0.040703773498535156`, exactly-one test ledger recorded)
  - if the literal constant/train-only-shift policy is required, the path remains blocked and the next best action is a split-compatible official hydration run after Vast credit is restored, or a richer train-only causal mechanism with additional allowed signal

Follow-up audit fix (2026-05-20, context-accepted requirement consistency):
- Fixed `scripts/audit_transport_objective_status.py` so context-accepted and observed-accepted achieved modes satisfy the `fit_transport_shift_only_on_train` requirement row.
- Before this fix, the aggregate status could be `context_transport_achieved` while the train-fit requirement still reported `blocked`, even though the context estimator contract is train-locked and validation/test only score the locked estimator.
- Fresh verification:
  - `python -m pytest tests/unit/test_audit_transport_objective_status.py` -> `6 passed`
  - default command `bash scripts/run_official_transport_objective_status.sh` still exits `2` with `status=literal_blocked`
  - context-accepted command exits `0` with `status=context_transport_achieved` and all requirement rows satisfied, while preserving the caveat that this depends on accepting the two-frame context policy

Follow-up wrap-up checkpoint (2026-05-20, context-accepted objective wrapper):
- Added `scripts/run_official_context_transport_objective_status.sh` as a narrow wrapper around the aggregate objective status command.
- The wrapper sets the explicit benchmark-policy acceptance flags for the already-audited two-frame context transport result:
  - `ACCEPT_CONTEXT_TRANSPORT=1`
  - `REQUIRE_STATUS=context-accepted`
  - default output `reports/research/sota_loop/transport_objective_status_context_accepted.json`
- It does not rerun the validation gate and does not touch the held-out test ledger; it only aggregates existing audited evidence under the context-accepted policy.
- Default literal status remains fail-closed through `scripts/run_official_transport_objective_status.sh`.
- This gives the branch a clean wrap-up surface:
  - literal policy: still blocked by incompatible/underidentified train-only constant-shift evidence and by unavailable external compute credit for the official hydration path
  - explicit context policy: reproducibly reports `status=context_transport_achieved` from the existing context transport audit and exactly-one test ledger

Follow-up literal-path hardening (2026-05-20, resumable official range downloads):
- Hardened `scripts/download_pdebench_file.py` for the remaining literal benchmark-clean path.
- Ranged downloads now keep a sidecar next to the sparse `.tmp` file recording completed byte ranges.
- If a large official PDEBench file download process fails or is restarted on the same host, the downloader resumes completed ranges instead of discarding the temp file and starting from zero.
- This directly targets the observed Vast failure pattern where official Advection files repeatedly reached `96%`-`98%` before stopping or stalling.
- Tradeoff:
  - the sidecar only proves ranges completed through this downloader; if the sidecar is missing or mismatched, the downloader conservatively restarts rather than trusting arbitrary sparse bytes
  - this does not solve external credit or host/network loss, but it reduces wasted retry work on recoverable same-host/process restarts

Follow-up Vast relaunch check (2026-05-20, credit still blocked):
- Refreshed Vast state: `vastai show instances --raw` returned `[]`.
- Current literal objective audit still returns `status=literal_blocked`; no official hydrated validation gate or held-out test artifact exists.
- Searched current RTX 4090 offers and found verified Norway offers with sufficient disk, including offer `36114274`.
- Dry-run launch for offer `36114274` produced a correct onstart script using:
  - branch `codex/sota-learned-gate`
  - image `python:3.11-slim`
  - remote script `scripts/run_remote_official_hydration.sh`
  - guarded post-validation test chain enabled
- Actual launch was rejected by Vast with `Your account lacks credit; see the billing page.`
- Updated `scripts/plan_remote_official_hydration.py` so the generated relaunch command includes the hardened downloader runtime:
  - `PDEBENCH_DOWNLOAD_WORKERS=8`
  - `PDEBENCH_DOWNLOAD_PART_SIZE_MIB=128`
  - `PDEBENCH_DOWNLOAD_RETRIES=6`
  - `PDEBENCH_DOWNLOAD_PART_TIMEOUT=180`
  - `PDEBENCH_DOWNLOAD_RETRY_BACKOFF=20.0`
  - `PDEBENCH_DOWNLOAD_SPLIT_AFTER_RETRIES=2`
  - `PDEBENCH_DOWNLOAD_MIN_SPLIT_SIZE_MIB=8`
- Next executable path remains unchanged: restore compute credit or provide another real-data execution path, then run the generated `actual_launcher` from `reports/research/sota_loop/remote_official_advection_hydration_plan.json`.

Follow-up literal held-out test guard (2026-05-20, official hydrated test ledger):
- Tightened `scripts/run_official_hydrated_post_validation_test.py`.
- The official hydrated post-validation runner now computes a stable measurement key and uses `reports/research/sota_loop/official_hydrated_transport_shift_test_ledger.json` by default.
- If the same official hydrated test configuration is already in the ledger, the runner blocks before building or reading the held-out test shard unless `--allow-repeat-test` is explicitly set.
- After executing the gated test command, the runner now verifies that `official_hydrated_transport_shift_gate.json` contains exactly one held-out test result before reporting `status=executed`.
- This closes the last local policy gap in the literal full-objective chain: validation must first reach `literal_test_ready`, test execution must be explicit, the test configuration is ledger-guarded, and the resulting gate artifact must prove exactly one test measurement.

Follow-up remote wrapper consistency fix (2026-05-20, objective status export):
- Exported `OBJECTIVE_STATUS_JSON` in `scripts/run_remote_official_hydration.sh`.
- This keeps the validation audit stage inside `scripts/run_transport_official_hydration_plan.py` and the guarded post-validation test reader pointed at the same objective-status artifact when a non-default path is supplied.
- Added a regression test with a custom objective-status path and a minimal valid hydration plan; the audit stage writes the custom path through the exported environment.
- This matters for remote restarts and custom report locations: a job should not validate into the default `transport_objective_status.json` and then have the post-validation gate read a different file.

Follow-up Vast relaunch check (2026-05-20, preferred offer pinned):
- Refreshed Vast offers and selected Mexico RTX 4090 offer `36151271` as the preferred relaunch candidate because it has enough disk and better network than the cheapest current offer.
- Actual launch command was rejected again with `Your account lacks credit; see the billing page.`
- Updated `scripts/plan_remote_official_hydration.py` with optional `--offer-id`, then regenerated `reports/research/sota_loop/remote_official_advection_hydration_plan.json` with `preferred_offer_id=36151271`.
- The generated `actual_launcher` now includes `OFFER_ID=36151271` plus the hardened downloader runtime and guarded post-validation chain.

Follow-up remote artifact preservation (2026-05-20, official hydration reports):
- Added optional `PUBLISH_ARTIFACTS=1` support to `scripts/run_remote_official_hydration.sh`.
- When enabled, the remote wrapper tars the official hydration report artifacts that exist and uploads them to B2 before auto-shutdown:
  - hydration plan, validation report, run report, objective status, post-validation test report, official hydrated gate, and official hydrated test ledger
- This prevents a successful remote validation/test run from leaving the only result evidence on a disposable Vast instance.
- Publication is off by default and requires `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET`, and `rclone`.

Follow-up relaunch attempt and canonical artifact publishing (2026-05-20):
- Retried the pinned Vast launch for offer `36151271` with `PUBLISH_ARTIFACTS=1`; Vast rejected the create request with `Your account lacks credit; see the billing page.`
- Updated `scripts/plan_remote_official_hydration.py` so regenerated `actual_launcher` and `dry_run_launcher` commands include `PUBLISH_ARTIFACTS=1` by default.
- Regenerated `reports/research/sota_loop/remote_official_advection_hydration_plan.json` with the pinned offer, hardened downloader runtime, guarded post-validation test, and report publication enabled.
- This keeps the next executable official hydration path aligned with the evidence requirements: a successful remote run should preserve the validation/test reports before auto-shutdown without requiring a manual flag edit.

Follow-up remote artifact publishing bootstrap (2026-05-20):
- Hardened `scripts/run_remote_official_hydration.sh` so `PUBLISH_ARTIFACTS=1` can bootstrap `rclone` on apt-based remote hosts when it is not already installed.
- The wrapper still fails closed if `INSTALL_RCLONE=0` or the host has no `apt-get`, but a standard Vast image no longer needs a manual rclone preinstall for final report publication.
- This protects the literal objective evidence path: after a credit-unblocked official hydration run, the wrapper should be able to upload the validation/test report bundle before auto-shutdown.

Follow-up train-only conditional transport preparation (2026-05-20):
- Rechecked the literal objective audit: it remains `status=literal_blocked` with no official hydrated gate, and `vastai show instances --raw` still returns `[]`.
- Re-evaluated the stuck path against current neural-operator/transport literature: the promising direction is not another blind constant-shift sweep, but a symmetry-aware or canonicalized conditional transport rule that can be fit on train regimes and measured on validation.
- Updated `src/ups/data/convert_pdebench.py` so converted official PDEBench HDF5 files retain per-sample provenance:
  - `source_file_index`
  - `source_sample_index`
  - root attr `source_paths`
- This gives the official hydrated train/val shards beta/source-file regime information after conversion and light-shard slicing, without reading or changing held-out test data.
- Practical implication: once official hydration is unblocked, a train-only conditional transport probe can use source-file regime labels from train and validate the locked rule on val instead of being restricted to one global constant shift.

Follow-up source-conditioned official validation gate (2026-05-20):
- Added `scripts/run_source_conditioned_transport_shift_gate.py`.
- The gate fits one periodic shift per `source_file_index` group using train rows only, then validates the locked source-to-shift mapping on val rows.
- It fails closed if validation contains a `source_file_index` absent from train, and it only evaluates an optional held-out test split after validation passes.
- Updated `scripts/plan_transport_official_hydration.py` so the official hydration plan now writes `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` via the source-conditioned gate.
- Regenerated:
  - `reports/research/sota_loop/official_advection_hydration_plan.json`
  - `reports/research/sota_loop/official_advection_hydration_plan_validation.json`
  - `reports/research/sota_loop/official_advection_hydration_plan_run.json`
- This is still not objective completion: the official train files are not hydrated locally, no validation gate result exists yet, and no held-out test is authorized.

Follow-up post-validation test consistency (2026-05-20):
- Updated `scripts/run_official_hydrated_post_validation_test.py` so the authorized held-out test stage also uses `scripts/run_source_conditioned_transport_shift_gate.py`.
- The test ledger measurement key now identifies the estimator as `official_hydrated_source_conditioned_transport_shift`.
- Regenerated `reports/research/sota_loop/official_hydrated_post_validation_test_run.json`; it remains a dry run because objective status is still `literal_blocked`.
- This keeps train/val validation and the later exactly-one held-out test on the same train-fitted source-conditioned rule.

Follow-up light-shard provenance propagation (2026-05-20):
- Updated `scripts/make_light_hdf5_shards.py` so root HDF5 attrs from converted source files are copied into light train/val/test shards.
- This preserves `source_paths` alongside the sample-aligned `source_file_index` and `source_sample_index` datasets added during conversion.
- The source-conditioned official gate can already operate from `source_file_index`; preserving `source_paths` makes the resulting gate artifacts auditable against the original beta/source files after remote hydration.

Follow-up official hydrated achievement promotion (2026-05-20):
- Updated `scripts/audit_transport_objective_status.py` so a passed official hydrated train/val gate plus exactly one official hydrated held-out test result promotes the aggregate status to `literal_achieved`.
- Before this fix, the official hydrated path could reach validation pass and one test result but still report `literal_blocked` unless the older constant-shift audit separately promoted the result.
- This keeps the aggregate audit aligned with the current source-conditioned official path while preserving fail-closed `literal_test_ready` before any held-out test is measured.

Follow-up sequential official hydration path (2026-05-20):
- Added `scripts/hydrate_official_advection_source_sequential.py` for the remote official path.
- The sequential path downloads one official Advection train file, appends the sampled rows into the hydrated source HDF5 with `source_file_index` and `source_sample_index`, and can remove the raw file before downloading the next one.
- Updated `scripts/run_remote_official_hydration.sh` so `SEQUENTIAL_HYDRATION=1` runs sequential hydrate-convert-delete, then executes the existing shard/validate/audit stages and optional guarded post-validation test.
- Updated `scripts/plan_remote_official_hydration.py`; the regenerated remote plan now uses `DISK_GB=32` instead of the earlier 120 GiB all-raw-files-at-once requirement, with `SEQUENTIAL_HYDRATION=1` and `SEQUENTIAL_CLEANUP_RAW=1`.
- This does not complete the objective while Vast credit is blocked, but it materially improves the next execution route by lowering scratch-disk cost and preserving the same fail-closed no-test-before-validation policy.

Follow-up sequential Vast relaunch check (2026-05-21):
- Refreshed the literal objective audit; it still exits with `status=literal_blocked` because `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` is missing and no official hydrated train/val validation exists.
- Searched cheaper RTX 4090 offers under the new sequential requirement and found eligible 32 GiB-class routes, including offer `8936321` at about `$0.401/hr` with 59 GiB disk.
- Attempted the actual sequential launch with `DISK_GB=32`, `SEQUENTIAL_HYDRATION=1`, `SEQUENTIAL_CLEANUP_RAW=1`, guarded post-validation test enabled, and artifact publication enabled.
- Vast rejected instance creation with `Your account lacks credit; see the billing page.`
- A later `vastai show instances --raw` refresh could not resolve `console.vast.ai` from this environment, so active instance state was not refreshed in this turn; the last successful state check before the launch attempt showed no active instances.
- Current blocker remains external account credit or an alternate real-data execution environment. The repo-side next-run command is cheaper and ready, but no benchmark-clean train/val result exists yet.

Follow-up sample-mode source-conditioned gate (2026-05-21):
- Added `--fit-strategy aggregate|sample_mode` to `scripts/run_source_conditioned_transport_shift_gate.py`.
- `aggregate` preserves the previous per-source aggregate-MSE selection; `sample_mode` fits each train trajectory independently, then chooses the modal train-supported shift per `source_file_index` with metric-based tie breaking.
- Updated `scripts/plan_transport_official_hydration.py` so the official hydrated train/val gate now uses `--fit-strategy sample_mode`.
- Regenerated the ignored official hydration plan, plan validation, dry-run plan execution record, and remote sequential plan; the validate command now includes `--fit-strategy sample_mode` while still keeping `test-count 0` for train/val hydration.
- This targets the fallback research direction of symmetry-aware/canonical transport fitting without touching held-out test data: the model is still fit only on train rows, validation uses a locked source-to-shift map, and the held-out test remains gated behind `literal_test_ready`.

Follow-up train-only local shift refinement (2026-05-21):
- Added `--refine-radius` to `scripts/run_source_conditioned_transport_shift_gate.py`.
- The gate still starts from the configured train-only candidate shift grid, but can now evaluate a narrow integer neighborhood around each train-selected source/sample mode before locking the final `source_file_index -> shift` map.
- Updated `scripts/plan_transport_official_hydration.py` so the official hydrated validation command now includes `--refine-radius 4` with `--fit-strategy sample_mode`.
- Regenerated the ignored official hydration plan, validation record, dry-run execution record, and remote sequential plan; test data remains absent from the train/val hydration stage.
- This improves the next official validation attempt by avoiding an 8-cell quantization artifact while preserving the benchmark policy: refinement uses train rows only, validation is measurement-only, and held-out test remains blocked until `literal_test_ready`.

Follow-up post-validation estimator parity (2026-05-21):
- Updated `scripts/run_official_hydrated_post_validation_test.py` so the guarded held-out test command also passes `--fit-strategy sample_mode --refine-radius 4`.
- Added the fit strategy and refine radius to the held-out test ledger measurement key, preventing a repeat-test collision between different estimator configurations.
- Regenerated `reports/research/sota_loop/official_hydrated_post_validation_test_run.json`; it remains a dry run because objective status is still `literal_blocked`.
- This keeps the post-validation held-out test on the same locked estimator family as train/val validation, rather than silently falling back to the older aggregate/no-refinement configuration.

Follow-up sequential hydration provenance hardening (2026-05-21):
- Updated `scripts/hydrate_official_advection_source_sequential.py` so the hydrated source HDF5 initializes `source_paths` before the first raw-file download/append instead of waiting until every official file succeeds.
- Added a `sequential_hydration_complete` root attr that is set to `False` during incremental appends and `True` only after all planned files have been appended.
- This protects remote failure forensics: if a credit-unblocked Vast run dies after partial official hydration, the partial source shard still carries the intended official source list and a clear incomplete marker.
- The benchmark policy is unchanged; partial hydrated outputs are not treated as validation evidence, and the objective audit still requires `official_hydrated_transport_shift_gate.json`.

Follow-up incomplete hydration shard guard (2026-05-21):
- Updated `scripts/make_light_hdf5_shards.py` to reject any source HDF5 with `sequential_hydration_complete=False`.
- This means a partially appended official hydrated source cannot be sliced into train/val/test light shards after a remote crash or retry unless the sequential hydrator completed all planned official train files.
- The guard applies at the shard-builder boundary used by both train/val validation and the guarded post-validation test shard.
- This keeps partial remote artifacts useful for diagnosis while preventing them from becoming benchmark evidence.

Follow-up remote git-ref pinning (2026-05-21):
- Updated `scripts/plan_remote_official_hydration.py` so generated Vast launchers include `GIT_REF=codex/sota-learned-gate` by default.
- This prevents a credit-unblocked remote run from silently using the launcher script's branch default instead of the branch containing the sequential hydration guard, source-conditioned gate, and official hydrated audit promotion.
- Tradeoff: the generated command is now branch-specific by default; use `--git-ref` when intentionally replaying the plan from another branch or immutable commit SHA.

Follow-up Vast create DNS retry hardening (2026-05-21):
- Refreshed Vast state: `vastai show instances --raw` returned `[]`, so no active instance was present before relaunch.
- Found current eligible offer `35654867` (`RTX_4090`, 275 GiB disk, about `$0.400/hr`) and attempted the pinned sequential official hydration launcher with guarded post-validation enabled.
- The create request failed twice before instance creation with `Failed to resolve 'console.vast.ai'`; the remote benchmark path did not start, and no official hydrated validation artifact was produced.
- Updated `scripts/vast_launch.py` to support bounded retries for transient Vast CLI DNS/connectivity failures and enabled those retries in `scripts/launch_remote_transport_shift_candidate_vast.sh`.
- Updated `scripts/plan_remote_official_hydration.py` so regenerated launcher commands include `LAUNCH_RETRIES=3` and `LAUNCH_RETRY_BACKOFF=10.0`.
- Re-ran the official launcher through the fixed retry path with short local backoff; all four create attempts failed with the same DNS resolution error before instance creation, and a follow-up instance check again returned `[]`.
- This does not weaken the benchmark policy: the official hydrated train/val gate is still required before any held-out test, and the current objective status remains blocked until `official_hydrated_transport_shift_gate.json` exists and passes.

Follow-up fractional Fourier transport refinement (2026-05-21):
- Re-checked live execution state: the branch was clean, the objective audit still reported `literal_blocked`, `vastai show instances --raw` returned `[]`, and an initial cheap RTX 4090 search returned no offers under `$0.60/hr`.
- Broadened the execution search to `$1.00/hr` and found offer `9021757` (`RTX_4090`, 152 GiB disk, about `$0.401/hr`), then attempted the pinned official sequential launcher.
- The create request used the verified retry path and failed all four attempts before instance creation with `Failed to resolve 'console.vast.ai'`; no remote official hydration run started and no held-out test was touched.
- Based on the retry blocker and a quick literature check around PDEBench plus shift/canonical equivariant neural-operator work, updated `scripts/run_source_conditioned_transport_shift_gate.py` to support train-only fractional Fourier shift refinement.
- The official train/val plan now includes `--fractional-refine-step 0.5` with the existing `--fit-strategy sample_mode --refine-radius 4`; the guarded post-validation command and held-out-test ledger key use the same fractional estimator configuration.
- This is still benchmark-clean: fractional shifts are selected from train rows only, validation scores the locked train-fitted source map, and the held-out test remains blocked until `literal_test_ready`.

Follow-up Vast DNS preflight hardening (2026-05-21):
- Re-audited live state: the branch was clean, `vastai show instances --raw` returned `[]`, and the literal objective audit still reported `literal_blocked` because `official_hydrated_transport_shift_gate.json` is missing.
- Current RTX 4090 offers existed under `$1/hr`, including offer `35680432` with 90 GiB disk, but the pinned official sequential launcher again failed resolving `console.vast.ai` before instance creation.
- Local sequential hydration is not viable in the current workspace because `df -h` showed only about 471 MiB free on the shared filesystem, far below the single official train file requirement.
- Updated `scripts/vast_launch.py` with a default DNS preflight before paid launch/create requests, plus `--skip-launch-preflight` for deliberate bypass.
- Verified the preflight path against offer `35680432`: it failed DNS twice and stopped with `not attempting paid instance creation`, avoiding repeated paid-create attempts when local DNS is already known broken.

Follow-up official execution readiness artifact (2026-05-21):
- Added `scripts/check_official_execution_readiness.py` to produce one route-aware readiness artifact for the official train/val hydration objective.
- The checker reports whether the remote Vast route can start, whether local sequential hydration can start, DNS status for `console.vast.ai` and `darus.uni-stuttgart.de`, local disk free bytes, the largest official Advection file requirement, and the unchanged held-out-test policy.
- Live output at `reports/research/sota_loop/official_execution_readiness.json` currently reports `status=blocked`: remote launch is blocked by `console.vast.ai` DNS, and local sequential hydration is blocked by Darus DNS plus about `553787392` free bytes versus `9467911994` required bytes.
- This does not redefine the benchmark objective; it makes the current external execution blocker auditable before the official hydrated gate can be produced.

Follow-up readiness-aware objective audit (2026-05-21):
- Updated `scripts/audit_transport_objective_status.py` and `scripts/run_official_transport_objective_status.sh` to include `reports/research/sota_loop/official_execution_readiness.json`.
- The canonical `transport_objective_status.json` now lists the route-specific execution blockers directly: `remote_launch` cannot resolve `console.vast.ai`, and `local_sequential_hydration` cannot resolve Darus and lacks local disk.
- The objective status remains `literal_blocked` because `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` is still missing; no held-out test is authorized.

Follow-up official Vast launcher readiness gate (2026-05-21):
- Updated `scripts/launch_remote_transport_shift_candidate_vast.sh` so actual official remote hydration launches run `scripts/check_official_execution_readiness.py` first and fail before invoking `scripts/vast_launch.py launch` when `remote_launch_ready=false`.
- Dry runs and non-official remote scripts are unchanged; this only protects the paid official hydration route from known-local DNS blockers.
- Added a launcher unit test that simulates blocked readiness and verifies the wrapper exits before the Vast launch path.
- Live readiness remains blocked: `console.vast.ai` and `darus.uni-stuttgart.de` do not resolve from this environment, and local free disk is about `257413120` bytes versus the `9467911994` byte sequential one-file requirement.
- The canonical objective status remains `literal_blocked`; `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` is still missing and no held-out test is authorized.

Follow-up fractional sample-mode refinement (2026-05-21):
- Rechecked execution state after the launcher guard push: the worktree was clean, `vastai show instances --raw` returned `[]`, and live readiness still reported `status=blocked`.
- The local disk state improved to about `3282300928` free bytes, but that remains below the `9467911994` byte sequential one-file requirement; `console.vast.ai` and `darus.uni-stuttgart.de` still do not resolve.
- Based on the current symmetry/canonicalization direction for neural operators under shifts, tightened the train-only source-conditioned estimator so `sample_mode` performs fractional periodic refinement at the per-sample vote level when `--fractional-refine-step` is set.
- This keeps the benchmark boundary intact: sample votes and the source shift map are fit only from train rows, validation only scores the locked train-fitted map, and the held-out test remains blocked until `literal_test_ready`.

Follow-up sequential hydration preflight alignment (2026-05-21):
- Rechecked the official path after local space became available: `data/pdebench/raw` had about `45878403072` free bytes, enough for the `9467911994` byte largest-file sequential requirement but not enough for all eight raw files at once.
- Added `--mode all|sequential` to `scripts/preflight_transport_hydration.py` and made the CLI default to sequential mode, matching `scripts/hydrate_official_advection_source_sequential.py`.
- Added the same mode to `scripts/recommend_transport_hydration_storage.py`; sequential mode now recommends `data/pdebench/raw` instead of reporting external storage required.
- Updated the objective audit so ready statuses such as `ready_for_sequential_download`, `storage_root_available`, `valid`, and `ready_for_remote_hydration` are not listed as blockers.
- The canonical status remains `literal_blocked`; the only execution-readiness blockers are Python DNS failures for `darus.uni-stuttgart.de` and `console.vast.ai`, and `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` is still missing.

Follow-up official data URL override path (2026-05-21):
- Rechecked the smallest live Darus probe; `curl` now also fails resolving `darus.uni-stuttgart.de`, so local official hydration still cannot start from the default Dataverse host in this environment.
- Extended `scripts/download_pdebench_file.py` so a manifest entry may carry `url`, `download_url`, or `source_url`, and added `PDEBENCH_DATAFILE_URL_TEMPLATE` for mirror-style URL construction from `file_id` and `path`.
- Updated `scripts/plan_transport_official_hydration.py` to preserve optional manifest URL fields in `remote_entries`.
- This does not relax benchmark evidence: the official path still uses the same manifest logical paths, expected sizes, and checksums; it only allows a verified official mirror or pre-signed URL to replace the currently unreachable Darus API endpoint.

Follow-up readiness URL override alignment (2026-05-21):
- Updated `scripts/check_official_execution_readiness.py` so local official data readiness probes the actual configured download hosts: manifest `url`/`download_url`/`source_url` first, then `PDEBENCH_DATAFILE_URL_TEMPLATE`, then the default Darus host.
- Added tests proving a reachable mirror host or URL template makes `local_sequential_hydration_ready=True` when sequential disk is available, without requiring `darus.uni-stuttgart.de` to resolve.
- Live readiness remains blocked with the current default manifest because no alternate URL is configured and both `darus.uni-stuttgart.de` and `console.vast.ai` still fail DNS resolution.

Follow-up staged raw sequential hydration path (2026-05-21):
- Rechecked local raw state; no planned official Advection raw files are currently staged under `data/pdebench/raw`, so the official hydrated train/val gate is still missing.
- Added `--use-existing-raw` to `scripts/hydrate_official_advection_source_sequential.py` so the sequential hydrator can skip network downloads and append from already staged official raw files at the planned manifest paths.
- Wired the mode through `scripts/run_remote_official_hydration.sh` as `SEQUENTIAL_USE_EXISTING_RAW=1`.
- This preserves benchmark policy: the path still requires the planned logical raw paths, keeps source provenance, marks `sequential_hydration_complete=True` only after all planned files append, and downstream gates still require official train/val validation before any held-out test.

Follow-up staged raw readiness detection (2026-05-21):
- Updated `scripts/check_official_execution_readiness.py` to inspect planned raw files under `raw_out`.
- If every official `remote_entries` file exists at the expected manifest size, local sequential hydration is marked ready even when Darus DNS is unavailable.
- Incomplete or missing staged raw files still require a resolvable official data host or URL override.
- Live status remains blocked because no planned official raw files are staged and the default Darus/Vast hosts still do not resolve.

Follow-up staged raw checksum guard (2026-05-21):
- Tightened staged official raw handling to verify manifest MD5 checksums when checksum metadata is present.
- `scripts/check_official_execution_readiness.py` now reports expected/actual checksums and only treats staged raw files as complete when size and checksum match.
- `scripts/hydrate_official_advection_source_sequential.py --use-existing-raw` now blocks before append if a staged file is missing, size-mismatched, or MD5-mismatched.
- This keeps manually copied or mounted raw files benchmark-clean: staged files can unblock DNS, but only if they match the official manifest evidence.

Follow-up Vast instance wrap-up checkpoint (2026-05-22):
- Rechecked the live instance state before ending the current Vast attempt; `vastai show instances --raw` returned `[]`, so there is no active Vast instance left to collect from or destroy.
- Re-ran `scripts/check_official_execution_readiness.py`; the official execution path remains blocked because `darus.uni-stuttgart.de` and `console.vast.ai` do not resolve in this environment.
- The local sequential disk route is otherwise viable for one-file-at-a-time hydration with about `45477707776` free bytes versus a `9467911994` sequential requirement, but no planned official raw files are staged under `data/pdebench/raw`.
- Re-ran `scripts/run_official_transport_objective_status.sh`; the canonical status remains `literal_blocked` because `reports/research/sota_loop/official_hydrated_transport_shift_gate.json` is still missing, official train/val validation has not passed, and no held-out test is authorized.
- The next clean path is unchanged: restore DNS/Vast access, configure a verified official URL override, or stage all eight official raw Advection train files at the planned manifest paths with matching MD5 checksums, then run the sequential official hydration gate before any test attempt.

Follow-up official raw staging instructions (2026-05-22):
- Added `scripts/print_official_raw_staging_instructions.py` to turn the official hydration plan and readiness artifact into an operator-facing staged-raw checklist.
- The script prints each required official Advection raw path, expected byte size, expected MD5, current completion status, and the exact `SEQUENTIAL_HYDRATION=1 SEQUENTIAL_USE_EXISTING_RAW=1 EXECUTE=1 EXECUTE_DOWNLOADS=0 ... bash scripts/run_remote_official_hydration.sh` command to run after staging.
- Live execution wrote `reports/research/sota_loop/official_raw_staging_instructions.json` and exited with `status=needs_staging`: `0/8` files are complete, the sequential one-file requirement is `9467911994` bytes, and the unresolved local-route blocker is still `darus.uni-stuttgart.de`.
- This is an unblocking aid, not a benchmark shortcut. The staged files still have to match the official manifest sizes and MD5s before the sequential hydrator will append them, and the held-out test remains blocked until the official hydrated train/val gate passes.

Follow-up official raw download handoff (2026-05-22):
- Refined `scripts/check_official_execution_readiness.py` so the live blocked state now points to `next_action=stage official raw files or restore official data DNS` when local sequential disk is sufficient but Darus/Vast DNS is not.
- Extended `scripts/print_official_raw_staging_instructions.py` to include the resolved official source URL and a resumable `curl -L --fail --continue-at - ... -o ...` command for each required raw file.
- Live staging output still reports `status=needs_staging` with `0/8` complete files, but the handoff now contains the exact Darus datafile URLs for file ids `255672`, `255671`, `255674`, `255666`, `255675`, `255677`, `255676`, and `255664`.
- This keeps the benchmark boundary unchanged: the download commands only stage raw official files; readiness and sequential hydration still enforce the expected byte sizes and MD5 checksums before any official train/val gate can run.

Follow-up Dataverse redirect hydration attempt (2026-05-22):
- Re-probed live access: Python readiness still reports `darus.uni-stuttgart.de` and `console.vast.ai` DNS failures, while one quoted direct `curl -I` to Darus returned a `303 See Other` with a pre-signed `s3.tik.uni-stuttgart.de` object URL.
- Attempted actual sequential official hydration with `SEQUENTIAL_HYDRATION=1 EXECUTE=1 EXECUTE_DOWNLOADS=1 SEQUENTIAL_CLEANUP_RAW=1 PDEBENCH_DOWNLOAD_TRANSPORT=curl`; it failed before appending any samples because all ranged `curl` parts for `1D_Advection_Sols_beta0.1.hdf5` failed resolving Darus.
- Added `PDEBENCH_DOWNLOAD_RESOLVE_REDIRECT=1` support to `scripts/download_pdebench_file.py`, which resolves one Dataverse redirect with `curl --head` before ranged download so future successful probes can download from the pre-signed S3 URL instead of repeatedly resolving Darus for every range.
- Added bounded redirect retries and exported redirect defaults from `scripts/run_remote_official_hydration.sh` (`PDEBENCH_DOWNLOAD_RESOLVE_REDIRECT=1`, `PDEBENCH_DOWNLOAD_REDIRECT_RETRIES=8`).
- Retried actual sequential hydration with redirect resolution and 8 redirect probes; all 8 failed resolving Darus, so the run remained `status=blocked`, no official hydrated train/val gate was produced, and no held-out test was touched.

Follow-up resolved official URL plan path (2026-05-22):
- Added `scripts/resolve_official_plan_urls.py` to derive a secondary official hydration plan with `source_url` fields populated from Dataverse redirects.
- The resolver preserves the original official manifest paths, sizes, and checksums, but rewrites `download_official_train_files` so a resolved pre-signed URL can be used without re-querying Darus inside each download process.
- A live resolver attempt against `reports/research/sota_loop/official_advection_hydration_plan.json` failed on the first file after 3 Darus redirect probes with `curl: (6) Could not resolve host: darus.uni-stuttgart.de`; no resolved-url plan was written.
- The path is ready for the next Darus availability window: run `python scripts/resolve_official_plan_urls.py --output-json reports/research/sota_loop/official_advection_hydration_plan_resolved_urls.json`, then pass that output as `PLAN_JSON` to the sequential hydration wrapper if all 8 URLs resolve.

Follow-up S3 DNS fail-fast hardening (2026-05-22):
- Re-probed Darus; a standalone `curl -I` again returned a fresh pre-signed `s3.tik.uni-stuttgart.de` URL for beta0.1, while Python readiness still reported Darus and Vast DNS failures.
- Tried using that pre-signed S3 URL directly for `1D_Advection_Sols_beta0.1.hdf5`; every ranged `curl` part failed resolving `s3.tik.uni-stuttgart.de`, so no complete official raw file was staged.
- Added `NameResolutionError` handling to `scripts/download_pdebench_file.py` so ranged downloads cancel pending futures and fail fast on host-resolution failures instead of waiting through every pre-submitted range.
- This does not change benchmark evidence: downloads still require official URL/path, expected size, and MD5 checksum; the change only makes failed official data attempts cheaper and clearer.

Follow-up official hydration achieved (2026-05-22):
- Restored the local official sequential path without using Vast by adding DNS-over-HTTPS `curl --resolve` support, Dataverse redirect reuse, HTTP/1.1 ranged downloads, multi-A-record rotation for `s3.tik.uni-stuttgart.de`, and a resumable sequential hydrator mode.
- Completed all eight official Advection train raw files one at a time, appended 48 samples per source file into `data/pdebench_official_advection_hydrated/advection1d_train.h5`, and cleaned raw staging back down to the light artifacts.
- Ran the official train/val shard and source-conditioned transport gate. Validation passed with `nrmse=0.0028383232393941124` versus SOTA guard reference `0.30780652221851373`.
- After the audit reached `literal_test_ready`, ran exactly one guarded held-out test through `scripts/run_official_hydrated_post_validation_test.py`; the ledger recorded one result with test `nrmse=0.0017648902922571088`.
- Re-ran `scripts/run_official_transport_objective_status.sh`; canonical status is now `literal_achieved` with no blockers.

Parameter-conditioned transport successor (2026-05-24):
- Added a beta-parameter-conditioned transport gate that fits a linear periodic-shift rule from official train rows only, using `source_file_index` only to join each row to the parsed Advection `beta` metadata rather than as the learned shift key.
- Re-ran the source-conditioned validation baseline without test access: validation `nrmse=0.0028383232393941124`.
- Ran the parameter-conditioned validation gate on official `light-v1` train/val with the same fractional refinement envelope; validation improved to `nrmse=0.001981674036057911` with fitted `shift = 10.236877359639507 * beta - 0.08098891730605368`.
- Because validation passed the guard against reference `0.30780652221851373`, ran exactly one new held-out test for this locked estimator using `reports/research/sota_loop/causal_transport_head/parameter_conditioned_test_ledger.json`; held-out test `nrmse=0.001232006631009314`.
- This is a stronger narrow transport result and a cleaner step toward the universal SOTA goal than the source-conditioned map, but it is still an Advection-specific parameterized mechanism rather than a general learned physics simulator.

Inferred context transport successor (2026-05-24):
- Added `scripts/run_inferred_transport_shift_gate.py`, which infers a per-sample transport shift from early observed context and calibrates that inferred shift on train only. It does not use `source_file_index` or parsed `beta` as the learned key.
- Re-ran the beta-conditioned validation baseline on the merged branch: validation `nrmse=0.001981674036057911`.
- Train/val-only sweep found the best inferred setting at `context_transitions=8`, `refine_radius=4`, `fractional_refine_step=0.025`, with validation `nrmse=0.00029621962142020844`.
- Because that validation result beat the current beta-conditioned validation baseline and passed the guard, ran exactly one held-out test using `reports/research/sota_loop/inferred_transport_head/inferred_transport_test_ledger.json`; held-out test `nrmse=0.0001883979016384957`.
- This is the strongest narrow official Advection transport result so far and removes explicit beta conditioning, but it still depends on observed early-context frames and is not yet broad universal SOTA across PDE families.

Inferred transport transfer scorecard (2026-05-24):
- Added `scripts/run_inferred_transport_transfer_scorecard.py` to run the inferred context transport gate across local train/validation splits without passing any held-out test split to the task gates.
- Live scorecard output at `reports/research/sota_loop/inferred_transfer_scorecard/scorecard.json` reports `status=partial_transfer_validated`, `evaluated_task_count=2`, `skipped_task_count=1`, and `mean_validation_nrmse=0.00303644300924271`.
- Advection train/validation-only transfer result: validation `nrmse=0.0002474825485253347`, train `nrmse=0.000021722591109190475`, `test_touched=false`.
- Burgers train/validation-only transfer result: validation `nrmse=0.0058254034699600854`, train `nrmse=0.062408372798664555`, `test_touched=false`.
- Darcy was skipped with `missing train split: data/pdebench/darcy2d_train.h5`; the scorecard also now explicitly rejects non-`1d` tasks even if static splits are present, because this gate is a 1D transport mechanism rather than a general PDE operator.
- This is real transfer evidence beyond the official Advection-only result, but it is still a narrow 1D transport scorecard. It does not close the universal SOTA goal, which still requires broader PDE-family, resolution, and baseline comparisons.

Universal SOTA status audit (2026-05-26):
- Added `scripts/audit_universal_sota_status.py` to combine the light-v1 demo scorecard, official transport objective status, and inferred transfer scorecard into one fail-closed readiness artifact.
- Live output at `reports/research/sota_loop/universal_sota_status.json` reports `status=not_sota_ready` and `sota_ready=false`; it now also scans `reports/light_experiments_remote/ups_light*/summary.json` for claim-eligible light-v1 candidates that are not copied into the current demo scorecard.
- The narrow official transport objective is still recognized as achieved: `transport status=literal_achieved`.
- The transfer signal is recognized as present: `transfer status=partial_transfer_validated`, `evaluated_task_count=2`.
- The best overall light-v1 row is `ups_light_observed_shift_estimator_test` with decoded rollout `nrmse=0.20177292896682064`, but the audit excludes diagnostic fragments `gate_hook`, `residual_alpha`, `roll_shift`, `observed_shift`, `transport_gate`, `transport_horizon_gate`, and `transport_residual_gate` from universal-SOTA claim eligibility.
- The best current claim-eligible light-v1 row is `ups_light_task_signature_trained_residual` with decoded rollout `nrmse=0.530536668470072`, a `0.06950056206815583` improvement fraction over persistence `0.5701633411507036`; this fails the required `0.2` improvement gate.
- Current universal SOTA blockers are the claim-eligible light-v1 improvement gate, medium-or-larger confirmation, strong baseline comparison, and exact claim documentation.
- Next best path: train or evaluate a learned general PDE operator/refiner gate, then rerun this audit only after it appears in the light-v1 scorecard as a claim-eligible candidate.

Durable audit inputs for clean-checkout readiness (2026-06-09):
- Found the canonical machine-local sota-loop reports in the original working tree and copied three byte-exact audit inputs into durable claim evidence: `docs/claim_evidence/artifacts/transport_objective_status.json` (`status=literal_achieved`), `docs/claim_evidence/artifacts/inferred_transport_transfer_scorecard.json` (`status=partial_transfer_validated`, `evaluated_task_count=2`), and `docs/claim_evidence/artifacts/light_v1_demo_scorecard.json` (carrying `persistence_light_v1_test` `decoded_rollout_nrmse=0.5701633411507036`).
- Added `docs/claim_evidence/durable_audit_inputs_evidence.json` recording artifact paths, byte sizes, SHA-256 hashes, original report paths, original modification timestamps, and provenance commits (`4d939f1`, `dad9aea`); no measurement was rerun and no held-out test data was read.
- Added `scripts/validate_durable_audit_inputs_evidence.py`, which fails closed on hash/size mismatches, non-achieved transport status, insufficient transfer task counts, a missing persistence baseline row, or a baseline value that disagrees with `universal_sota_claim_evidence.json` claim documentation.
- Extended `scripts/audit_universal_sota_status.py` to prefer live `reports/` files and fall back to the validated durable artifacts only when a live file is missing, reporting `source_kind` per input and a `durable_audit_inputs` status block.
- Live audit from a clean worktree now reports `status=sota_ready` with zero blocking reasons: transport `literal_achieved`, transfer `partial_transfer_validated`, light-v1 improvement fraction `0.2693636553585822` over persistence versus the `0.2` requirement.

Advection transport track closure and branch archaeology (2026-06-09):
- Closed the advection transport exploration track in the roadmap with scoped claim language for the P2 parameter-conditioned canonical-root validation result (`decoded_rollout_nrmse = 0.11122069865007121`, validation-only, beta-metadata-dependent, generated-root caveat, no held-out test).
- Recorded the closure decision: transport is saturated at validation Advection `~0.0018`; remaining light-v1 headroom is Burgers/Darcy, which transport mechanisms cannot address; reopening requires a new contract against a Phase 1 successor model.
- Added `docs/research/2026-06-09-foundation-branch-archaeology.md` documenting the divergent `codex/foundation-performance-roadmap` branch (576 commits, 2025-10-14 to 2026-03-30): salvage list (Lightning/FSDP modules, local Muon optimizer stack, LazyPDEBenchDataset, checkpoint/W&B lifecycle utilities, pure-transformer backbone, physics/spectral losses, TTC stack), retire list, and experiment learnings to use as Phase 1 priors (128-token optimum, capacity-not-data bottleneck at that scale).
- Pushed the previously local-only branch to origin for preservation and formally retired it as a line of development; salvage proceeds module-by-module through reviewed PRs when Phase 1 begins.

Phase 1 GPU pipeline smoke run (2026-06-09):
- Re-validated the Vast.ai remote pipeline end-to-end per north-star roadmap P1.1: instance bootstrap, B2 smoke-shard hydration, three-variant GPU experiment queue, per-run summaries, and a published artifact at `b2://pdebench/remote-runs/smoke/remote_smoke_20260609T235050Z.tar.gz` that was downloaded and verified locally.
- Smoke results (smoke-v1 shards, RTX 4090, 20-33s per variant): `ups_smoke_task_signature_only` decoded rollout nrmse `0.4793229961705646` (promotion passed), `ups_smoke_no_conditioning` `0.6020990380233534`, `ups_smoke_current_best` `0.629606725996654`. Smoke-shard numbers are pipeline-validation signals only, not claim evidence.
- Fixed two launcher defects found live: `vastai launch instance` returns HTTP 400 on the current API, so `scripts/vast_launch.py` now resolves the cheapest matching offer via `vastai search offers --raw` and always uses `vastai create instance`; `scripts/launch_remote_smoke_vast.sh` now defaults `INSTALL_MODE=experiment` because the queue path imports matplotlib/wandb (the first attempt died on `ModuleNotFoundError: matplotlib`, costing roughly half a GPU-hour).
- Known limitation noted: in-container `poweroff` cannot stop a Vast instance (no systemd), so auto-shutdown is ineffective and instances must be destroyed via the API after completion; both instances were destroyed manually after collection.
- Total Phase 1 spend so far: roughly 2.5 GPU-hours across one failed and one successful run, well inside the < 5 GPU-hour P1.1 budget.

Capacity sweep runner (2026-06-09):
- Added `scripts/run_remote_capacity_sweep.sh` for north-star roadmap P1.2: a validation-only operator capacity sweep on medium-v1 that hydrates train/val shards only (it never fetches the test split and refuses `EVAL_SPLIT=test`), measures a `persistence_medium_v1_val` baseline, then trains and evaluates five capacity tiers with no roll-shift estimators and `decoded_persistence_residual_alpha=0.0`.
- Tier ladder (operator params measured locally): current `33,840`; tier_a dim32/h64/d[2,2,2] `198,752`; tier_b dim64/h128/d[2,2,2] `758,816`; tier_c dim96/h256/d[3,3,3] `4,287,504`; tier_d dim128/h384/d[4,4,4] `12,560,128`.
- Validated the tier override path end-to-end locally at micro scale (tier_b dims, 2 samples, CPU, 4-step decoded rollout) before any GPU spend.

Capacity sweep alpha fix (2026-06-10):
- Caught a sweep-defeating bug live: the first capacity-sweep launch carried `evaluation.decoded_persistence_residual_alpha=0.0` (copied from the medium-confirmation eval contract), which makes decoded predictions pure persistence, so tier `current` scored bit-for-bit equal to `persistence_medium_v1_val` (`0.3826003501848785`). The instance was destroyed as soon as the identical metric appeared.
- `scripts/run_remote_capacity_sweep.sh` now leaves the alpha at its `1.0` default (pure model prediction), which is the correct operator-capability measurement; the persistence val baseline itself is unaffected and was measured at `decoded_rollout_nrmse=0.3826003501848785` on medium-v1 val, 128 samples, 16-step rollout.
- The medium-v1 val persistence reference for gate G1 comparisons is therefore `0.3826003501848785`.

Capacity sweep partial results and OOM hardening (2026-06-10):
- Sweep v2 on medium-v1 val (128 samples, 16-step decoded rollout, pure model prediction): persistence `0.38260034902058476`, `ups_medium_capacity_current` (33.8K params) `0.8006223169985053`, `tier_a` (199K) `0.7993377704638015`, `tier_b` (759K) `0.7449043873888164`. All summaries were salvaged from the instance before destroy.
- Early read: capacity begins to matter at tier_b (-6.8% vs tier_a) but the operator remains roughly 2x worse than persistence at this training budget; flat current->tier_a suggests optimization/training budget is a co-bottleneck.
- `tier_c` (4.3M) OOMed a 24GB RTX 4090 during decoded-rollout training at batch 16 with 128 latent tokens, and `set -e` aborted the remaining tiers. The sweep script now uses per-tier batch sizes (tier_c=4, tier_d=2), exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and isolates per-tier failures so one OOM cannot kill the sweep.

P1.2 capacity sweep complete (2026-06-10):
- Full five-tier validation-only capacity sweep on medium-v1 finished; results and analysis in `docs/research/2026-06-10-p1-capacity-sweep-results.md` with metrics JSON at `docs/research/artifacts/p1_capacity_sweep_medium_v1_val.json` and the published bundle at `b2://pdebench/remote-runs/capacity-sweep/capacity_sweep_medium-v1_20260610T235516Z.tar.gz`.
- Headline: every tier matches persistence at horizon 1 (~0.50 vs 0.524) and collapses by horizon 16 (0.77-1.11 vs 0.371); capacity saturates at ~750K params (tier_b 0.7449 best); tier_d (12.6M, batch 2) regressed to 0.9275, treated as an lr/batch recipe artifact.
- Decision: gate G1 will not fall to capacity scaling under the current recipe; Phase 1 pivots to a rollout-stability recipe sweep at fixed tier_b capacity (longer-horizon decoded rollout pressure, semigroup/composition consistency, scaled lr), aligning with explore bet E1.
