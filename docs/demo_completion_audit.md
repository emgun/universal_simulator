# UPS Demo Completion Audit

This audit tracks whether the working-demo/SOTA roadmap has reached a runnable
light-experiment state. It should be updated from real artifacts, not intent.

## Objective

Iterate toward a working UPS demo with good held-out results, preserving
progress and learnings in repo artifacts.

## Success Criteria

- Smoke shards are published to B2 under `smoke-v1`.
- Smoke experiment queue can run remotely against `smoke-v1`.
- Smoke summaries are copied back under `reports/light_experiments_remote/`.
- Held-out light shards are published to B2 under `light-v1`.
- Matched persistence baseline summary exists for `light-v1`.
- UPS candidate summary exists for `light-v1`.
- `reports/demo/latest/index.html`, `metrics.tsv`, `scorecard.json`, plots,
  copied summaries, and cost fields are generated from real summaries.
- UPS candidate beats persistence on held-out `decoded_rollout_nrmse` by the
  configured baseline gate before any scale-up or SOTA-style claim.

## Current Evidence

- Existing real-data artifact:
  - `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`
  - `decoded_rollout_nrmse = 0.9488316819858322`
  - Status: plumbing smoke only, not benchmark evidence.
- Remote smoke-v1 experiment artifact:
  - B2 tarball:
    `remote-runs/smoke/smoke_current_best_20260505T0600Z.tar.gz`
  - local summary:
    `reports/light_experiments_remote/ups_smoke_current_best/summary.json`
  - local scorecard: `reports/demo/smoke_latest/scorecard.json`
  - `decoded_rollout_nrmse = 0.6297059754071941`
  - Status: three-task smoke gate passed; still not benchmark evidence because
    smoke shards are tiny and partly derived from shortcut splits.
- Matched smoke persistence baseline:
  - B2 tarball:
    `remote-runs/smoke/persistence_smoke_v1_test_20260505T0615Z.tar.gz`
  - local summary:
    `reports/light_experiments_remote/persistence_smoke_v1_test/summary.json`
  - `decoded_rollout_nrmse = 0.1876487120420463`
  - Baseline comparison: `ups_smoke_current_best` fails the baseline
    improvement gate (`baseline_improvement_passed=false`) because its rollout
    NRMSE is higher than persistence.
- Remote smoke-v1 variant matrix:
  - B2 tarball:
    `remote-runs/smoke/smoke_variants_20260505T0625Z.tar.gz`
  - local summaries:
    `reports/light_experiments_remote/ups_smoke_no_conditioning/summary.json`,
    `reports/light_experiments_remote/ups_smoke_task_signature_only/summary.json`,
    `reports/light_experiments_remote/ups_smoke_semigroup0/summary.json`, and
    `reports/light_experiments_remote/ups_smoke_semigroup10/summary.json`
  - local scorecard: `reports/demo/smoke_latest/scorecard.json`
  - best UPS variant: `ups_smoke_task_signature_only` with
    `decoded_rollout_nrmse = 0.4793234406026068`
  - Interpretation: task-signature-only conditioning improves over
    `ups_smoke_current_best` (`0.6297059754071941`) but still fails the matched
    persistence baseline (`0.1876487120420463`), so this is a useful direction
    for cheap iteration but not a scale-up candidate yet.
- Live B2 readiness:
  - `smoke-v1`: `9/9` expected keys present after remote args-mode shard prep
    on 2026-05-05 UTC.
  - `light-v1`: `0/9` expected keys present.
- Local machine readiness:
  - local filesystem has about `3.0 GiB` free.
  - optimized smoke shard prep needs roughly `10-12 GiB` scratch plus output
    room, so shard prep should not run locally.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Preserve branch/work progress | `worklog.md`, `docs/demo_runbook.md` | Done |
| Avoid local training | no local training launched; only tests/dry-runs ran | Done |
| Use B2-backed data | B2 readiness and shard-prep scripts use `.env` and `rclone` | Done for smoke, light pending |
| Publish smoke shards | `reports/demo/smoke_readiness_after_remote.json`, live B2 check shows `9/9` keys | Done |
| Run smoke experiments | `ups_smoke_current_best`, four variant summaries, B2 artifact tarballs, local smoke scorecard | Done for smoke |
| Find cheap remote box | `scripts/search_vast_smoke_offers.py`, `scripts/launch_remote_smoke_vast.sh`, optional `OFFER_ID` pinning | Done for smoke prep |
| Publish light shards | `docs/demo_data_manifest.yaml`, `scripts/run_remote_shard_prep_b2.sh` | Missing remote run |
| Run persistence baseline | `persistence_smoke_v1_test`, B2 artifact tarball, local smoke scorecard | Done for smoke, light pending |
| Run UPS candidate | `ups_smoke_task_signature_only` is best smoke UPS row; baseline comparison columns show persistence still wins | Done for smoke, light pending |
| Build demo report | `reports/demo/smoke_latest/index.html`, `metrics.tsv`, `scorecard.json` | Done for smoke with baseline, light pending |
| Make performance claim | baseline-delta scorecard fields | Blocked on held-out results |

## Next Command On Remote

Run on a remote/data-prep box with at least 12 GiB scratch for smoke prep:

```bash
DRY_RUN=0 \
ENV_FILE=/workspace/.env \
PIPELINE_ROOT=reports/demo/remote_smoke_pipeline \
bash scripts/run_remote_smoke_pipeline.sh
```

Dry-run Vast launch wrapper:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
GIT_REF=codex/vast-no-apt-onstart \
DISK_GB=32 \
ORDER=dph_total \
LIMIT=10 \
SSH=0 \
ARGS_MODE=1 \
INSTALL_MODE=smoke \
bash scripts/launch_remote_smoke_vast.sh
```

Pin a reviewed Vast offer ID instead of re-running implicit launch search:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
GIT_REF=codex/vast-no-apt-onstart \
DISK_GB=32 \
OFFER_ID=<offer_id_from_search> \
SSH=0 \
ARGS_MODE=1 \
INSTALL_MODE=smoke \
bash scripts/launch_remote_smoke_vast.sh
```

Current offer snapshot command:

```bash
python scripts/search_vast_smoke_offers.py \
  --limit 10 \
  --output-json reports/demo/vast_smoke_offers.json \
  --output-tsv reports/demo/vast_smoke_offers.tsv
```

After reviewing the generated queue:

```bash
RUN_EXPERIMENTS=1 \
QUEUE_DRY_RUN=0 \
ENV_FILE=/workspace/.env \
PIPELINE_ROOT=reports/demo/remote_smoke_pipeline \
bash scripts/run_remote_smoke_pipeline.sh
```

The pipeline refuses live smoke experiment execution when `CHECK_B2=1` and
`smoke-v1` shards are not ready.
It also refuses unchecked live queue execution when `CHECK_B2=0` unless
`ALLOW_UNCHECKED_LIVE_QUEUE=1` is explicitly set for a controlled test
environment.

Vast launch note: use `codex/vast-no-apt-onstart` or later for remote launches
with `SSH=0 ARGS_MODE=1 INSTALL_MODE=smoke` when running a one-shot smoke
pipeline. Earlier launch attempts reached paid instances but stalled in
Vast/Ubuntu apt setup before repo checkout, then a full dev install started
pulling a large replacement Torch/CUDA stack.
For one cheap smoke experiment, use `INSTALL_MODE=experiment` and
`EXTRA_PIPELINE_ARGS="PREP_SHARDS=0 RUN_EXPERIMENTS=1 QUEUE_DRY_RUN=0
QUEUE_VARIANTS=current_best"` so the launcher does not republish shards or run
the full variant matrix.
Add `PUBLISH_PIPELINE_ARTIFACTS=1 PIPELINE_ARTIFACT_NAME=<name>.tar.gz` to that
same argument string for no-SSH runs so summaries/logs are uploaded to B2 before
the instance is destroyed.

Before tearing down the remote box:

```bash
OUTPUT=reports/demo/demo_artifacts.tar.gz \
bash scripts/package_demo_artifacts.sh
```

## Stop Conditions

- Do not run shard prep on a machine with less than required scratch space.
- Do not run full-data hydration without explicit approval.
- Do not treat smoke metrics as benchmark evidence.
- Do not make SOTA-style claims until held-out light/medium artifacts and
  matched baseline deltas exist.
