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
- Live B2 readiness:
  - `smoke-v1`: `0/9` expected keys present.
  - `light-v1`: `0/9` expected keys present.
- Local machine readiness:
  - local filesystem has about `1.9 GiB` free.
  - optimized smoke shard prep needs roughly `10-12 GiB` scratch plus output
    room, so smoke prep should not run locally.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Preserve branch/work progress | `worklog.md`, `docs/demo_runbook.md` | Done |
| Avoid local training | no local training launched; only tests/dry-runs ran | Done |
| Use B2-backed data | B2 readiness and shard-prep scripts use `.env` and `rclone` | Ready, not complete |
| Publish smoke shards | `docs/demo_smoke_data_manifest.yaml`, `scripts/run_smoke_shard_prep_b2.sh` | Missing remote run |
| Run smoke experiments | `scripts/run_remote_smoke_pipeline.sh`, live queue requires `CHECK_B2=1` and ready shards | Missing remote run |
| Publish light shards | `docs/demo_data_manifest.yaml`, `scripts/run_remote_shard_prep_b2.sh` | Missing remote run |
| Run persistence baseline | `scripts/run_persistence_baseline.py` | Missing summary |
| Run UPS candidate | `scripts/run_remote_light_promotion.sh`, queue planner | Missing summary |
| Build demo report | `scripts/build_demo_report.py` | Blocked on summaries |
| Make performance claim | baseline-delta scorecard fields | Blocked on held-out results |

## Next Command On Remote

Run on a remote/data-prep box with at least 12 GiB scratch for smoke prep:

```bash
DRY_RUN=0 \
ENV_FILE=/workspace/.env \
PIPELINE_ROOT=reports/demo/remote_smoke_pipeline \
bash scripts/run_remote_smoke_pipeline.sh
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
