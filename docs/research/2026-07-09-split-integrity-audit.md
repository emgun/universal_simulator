# Split Integrity Audit: Burgers Test Is Contaminated; Darcy Is the Only Honest Split

Date: 2026-07-09

Status: protocol-level diagnostic, companion to `docs/research/2026-07-08-advection-split-regime-diagnosis.md`. Deterministic, model-free; nothing was trained, tuned, selected, or scored; no ledger writes. Test shards were read only to audit split construction, recorded truthfully (`held_out_test_data_read = true`, `candidate_scored = false`). Existing claim evidence and claim language are unchanged by this note; interpretation guidance is added below.

## Method

`scripts/audit_split_integrity.py` computes exact byte-level overlap of per-sample keys across splits: the initial-condition frame for time-dependent tasks (advection, Burgers), the full field for steady tasks (Darcy). Exact float32 equality means shared source trajectories, not coincidence.

## Result (light-v1 root `data/pdebench`, all samples)

| task | train<->val | train<->test | val<->test | verdict |
|---|---|---|---|---|
| burgers1d | 32/32 | 32/32 | 32/32 | **test fully contained in train; val and test are the same trajectories** (later frames differ only at ~1e-4, consistent with storage/precision noise) |
| advection1d | 32/32 | 32/32 | 32/32 | same initial conditions in every split; splits differ only by transport speed (beta 0.1 / 4.0 / 7.0 per the regime diagnosis) |
| darcy2d | 0/32 | 0/32 | 0/32 | clean: fully disjoint |

Artifact: `docs/research/artifacts/split_integrity_audit_light_v1.json`.

**Medium-v1 confirmed identical**: advection and Burgers both show 128/128 overlap for every split pair at 512/128/128 scale (`docs/research/artifacts/split_integrity_audit_medium_v1.json`). The shared shard-prep pipeline propagated the construction to every protocol tier.

## What each light-v1 task actually measures

- **Burgers held-out test measures training-set reproduction.** Every test trajectory was available at training time. This explains why Burgers "transferred" so well for every candidate ever measured (persistence val 0.147 vs test 0.174; every model's Burgers val ~= test): the test split is in-distribution to the point of identity.
- **Advection held-out test measures parameter extrapolation on memorized initial conditions.** Same ICs, unseen speed (beta 7 vs trained beta 0.1). IC generalization is never tested.
- **Darcy held-out test is the only honest generalization measurement in the protocol.** Notably it is also the task where every candidate's val->test behavior was most consistent.

## Interpretation guidance for existing claims

- All light-v1/medium-v1 claims remain scoped to their frozen protocol, as always. This audit documents what that protocol measures: a mixture of training-set reproduction (Burgers), parameter extrapolation (advection), and honest generalization (Darcy).
- Comparative statements (UPS vs external baselines under the same protocol) remain internally fair: every candidate saw the same contaminated splits. Absolute statements about generalization should cite Darcy or wait for strat-v1.
- The claim ledger should receive an interpretation entry referencing this audit the next time claim language is touched; no retraction is needed because no published language asserted trajectory-level held-out independence.

## Consequences for the experimentation roadmap

The 2026-07-09 universal-baseline roadmap Track A is amended (same PR as this note) to require, for `strat-v1` and all future shard builds:

1. **Trajectory-disjoint splits**: initial conditions (or full fields for steady tasks) must be exactly disjoint across train/val/test, verified by `scripts/audit_split_integrity.py` as a build gate.
2. **Regime stratification**: every split contains every regime (per the regime diagnosis).
3. **Both audits committed as build artifacts** with the shard manifests before any candidate runs.

## Provenance

- Audit script: `scripts/audit_split_integrity.py` (requires `--include-test`; honest flags; no ledger writes).
- Light-v1 artifact: `docs/research/artifacts/split_integrity_audit_light_v1.json`.
