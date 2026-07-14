# Public Overview

Universal Physics Stack (UPS) is research software for latent-space neural
simulation of PDE-style physical systems. It combines grid and field encoders,
latent transformer operators, any-point decoders, rollout evaluation, and
reproducible result artifacts.

## Current Scope

UPS currently reports bounded, legacy PDEBench-shaped `light-v1` results:

- protocol family: `light-v1` / bounded PDEBench-shaped tasks
- representative tasks: `advection1d`, `burgers1d`, `darcy2d`
- primary metric: decoded physical-space rollout NRMSE
- result records: `docs/claim_evidence/`
- latest machine-readable record: `docs/claim_evidence/universal_sota_claim_evidence.json`

Results outside this protocol need their own split, command, metric, and
artifact record before they should appear in public tables.

`light-v1` and `medium-v1` are mixed protocols rather than uniform held-out
generalization tests. Burgers test trajectories occur in training; Advection
reuses initial conditions across splits while changing transport speed; Darcy
is trajectory-disjoint. Matched-protocol comparisons remain scoped and useful,
but they do not establish broad generalization. The replacement protocol,
`strat-v1`, requires trajectory-disjoint and regime-stratified splits. Its
Advection root is complete; Burgers source hydration is the critical path, and
full-tier Darcy is blocked on duplicate-data provenance. See
`docs/research/2026-07-09-split-integrity-audit.md` and
`docs/research/2026-07-09-strat-v1-advection-root.md`.

## Where To Start

- `README.md`: top-level installation, repo map, and current scope.
- `docs/public/reproducibility.md`: how to reproduce or inspect result records.
- `docs/public/artifact_policy.md`: what belongs in Git and what belongs in
  external artifact storage.
- `docs/results/README.md`: generated benchmark figures and third-party
  baseline tables.
- `docs/claim_evidence/`: machine-readable records, pretest contracts, and
  committed compact artifact bundles.
- `docs/research/`: literature and design notes that inform future work.
- `worklog.md`: append-only operational trace, not a polished public narrative.

## Figures And Benchmarks

The generated figures render committed result records into a matched-protocol
scorecard, per-task breakdown, secondary metric suite, horizon profile,
validation-only diagnostics, and external benchmark matrix. Trace numbers back
to the machine-readable records in `docs/claim_evidence/`.

The generated cards cover cost/reproducibility, benchmark readiness, ecosystem
compatibility, and qualitative-preview status. Qualitative rollout panels render
only after a compact preview manifest and artifact are committed.

Check generated assets with:

```bash
python scripts/build_public_assets.py --check
```

## North Star

The north star is one model family that improves physical-space rollout quality
across task families while preserving strict validation/test separation and
reproducible artifacts.

The current gate is protocol integrity rather than model selection: complete
and freeze `strat-v1`, remeasure its baselines, then compare candidate families.
The scoped beta-parameter transport-head diagnostic demonstrates that explicit
regime conditioning can work, but it is not a primary or public-comparable
result.
