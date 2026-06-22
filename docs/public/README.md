# Public Overview

Universal Physics Stack (UPS) is research software for latent-space neural
simulation of PDE-style physical systems. It combines grid and field encoders,
latent transformer operators, any-point decoders, rollout evaluation, and
reproducible result artifacts.

## Current Scope

UPS currently reports bounded PDEBench-shaped `light-v1` results:

- protocol family: `light-v1` / bounded PDEBench-shaped tasks
- representative tasks: `advection1d`, `burgers1d`, `darcy2d`
- primary metric: decoded physical-space rollout NRMSE
- result records: `docs/claim_evidence/`
- latest machine-readable record: `docs/claim_evidence/universal_sota_claim_evidence.json`

Results outside this protocol need their own split, command, metric, and
artifact record before they should appear in public tables.

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

The current highest-signal model-side blocker is long-horizon transport and
advection phase tracking. Broader backbone or pretraining changes should be
ranked against that measured blocker.
