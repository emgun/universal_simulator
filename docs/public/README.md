# Public Overview

Universal Physics Stack (UPS) is research software for latent-space neural
simulation of PDE-style physical systems. It combines grid and field encoders,
latent transformer operators, any-point decoders, rollout evaluation, and
claim-evidence artifacts.

## Evidence Boundary

UPS currently supports a guarded research claim around bounded PDEBench-shaped
experiments. The most important public contract is:

- protocol family: `light-v1` / bounded PDEBench-shaped tasks
- representative tasks: `advection1d`, `burgers1d`, `darcy2d`
- primary metric: decoded physical-space rollout NRMSE
- evidence root: `docs/claim_evidence/`
- latest claim manifest: `docs/claim_evidence/universal_sota_claim_evidence.json`

Results outside the recorded protocol need their own evidence contract before
they become public claims.

## Where To Start

- `README.md`: top-level installation, repo map, and evidence boundary.
- `docs/public/reproducibility.md`: how to reproduce or inspect evidence.
- `docs/public/artifact_policy.md`: what belongs in Git and what belongs in
  external artifact storage.
- `docs/results/README.md`: generated benchmark figures and third-party
  baseline tables.
- `docs/claim_evidence/`: machine-readable evidence, pretest contracts, and
  committed compact artifact bundles.
- `docs/research/`: literature and design notes that inform future work.
- `worklog.md`: append-only operational trace. Useful for provenance, not a
  polished public narrative.

## Evidence Figures

The generated figures render committed claim evidence into a matched-protocol
scorecard, per-task breakdown, secondary metric suite, horizon profile,
validation-only diagnostics, and external benchmark matrix. Trace claims back to
the machine-readable manifests in `docs/claim_evidence/`.

The generated cards cover cost/reproducibility, benchmark readiness, ecosystem
compatibility, and qualitative-preview status. Qualitative rollout panels render
only after a compact claim-linked preview manifest and artifact are committed.

Check generated assets with:

```bash
python scripts/build_public_assets.py --check
```

## North Star

The north star is a defensible universal-simulation research claim: one model
family that improves physical-space rollout quality across task families while
preserving strict validation/test separation and artifact-level auditability.

The current highest-signal model-side blocker is long-horizon transport and
advection phase tracking. Broader backbone or pretraining changes should be
ranked against that measured blocker.
