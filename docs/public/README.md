# Public Overview

Universal Physics Stack (UPS) is research software for latent-space neural
simulation of PDE-style physical systems. It combines grid and field encoders,
latent transformer operators, any-point decoders, rollout evaluation, and
claim-evidence artifacts.

This repository is public, but it should be read as an active research workbench
with auditable evidence, not as a production simulator or a finished foundation
model.

## Current Public Claim Boundary

UPS currently supports a guarded research claim around bounded PDEBench-shaped
experiments. The most important public contract is:

- protocol family: `light-v1` / bounded PDEBench-shaped tasks
- representative tasks: `advection1d`, `burgers1d`, `darcy2d`
- primary metric: decoded physical-space rollout NRMSE
- evidence root: `docs/claim_evidence/`
- latest claim manifest: `docs/claim_evidence/universal_sota_claim_evidence.json`

The claim evidence is intentionally narrower than the project name. Results
outside the recorded protocol should be treated as research notes until they are
rerun under an auditable contract.

## Where To Start

- `README.md`: top-level installation, repo map, and claim boundary.
- `docs/public/reproducibility.md`: how to reproduce or inspect evidence.
- `docs/public/artifact_policy.md`: what belongs in Git and what belongs in
  external artifact storage.
- `docs/claim_evidence/`: machine-readable evidence, pretest contracts, and
  committed compact artifact bundles.
- `docs/research/`: literature and design notes that inform future work.
- `worklog.md`: append-only operational trace. Useful for provenance, not a
  polished public narrative.

## What This Is Not Yet

- Not a production-supported solver API.
- Not a general physics foundation model claim across arbitrary physical
  systems.
- Not a benchmark leaderboard unless the relevant evidence manifest records the
  exact protocol, split, metric, and baseline comparison.
- Not a place to store large generated checkpoints or raw datasets in normal Git
  history.

## North Star

The north star is a defensible universal-simulation research claim: one model
family that improves physical-space rollout quality across task families while
preserving strict validation/test separation and artifact-level auditability.

The current highest-signal model-side blocker is long-horizon transport and
advection phase tracking. Broader backbone or pretraining changes should be
ranked against that measured blocker.
