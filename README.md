# Universal Physics Stack (UPS)

Universal Physics Stack is research software for latent-space neural simulation
of PDE-style physical systems. It encodes physical fields into compact latent
states, evolves them with transformer-style operators, decodes predictions at
query points, and evaluates physical-space rollouts with auditable evidence.

This repository is public, but it is still an active research workbench. Treat
the current results as narrow, protocol-bound evidence rather than as a finished
production simulator or broad universal foundation-model claim.

## Current Public Boundary

The strongest public surface is the bounded claim-evidence stack under
`docs/claim_evidence/`. It records protocol, split, metric, command, artifact
hashes, and baseline context for held-out PDEBench-shaped experiments.

Start here:

- `docs/public/README.md`: public overview and claim boundary.
- `docs/public/reproducibility.md`: how to inspect evidence and reproduce local checks.
- `docs/public/artifact_policy.md`: what belongs in Git versus external artifact storage.
- `docs/claim_evidence/universal_sota_claim_evidence.json`: current machine-readable claim evidence.
- `docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md`: current research landscape and technical blocker framing.

## Quickstart

```bash
pip install -e .[dev]
pre-commit run --all-files
pytest -q tests/unit
```

For deterministic local setup:

```bash
bash scripts/prepare_env.sh
```

Some experiments require PDEBench data, GPU hardware, W&B, B2/S3, or remote
compute credentials. Keep those credentials in environment variables or ignored
local files; do not commit them.

## Repository Structure

The Python package is namespaced under `ups`.

- `src/ups/core`: latent state, conditioning, and PDE-transformer blocks.
- `src/ups/io`: grid, mesh, particle, and any-point encode/decode paths.
- `src/ups/models`: latent operators, residual/corrector modules, physics guards, and factor-graph pieces.
- `src/ups/training`: losses, loops, optimizers, curricula, and distributed helpers.
- `src/ups/data`: schemas, datasets, transforms, collate logic, and PDEBench helpers.
- `src/ups/inference`: rollout, data assimilation, and control utilities.
- `src/ups/eval`: metrics, calibration, gates, reports, and claim checks.
- `src/ups/discovery`: nondimensionalization and symbolic discovery utilities.
- `src/ups/active`: active-learning and multi-fidelity calibration experiments.
- `configs/`: training and evaluation configs.
- `scripts/`: local, remote, audit, and evidence-generation entrypoints.
- `docs/`: public overview, research notes, runbooks, and claim evidence.

## Evidence And Artifacts

UPS intentionally separates source code from generated artifacts:

- small, claim-relevant evidence lives in `docs/claim_evidence/`;
- broader research notes live in `docs/research/`;
- generated checkpoints, raw datasets, W&B runs, provider logs, and ad hoc remote outputs stay out of normal Git.

The committed bundles under `docs/claim_evidence/artifacts/` are compact
evidence bundles for auditability, not a general artifact store.

## Showcase

For a public visual summary of the current bounded evidence, see
`docs/showcase/README.md`.

- `docs/showcase/generated/claim_scorecard.png`: matched `light-v1` scorecard
  with persistence, local neural, third-party, and UPS rows.
- `docs/showcase/generated/per_task_breakdown.png`: task-level view of the
  transport/advection bottleneck.
- `docs/showcase/generated/primary_metric_suite.png`: secondary metrics beyond
  NRMSE for the primary UPS claim versus persistence.
- `docs/showcase/generated/horizon_profile.png`: step-1, H4, and H16 rollout
  profile for the primary claim.
- `docs/showcase/research_diagnostics.md`: validation-only transport ablation
  and transfer diagnostics, kept separate from held-out claims.
- `docs/showcase/credibility_cards.md`: reproducibility, cost-status, official
  benchmark readiness, and ecosystem compatibility cards.
- `docs/showcase/rollout_preview_artifact_contract.md`: required artifact
  contract before qualitative rollout panels are public evidence.
- `docs/showcase/external_benchmarks.md`: measured external baselines and
  future third-party benchmark surfaces.

These figures are generated from committed evidence JSON. They are not a
substitute for the claim manifests under `docs/claim_evidence/`.

Regenerate or check them with:

```bash
python scripts/build_showcase_assets.py
python scripts/build_showcase_assets.py --check
```

## Common Commands

Inspect the current claim status:

```bash
python scripts/audit_universal_sota_status.py --medium-confirmed
```

Run a bounded light experiment:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name local_light_check \
  --output-root reports/research/local_light_check
```

Hydrate registered datasets when credentials and storage are configured:

```bash
python scripts/fetch_datasets.py burgers1d_subset_v1 --root data/pdebench --cache artifacts/cache
```

## Status

- License: Apache-2.0.
- Python: 3.10+.
- Package status: research alpha.
- CI: GitHub Actions runs lint and unit tests on Python 3.10.

Current technical north star: improve decoded physical-space rollout quality
across task families while preserving validation/test separation and artifact
auditability. The most important measured blocker is long-horizon transport and
advection phase tracking.
