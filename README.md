# Universal Physics Stack (UPS)

Universal Physics Stack is research software for latent-space neural simulation
of PDE-style physical systems. It encodes physical fields into compact latent
states, evolves them with transformer-style operators, decodes predictions at
query points, and evaluates physical-space rollouts with reproducible result
records.

## Results At A Glance

Legacy matched-`light-v1` decoded rollout NRMSE, lower is better:

| Row | NRMSE | Scope |
| --- | ---: | --- |
| UPS primary | `0.4166` | Legacy matched-protocol UPS result |
| Persistence baseline | `0.5702` | Non-learned reference |
| Fourier baseline | `0.5637` | Repo-native neural baseline |
| NeuralOperator UNO | `0.5561` | Third-party model rerun under `light-v1` |
| CNO1d | `0.5919` | Official-source CNO adapter under `light-v1` |
| PDEBench U-Net | `0.6096` | Official PDEBench architecture adapter |
| NeuralOperator FNO | `0.6392` | Canonical FNO family under `light-v1` |

These rows are matched comparisons under one frozen, mixed protocol, not broad
held-out-generalization claims. A split-integrity audit found that Burgers test
trajectories occur in training, Advection reuses initial conditions across
splits while extrapolating transport speed, and Darcy is trajectory-disjoint.
The comparison remains useful because every row used the same protocol; new
generalization claims wait for the disjoint, regime-stratified `strat-v1`
protocol. New construction cannot use the `light-v1` or `medium-v1` labels;
those namespaces and their artifacts are frozen as historical evidence.

The generated figures and tables live in `docs/results/`. The source records
under `docs/claim_evidence/` capture protocol, split, metric, command, artifact
hashes, and baseline context. See the
[`light-v1`/`medium-v1` split-integrity audit](docs/research/2026-07-09-split-integrity-audit.md)
for task-level interpretation.

## Architecture

```mermaid
flowchart LR
  data["Locked PDEBench + The Well trajectories<br/>src/ups/data"] --> enc["Grid, mesh, particle encoders<br/>src/ups/io"]
  enc --> core["Latent state and conditioning<br/>src/ups/core"]
  core --> models["Latent operators, residuals, guards<br/>src/ups/models"]
  models --> dec["Any-point decoding<br/>src/ups/io"]
  dec --> rollout["Rollout, assimilation, control<br/>src/ups/inference"]
  rollout --> eval["Metrics, gates, reports<br/>src/ups/eval"]
  eval --> records["Result records and figures<br/>docs/claim_evidence + docs/results"]

  configs["Configs and scripts<br/>configs + scripts"] --> data
  configs --> models
  train["Training loops and losses<br/>src/ups/training"] --> models
  discovery["Discovery and active learning<br/>src/ups/discovery + src/ups/active"] --> train
```

## Run Locally

```bash
pip install -e .[dev]
pytest -q tests/unit
python scripts/build_public_assets.py --check
```

For deterministic local setup:

```bash
bash scripts/prepare_env.sh
```

Run a bounded light experiment:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name local_light_check \
  --output-root reports/research/local_light_check
```

Some experiments require PDEBench data, GPU hardware, W&B, B2/S3, or remote
compute credentials. Keep credentials in environment variables or ignored local
files; do not commit them.

## Repository Map

- `src/ups/core`: latent state, conditioning, and PDE-transformer blocks.
- `src/ups/io`: grid, mesh, particle, and any-point encode/decode paths.
- `src/ups/models`: latent operators, residual/corrector modules, physics guards, and factor-graph pieces.
- `src/ups/training`: losses, loops, optimizers, curricula, and distributed helpers.
- `src/ups/data`: immutable manifests/locks, verified staging, lazy PDEBench and The Well adapters, transforms, and collate logic.
- `src/ups/inference`: rollout, data assimilation, and control utilities.
- `src/ups/eval`: metrics, calibration, gates, and reports.
- `configs/`: training and evaluation configs.
- `scripts/`: local, remote, audit, and asset-generation entrypoints.
- `docs/`: public overview, research notes, runbooks, and result records.

## Start Here

- `docs/results/README.md`: generated figures, benchmark tables, and regeneration command.
- `docs/public/README.md`: public overview and current result scope.
- `docs/public/reproducibility.md`: how to inspect records and reproduce local checks.
- `docs/public/artifact_policy.md`: what belongs in Git versus external artifact storage.
- `docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md`: research landscape and technical blocker framing.

## Artifacts

UPS intentionally separates source code from generated artifacts:

- small result records live in `docs/claim_evidence/`;
- broader research notes live in `docs/research/`;
- generated checkpoints, raw datasets, W&B runs, provider logs, and ad hoc remote outputs stay out of normal Git.

The committed bundles under `docs/claim_evidence/artifacts/` are compact result
bundles for reproducibility, not a general artifact store.

## Status

- License: Apache-2.0.
- Python: 3.10+.
- Package status: research alpha.
- CI: GitHub Actions runs lint, YAML integrity checks, unit and integration
  tests, and deterministic public-asset verification on Python 3.10.

Current technical north star: determine whether one shared simulator can
approach credible specialist accuracy while improving transfer, data
efficiency, or operational consolidation on trajectory-disjoint,
regime-stratified data. The immediate gates are a versioned regime-metric
erratum, claim-grade validation recipe adequacy, and shared-candidate value
testing. Held-out measurement follows selection; it is not the next baseline
step.
