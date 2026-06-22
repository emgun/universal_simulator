# Artifact Policy

UPS needs artifacts for reproducibility, but normal Git history should stay small
and reviewable.

## Keep In Git

- Source code, tests, configs, and small scripts.
- Public documentation and runbooks.
- Small machine-readable result records under `docs/claim_evidence/`.
- Compact result bundles when they are required to make a result reviewable and
  no external artifact handle exists.
- Small research summaries under `docs/research/`.

## Keep Out Of Git

- Raw datasets.
- Local `.env` files and credentials.
- W&B run directories.
- Cloud instance logs and copied provider output.
- Checkpoints and model weights (`*.pt`, `*.pth`, `*.ckpt`) unless there is a
  deliberate release process.
- Ad hoc `remote_artifacts/` and `remote_run*/` directories.

Use GitHub Releases, W&B artifacts, B2/S3, or another explicit artifact store
for large generated outputs. Record stable handles and hashes in the relevant
result record.

## Existing Compact Bundles

The committed bundles in `docs/claim_evidence/artifacts/` are treated as compact
result artifacts. They should not become a general dumping ground for experiment
outputs. New bundles need a clear record, protocol, split, metric, and hash.
