# Contributing

UPS is a research codebase. Contributions are welcome when they preserve the
separation between source code, experiment evidence, generated artifacts, and
claims.

## Development Setup

```bash
pip install -e .[dev]
pre-commit run --all-files
pytest -q tests/unit
```

Some integration tests require local datasets, cloud credentials, or GPU
hardware. Keep ordinary pull requests reviewable without requiring paid remote
compute.

## Pull Request Expectations

- Keep code, docs, configs, and evidence changes separate when practical.
- Include the command used to validate the change.
- Do not commit local datasets, W&B run directories, checkpoints, `.env` files,
  remote instance logs, or provider credentials.
- Put small, claim-relevant evidence under `docs/claim_evidence/` only when it is
  part of an auditable contract. Prefer external artifact storage for larger
  outputs.
- State whether a result used validation data, held-out test data, or only a
  smoke/dev split.

## Research Claim Discipline

Do not frame a result as universal, production-ready, or state of the art unless
the supporting evidence file says that explicitly and records the protocol,
split, metric, command, artifact hash, and comparison baseline.

The public-facing summary starts at `docs/public/README.md`. Detailed research
notes and append-only plans may be useful background, but they are not a
substitute for claim evidence.
