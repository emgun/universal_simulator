# Reproducibility Guide

This project separates three result levels:

1. Smoke and plumbing checks show that scripts, data hydration, and remote
   launch paths execute.
2. Validation experiments select or reject candidate ideas.
3. Held-out runs support public result tables and must be recorded in
   `docs/claim_evidence/`.

Do not mix those levels when describing results.

## Local Setup

```bash
pip install -e .[dev]
pre-commit run --all-files
pytest -q tests/unit
```

The full dependency stack includes PyTorch and scientific Python packages. Some
experiments require GPU hardware, PDEBench data, or cloud credentials. Those
requirements should be called out in the specific runbook or result record.

## Result Inspection

Start with:

```bash
python scripts/audit_universal_sota_status.py --medium-confirmed
```

Then inspect:

- `docs/claim_evidence/universal_sota_claim_evidence.json`
- `docs/claim_evidence/universal_sota_roadmap.md`
- `docs/claim_evidence/*_pretest_contract.json`
- `docs/claim_evidence/*_evidence.json`

Every held-out result should record the command, split, metric, artifact handle,
and artifact hash needed for review.

## Generated Asset Check

The public-facing figures and tables under `docs/results/generated/` are
derived from committed records. Verify they are current with:

```bash
python scripts/build_public_assets.py --check
```

The command regenerates the public asset packet in a temporary directory and
compares it with the committed files. A failure means generated assets are stale
relative to the source records or generator.

The generated credibility cards under `docs/results/generated/` are covered by
the same check. They should not be edited by hand.

## Data And Artifact Notes

The repository intentionally keeps compact result bundles when they are needed
for reproducibility. Large generated data, W&B run directories, checkpoints, and
remote launch outputs should stay out of normal Git and live in external
artifact storage or release assets.

Credential-gated hydration paths use environment variables. Never commit local
`.env` files, literal cloud keys, or copied provider passwords.

## Interpreting Results

Use the strongest applicable wording:

- "smoke": execution-path check only.
- "validation": candidate selection result only.
- "held-out": public result table only when the pretest contract and ledger agree.
- "external baseline": comparable only if the baseline was run or mapped under
  the same protocol.

When in doubt, describe the run as a research result and point to the record.
