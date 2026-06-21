# Reproducibility Guide

This project separates three levels of evidence:

1. Smoke and plumbing checks show that scripts, data hydration, and remote
   launch paths execute.
2. Validation experiments select or reject candidate ideas.
3. Held-out evidence supports narrow public claims and must be recorded in
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
requirements should be called out in the specific runbook or evidence file.

## Evidence Inspection

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

## Showcase Asset Check

The public-facing figures and tables under `docs/showcase/generated/` are
derived from committed evidence. Verify they are current with:

```bash
python scripts/build_showcase_assets.py --check
```

The command regenerates the showcase packet in a temporary directory and
compares it with the committed files. A failure means the generated public
assets are stale relative to the evidence inputs or generator.

The generated credibility cards under `docs/showcase/generated/` are covered by
the same check. They should not be edited by hand.

## Data And Artifact Notes

The repository intentionally keeps compact evidence bundles when they are needed
for auditability. Large generated data, W&B run directories, checkpoints, and
remote launch outputs should stay out of normal Git and live in external
artifact storage or release assets.

Credential-gated hydration paths use environment variables. Never commit local
`.env` files, literal cloud keys, or copied provider passwords.

## Interpreting Results

Use the strongest applicable wording:

- "smoke": execution-path evidence only.
- "validation": candidate selection evidence only.
- "held-out": claim evidence only when the pretest contract and ledger agree.
- "external baseline": comparable only if the baseline was run or mapped under
  the same protocol.

When in doubt, describe the run as research evidence and point to the manifest.
