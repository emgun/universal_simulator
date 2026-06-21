# Showcase Credibility Cards Plan

Goal: extend the public showcase with repeatable credibility cards for
cost/reproducibility, official benchmark readiness, ecosystem compatibility, and
qualitative rollout-preview status.

## Scope

- Add generated reproducibility/cost card assets.
- Add generated benchmark-readiness card assets.
- Add generated rollout-preview status assets.
- Add documentation that keeps measured matched-protocol baselines separate from
  official external protocols and ecosystem compatibility checks.
- Define the artifact contract required before qualitative rollout panels can be
  rendered.

## Implementation

- [x] Extend `scripts/build_showcase_assets.py` with card row builders and PNG
  renderers.
- [x] Add unit coverage for card row classification.
- [x] Regenerate `docs/showcase/generated/` assets.
- [x] Keep `python scripts/build_showcase_assets.py --check` as the repeatable
  source-of-truth check.
- [x] Document the new cards and rollout-preview contract.

## Claim Boundary

The cards do not broaden the benchmark claim. They show what is measured,
repeatable, planned, or missing. Dollar cost remains "not recorded" until
committed scorecards contain provider and cost fields. Qualitative panels remain
blocked until a compact claim-linked preview artifact exists.
