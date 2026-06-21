# Showcase Rollout Preview Artifact Support Plan

Goal: make qualitative rollout previews repeatable and claim-linked without
promoting ignored local `reports/` files or latent-space debug previews as public
evidence.

## Scope

- Add an optional manifest path for compact rollout preview artifacts.
- Validate artifact hash, required arrays, and shape conventions before rendering.
- Render a qualitative target/prediction/error panel only when the manifest and
  artifact exist.
- Keep the current public showcase in a gated missing-artifact state until a real
  decoded preview artifact is committed.

## Implementation

- [x] Extend `scripts/build_showcase_assets.py` with optional rollout preview
  manifest validation.
- [x] Add conditional `generated/rollout_preview_summary.tsv` and
  `generated/rollout_preview_panel.png` outputs for valid artifacts.
- [x] Add unit coverage for present-manifest status and panel generation.
- [x] Update public docs with the manifest path and conditional generated files.
- [ ] Produce a real decoded `light-v1` preview artifact from a current
  claim-linked run.

## Claim Boundary

The renderer support does not create a new qualitative claim by itself. It only
defines the repeatable path. The next artifact must record command, split,
metric, source summary, artifact SHA-256, and whether the preview is
validation-only or authorized held-out evidence.
