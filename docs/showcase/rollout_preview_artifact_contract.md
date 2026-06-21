# Rollout Preview Artifact Contract

Qualitative rollout panels should not be rendered from ignored local `reports/`
files. They need a compact committed artifact tied to the same evidence rules as
the numeric showcase.

## Required Artifact

Recommended path:

```text
docs/claim_evidence/artifacts/rollout_preview_<run_name>.npz
```

Required arrays:

- `target`: decoded physical-space target frames.
- `prediction`: decoded physical-space prediction frames.
- `baseline`: optional decoded baseline frames, if the panel compares against a
  baseline.
- `time_index`: integer or float frame indices.

Latent debug previews, including `details.preview_predicted` and
`details.preview_target` arrays from latent evaluation summaries, do not satisfy
this contract. The public panel is only for decoded physical-space rollouts.

Required shape convention:

```text
sample x time x channel x spatial...
```

The artifact should be intentionally small: enough for one or two public
examples, not a hidden dataset.

## Required Manifest

A JSON evidence manifest should live beside the numeric evidence:

```text
docs/claim_evidence/rollout_preview_manifest.json
```

It should record:

- command;
- run name;
- split;
- metric name and value;
- task;
- sample count;
- frame count;
- source summary JSON;
- artifact path;
- artifact SHA-256;
- statement that held-out access was authorized or that the artifact is
  validation-only.

When this manifest exists, `python scripts/build_showcase_assets.py` validates
the artifact hash and required arrays. It then writes:

- `generated/rollout_preview_summary.tsv`
- `generated/rollout_preview_panel.png`

If the manifest is missing, the public showcase remains in the current gated
status and no qualitative panel is rendered.

## Current Status

`generated/rollout_preview_status.tsv` records that no claim-linked preview
artifact exists yet. Local ignored files such as `reports/evaluation_preview.npz`
must stay excluded from public showcase evidence until they are promoted through
this contract.
