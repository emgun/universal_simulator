# P2 Source Blocker Branch Check

Date: 2026-06-22

Status: no-provider steward branch check. No external source restore, GPU run,
held-out test, credential use, or claim evidence change.

## Trigger

The next P2.2 step is a 2-sample CPU smoke for the implemented Poseidon
`channel_lift` path. Local checks found:

- `scripts/run_external_poseidon_scot_finetune.py` already supports
  `--adapter-mode channel_lift`.
- `tests/unit/test_external_poseidon_scot_finetune.py` covers replicate init,
  trainable parameter boundaries, frozen backbone behavior, rollout loss, and
  summary validation.
- `python -m pytest tests/unit/test_external_poseidon_scot_finetune.py -q`
  passed on 2026-06-22.
- Local `data/pdebench` train/val shards and the cached Poseidon-T checkpoint
  are present.
- Official Poseidon `scOT` is not importable in the canonical environment.
- Prior run summaries recorded `/tmp/poseidon-official` at commit
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, but that temp path now contains
  only an invalid/empty `.git` directory.

## Options

### A. Restore Poseidon source and run CPU smoke

Pros:

- Directly unblocks the already-implemented P2.2 path.
- Best aligned with the completed P2.1 design and current roadmap gate G2a.
- Lowest new implementation risk because the runner and tests already exist.
- CPU smoke can remain local and validation-only after the source checkout is
  available.

Cons:

- Restoring or cloning official source requires explicit approval because it is
  network/external-source setup.
- A later GPU validation run still needs separate approval and likely Vast
  balance top-up.

Decision: preferred path after approval.

### B. Pivot to DPOT probe design

Pros:

- DPOT is a credible backbone challenger if Poseidon stalls.
- It tests a different foundation recipe: autoregressive denoising operator
  pretraining instead of Poseidon ScOT.

Cons:

- No local DPOT runner is already implemented in this repo.
- It likely needs model/source retrieval and GPU validation too, so it does not
  avoid the current approval class.
- It should remain a backup after Poseidon Option A/B, not a reason to abandon
  the ready `channel_lift` path before smoke testing it.

Decision: keep as fallback, not the next move.

### C. Pivot to local UPS-side challenger/refiner work

Pros:

- Avoids external source dependency.
- Stays fully inside the UPS stack and could inform long-run architecture.

Cons:

- Prior decoded pointwise/spatial refiner probes are recorded as negative or
  not transferable in existing roadmap notes.
- Generic transport sidecar work is already saturated/deprioritized for the
  current track unless a phase gate reopens it.
- Reopening this branch now would be a local-minimum response to source setup
  friction rather than a better north-star move.

Decision: do not pivot here just because Poseidon source is missing.

## Conclusion

Continue the Poseidon `channel_lift` route, but treat official source restore as
an explicit user-approval gate. If approval is not granted, the next safe
no-provider work should prepare a precise source-restore and CPU-smoke runbook
or tighten the DPOT fallback design, not start another local refiner experiment.

