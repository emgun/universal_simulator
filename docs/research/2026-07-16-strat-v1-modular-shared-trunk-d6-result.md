# D6 Modular Shared-Trunk Result

Date: 2026-07-16

## Decision

D6 is negative on both preregistered universal-value gates. Stop broadening the
shared-model branch at this scale. The next product architecture should expose
one unified simulator interface over family-specific models rather than force
one shared neural trunk across transport, conservation, and elliptic systems.

No extra seed, longer schedule, relaxed gate, replacement run, U3/U4 follow-up,
or held-out measurement is authorized by this result.

## Execution boundary

- Plan SHA-256: `e05db0d5bbf04bd9ca603225a1aa8a2529390b579a642bcd4a7d84f519cb1709`
- Training-lock SHA-256: `5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd`
- Vast offer: `44203279`
- Vast instance: `45129642`
- GPU: one RTX 4090
- Contract price: `$0.328888888889/hour`, including 96 GB disk
- Paid interval: approximately 103 minutes; estimated charge no more than `$0.57`
- Scientific run duration: `3924.218765` seconds
- Staged objects: exactly six, all roles `train` or `valid`
- Held-out reads: `0`
- Active Vast instances after reconciliation: `0`

## Result

| Metric | Joint modular | Matched single-task ensemble | Frozen D5 specialists |
| --- | ---: | ---: | ---: |
| Macro rollout NRMSE | `0.809862` | `0.662876` | `0.689476` |
| Advection1D | `0.740055` | `0.553564` | `0.532333` |
| Burgers1D | `0.770677` | `0.525748` | `0.574650` |
| Darcy2D | `0.918854` | `0.909315` | `0.961443` |

The joint-to-matched-ablation ratios are `1.336892` for Advection,
`1.465869` for Burgers, and `1.010490` for Darcy. The macro negative-transfer
gate and every per-task negative-transfer gate fail despite exact scheduled
update parity. U2 therefore fails.

U1 also fails. The joint macro is `1.1746x` the matched specialist-oracle
reference and is worse than the frozen D5 specialist on Advection and Burgers.
Darcy corrected regime spread is `2.173540`, above the frozen `1.5` limit.
Parameter shuffling changes macro NRMSE by only
`0.0001184%` (`1.184208e-6` relative degradation), so the parameter-use gate
fails. Persistence, per-task specialist, absolute Darcy, and corrected-spread
checks also fail.

The consolidation checks pass but do not rescue the candidate. The joint
checkpoint is `23,870,148` bytes versus `68,296,675` bytes across the three
matched ablations, and the joint has `5,868,868` initialized tensor elements
versus `16,785,804` across the ensemble. Total scheduled optimizer updates are
exactly equal: `7,044` joint and `7,044` across the ablations.

## Artifact and teardown recovery

The remote run completed and uploaded a verified ingress archive with SHA-256
`3e58f7fea593f46e05389c9260a13ac33f60eca44e157cdb06234a9c1eaf9bcc`.
Remote auto-shutdown failed because the provider container did not run systemd,
and the turn-scoped local watchdog was no longer alive. The idle instance was
manually destroyed and Vast then returned an empty active-instance list.

The original receipt was preserved with conservative status `instance_absent`
and `destroyed=true`. A tested recovery gate required that exact receipt state,
tracked-bootstrap evidence, the pre-teardown expected digest, ingress sidecar
agreement, full source hashing, immutable server-side copy, and full immutable
read-back before cleanup. An independent local download matched the same
SHA-256.

Immutable artifact:
`b2://pdebench/remote-runs/strat-v1-modular-shared-trunk/immutable/sha256/3e58f7fea593f46e05389c9260a13ac33f60eca44e157cdb06234a9c1eaf9bcc/strat_v1_modular_shared_trunk_d95e44270a5c118beb87a4c26e587053.tar.gz`

The result artifact inside the archive is self-hashed as
`9caeb7f699794bf90dda66a73691a80fb503a9616f74c43842a96f1c586e6ef6`.
The stage-report artifact SHA-256 is
`7e03c0326fb0f62300a8f16e31584dadebef0cc4f9e2b68309fb30730584fb27`.

## Forward path

Preserve the shared product contract at the interface layer: common task and
parameter schemas, data locks, metrics, provenance, artifact handling, and
routing. Move model selection beneath that interface, with family-specific
transport, conservation, and elliptic implementations. The first next step is
a no-provider interface-and-routing design, not another model training run.
