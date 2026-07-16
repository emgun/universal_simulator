# `strat-v1` Shared `tier_b` D5 Result

Date: 2026-07-15

Status: complete, validation-only, negative

Decision: do not promote this candidate or proceed to D5b, U3/U4, or held-out measurement

## Question

D5 was the one allowed native `tier_b` retrial under the frozen `strat-v1.1`
contract. It asked whether one parameter-conditioned model could match the
frozen single-task controls across Advection, Burgers, and Darcy while using
its conditioning, controlling regime imbalance, and materially consolidating
the checkpoint footprint.

The run used seed 17, the exact six-object training lock, and validation data
only. The measurement lock and all test objects remained sealed.

## Result

| Evidence | Shared candidate | Frozen comparison or gate | Outcome |
| --- | ---: | ---: | --- |
| Three-task macro NRMSE | `0.797232` | specialist oracle `0.689476`; ratio `1.156288 <= 1.05` | fail |
| Advection NRMSE | `0.960067` | specialist `0.532333`; persistence `0.673296` | fail |
| Burgers NRMSE | `0.993537` | specialist `0.574650`; persistence `0.640048` | fail |
| Darcy NRMSE | `0.438093` | specialist `0.961443`; absolute wall `0.140330` | relative gain; absolute wall fail |
| Maximum corrected regime spread | `2.193058` on Darcy | `<= 1.5` on every task | fail |
| Shuffled-conditioning degradation | `4.4947%` | `>= 5%` | fail |
| Checkpoint bytes | `23,688,828` | specialist ensemble `67,752,971` | pass |
| Held-out reads | `0` | exactly `0` | pass |

Only the checkpoint-consolidation and zero-held-out gates passed. The shared
checkpoint is about 65% smaller than the three specialist checkpoints, but
that operational gain does not compensate for the accuracy loss. The result
is therefore `shared_tier_b_not_validated`.

The task pattern is also informative. Native sharing substantially improved
Darcy relative to the frozen weak specialist, but regressed both 1D evolution
tasks and retained Darcy's high-regime imbalance. The conditioning diagnostic
missed its preregistered threshold narrowly, so the run does not establish
robust parameter use either. The pattern is consistent with cross-task
interference in this monolithic candidate, although capacity or optimization
could also contribute; it is not a reason to relax the gates.

## Integrity and provenance

- Frozen plan SHA-256: `5e44e12eb387eec037ac8b7200e7577f9f4d6f806a056b7516342702c9bd7bfd`
- Training lock SHA-256: `5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd`
- Canonical result identity SHA-256: `737c08903ca4f45bdc992e5abb53ebe39fe948ea2feac375469bf901c9e9d762`
- Committed result file SHA-256: `c50ff19f896edbe332629d8f87140e8e0b62b427381a35ded023d0929e751747`
- Source summary artifact SHA-256: `15a90072fcd2225ffc85421869e47e2a4162029d33eccf1c0b80f089a627bbc8`
- Implementation bound at plan time: `8872c685c6c985896e3fd95f8669caef66969ffe`
- Executed merge: `68c3ad2065e1c8bd39d6713dbaffeb707485a421`
- Vast contract: `45040272`, destroyed after evidence recovery
- Total orchestrator duration: `2928.18` seconds
- Immutable archive SHA-256: `5b9a8e40d06e1259fa2bdad856e7800f064e33f03fd0a6dde55b24c0c216c04f`
- Immutable archive:
  `b2://pdebench/remote-runs/strat-v1-shared-tier-b/immutable/sha256/5b9a8e40d06e1259fa2bdad856e7800f064e33f03fd0a6dde55b24c0c216c04f/strat_v1_shared_tier_b_3a7ff1cc00572f16a545039e5ada3098.tar.gz`

The archive was downloaded independently, its SHA-256 was rechecked, and the
compact result was rebuilt from the returned summaries with an exact match.
All recorded result and source-summary hashes verify. Scoped signed transfers
were used for B2 hydration and publication, so the Vast worker received no
reusable object-store credential. The exact compact stage record is committed
at `docs/research/artifacts/strat_v1_shared_tier_b_stage.json`.

## Decision and next architecture

Close the native monolithic `tier_b` branch at this frozen scale. Do not add
seeds, extend training, lower the parameter-use threshold, or spend held-out
access on it.

The next bounded hypothesis should preserve sharing where it is likely to help
while isolating incompatible geometry and dynamics: family/task-specific input
and output codecs or lightweight experts around one conditioned operator
trunk. Pre-register it against the same seed, exposure, specialist controls,
and U1/U2 gates. If that modular candidate also fails U1, narrow the product to
a unified interface over family-specific models. PDEBench breadth and The Well
remain downstream of credible shared-value evidence.
