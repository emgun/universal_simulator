# `strat-v1.1` Reference-Recipe Adequacy Result

Date: 2026-07-14

## Decision

No FNO or UNO recipe is eligible to become the claim-grade specialist
reference. Both curves plateaued, but both selected checkpoints failed the
pre-registered corrected Darcy regime-spread gate. Confirmation seeds 29 and
43 were not run. No held-out or measurement-lock access occurred.

## Frozen execution

- Plan: `strat-v1.1-fno-uno-reference-recipe-adequacy-v1`
- Plan SHA-256: `59bab7d1f7efb69f98aad41b3de41d46cc19cc5174ce00f660c0cb8a4dcb0b05`
- Data lock SHA-256: `5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd`
- External package: `neuraloperator==2.0.0`
- Discovery seed: `17`
- Continuous validation rungs: `3, 6, 12, 24, 48`
- Provider: Vast contract `44811726`, one RTX 4090
- Actual hourly rate: `$0.4192592593`; active contract time was approximately
  22 minutes, or less than `$0.16` for this successful attempt.

The stage report contains exactly three train and three validation objects,
427,029,641 bytes total. It contains no test object.

## Results

| Architecture | Parameters | Duration | Plateau epoch | Selected epoch | Macro validation NRMSE | Selected Darcy spread | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FNO | 176,339 | 452.27 s | 24 | 6 | `0.6518386020` | `1.910791526` | No |
| UNO | 266,771 | 476.37 s | 24 | 6 | `0.6582274683` | `1.917583452` | No |

The plateau label only establishes that more epochs under this recipe are not
the missing ingredient. Eligibility additionally requires every task's
corrected regime spread to be at most `1.5`. Darcy is the decisive failure for
both architectures.

## Durable evidence

- Selection SHA-256: `616e2d7b616a8f8d21d53ca189590b194829b796d428dc00035d2bbc90194270`
- Immutable archive SHA-256:
  `bc917695cdec16a517995036576933628a8d9a3136ad2f1fb1bffaaa2e5b78b7`
- Archive size: 16,044,247 bytes
- B2 object:
  `b2://pdebench/remote-runs/reference-recipe-adequacy/immutable/sha256/bc917695cdec16a517995036576933628a8d9a3136ad2f1fb1bffaaa2e5b78b7/reference_recipe_adequacy_20260714T062421Z.tar.gz`

The archive was downloaded after publication and independently verified
against its SHA-256 before the Vast contract was destroyed. It contains the
plan, staging report, selection artifact, both complete summaries, all rung
checkpoints, and group tables. Large checkpoints remain outside Git.

## Forward path

Do not extend these recipes, run their confirmation seeds, or use held-out
data. The highest-signal next step is to test the already-prioritized explicit
parameter/task conditioning candidates on validation and determine whether
they reduce Darcy regime sensitivity while preserving Advection and Burgers.
If none pass, narrow the product claim or pre-register one different specialist
whose mechanism directly targets heterogeneous elliptic regimes.
