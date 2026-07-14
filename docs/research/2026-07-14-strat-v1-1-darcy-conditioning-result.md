# `strat-v1.1` Darcy conditioning result

Date: 2026-07-14

Status: mechanism confirmed; specialist gate not passed

## Execution

The pre-registered D1 plan ran once on Vast contract `44908875`, using an RTX
4090 and only the frozen six-object universal training lock. The executed Git
commit was `af905c67759bb61f96f284d37e579a879832b279`; it contains implementation
commit `17f1305d720324af492039e1205bffde94fd3fae`. The plan verified all 65 bound
source hashes before training.

The run staged and checksum-verified `427,029,641` train/validation bytes. No
measurement lock or test object was staged or read. Both arms completed 792
optimizer steps and 6,240 examples at seed 17. Training and evaluation took
about nine GPU seconds; instance setup, dependency installation, staging, and
publication dominated the roughly three-minute rental, costing about two
cents.

The immutable run archive is:

`b2://pdebench/remote-runs/darcy-fno-conditioning-ablation/immutable/sha256/5f81712fb381ee1fa4179b46139affd7143825592ef9f5492a039bc8eeed1e2e/darcy_fno_conditioning_ablation_20260714T181109Z.tar.gz`

The retrieved archive SHA-256 matched
`5f81712fb381ee1fa4179b46139affd7143825592ef9f5492a039bc8eeed1e2e`.
Contract `44908875` was then destroyed manually after the container's systemd
shutdown command proved unavailable.

## Result

| Arm | Selected epoch | Validation NRMSE | Max corrected spread | Plateau by 24 |
| --- | ---: | ---: | ---: | --- |
| `U`, coefficient only | 24 | `0.876806` | `1.939350` | yes |
| `K`, coefficient + beta + presence | 24 | `0.189475` | `2.170477` | no |

Conditioning improved primary NRMSE by `78.39%`. The causal diagnostics also
passed: counterfactual beta changed `K` predictions substantially, while
deterministically shuffled beta raised NRMSE to `1.358415`, a `616.94%`
degradation from the correctly conditioned result. `U` is exactly invariant to
counterfactual beta.

This confirms the omitted-input mechanism: beta must be part of the universal
operator interface. The improvement is far too large, and the shuffled-beta
penalty far too strong, to attribute to the 64 additional lifting parameters.

The full promotion gate nevertheless failed for two pre-registered reasons:

- `K` maximum corrected spread is `2.170477`, above `1.5`; beta `100` remains
  the dominant absolute-error regime with global-scale NRMSE `0.411251`.
- `K` improved at every rung (`0.645845`, `0.264139`, `0.219361`, `0.189475`)
  and did not plateau by epoch 24.

Therefore this is positive mechanism evidence but not a claim-grade specialist
recipe and not authorization for held-out access.

## Interpretation and next decision

Do not jump directly to Poseidon or `tier_b`, and do not relax the corrected
spread gate. Parameter value and presence must become universal inputs first.
The next experiment should target the remaining beta-100 error concentration,
not re-prove that conditioning helps.

A simple beta output-rescaling shortcut was checked and rejected: matched
targets are not linear in beta on either train or validation. The strongest
next design is therefore a validation-only, regime-balanced conditioned
specialist that keeps the explicit beta input but separates shape prediction
from parameter-dependent amplitude calibration. Before running it, add
optimizer state to rung checkpoints and pre-register enough budget to establish
a real plateau. Compare against this frozen `K` trajectory and retain the
shuffled-beta and counterfactual gates.

The compact self-hashed result is
`docs/research/artifacts/strat_v1_1_darcy_fno_conditioning_ablation_result.json`.
