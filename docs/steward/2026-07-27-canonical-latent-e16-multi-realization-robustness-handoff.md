# Canonical Latent E16 Multi-Realization Robustness Handoff

Date: 2026-07-27

## Decision

E16 is `both_practical_recovery_packages_stable`.

The sealed E15 result plus two preregistered fresh training/schedule
realizations produce a complete six-bit pass vector:

- deterministic schedule-weighted AdamW with fresh moments repairs each
  realization's own E12 checkpoint; and
- schedule-weighted componentwise strong-Wolfe L-BFGS recovers from neutral on
  every realization.

The linear structured-generator branch has reached its intended robustness
gate. Stop optimizer archaeology and open the representation-preserving
nonlinear gate.

## Decisive evidence

| Realization | E12 checkpoint basis / HF | AdamW restart basis / HF | L-BFGS neutral basis / HF |
| --- | ---: | ---: | ---: |
| sealed E15 | `0.099922` / `0.261281` | `0.001049` / `0.006766` | `1.602e-5` / `3.840e-5` |
| `r1` | `0.089873` / `0.295232` | `0.000655` / `0.005247` | `1.392e-5` / `1.892e-5` |
| `r2` | `0.094367` / `0.278676` | `0.003264` / `0.004439` | `1.902e-5` / `4.386e-5` |

The ordered-AdamW checkpoint failure repeats on both fresh realizations, so the
repair result is not an artifact of the original E12 sample. Both registered
practical packages then clear all eight unchanged E15 recovery gates on all
three realizations.

All literal seed, schedule, canonical-byte, excitation-rank,
grouped/literal-objective, separability, finiteness, coverage, replica, state,
and boundary checks pass. Independent pre-state and post-result review both
returned GO with no P0/P1 blocker.

## Evidence seal

- execution HEAD:
  `139c1c0d1ecf626b3a962cb4824337a75a6806f9`;
- bundle:
  `71fc490c2bc361fbf0b26d5bfcccfc460bcf5af223b5000d1e6043672504a586`;
- compact result:
  `6716273a3ea980f7d24462ec3e40eb37091d229d524aec5f9a0ad89bbb9d325a`;
- detached manifest:
  `6af927037eebeebc3a9a95842d549c633279391d28715779b7fc04c05b59720f`;
- raw replicate:
  `621e59af7dca5312963db4b32b88aafe02f3f9d03fbbf63ed971e0c059e637b1`;
- canonical replicate:
  `2defe29b2b13839215484bc8595ec3a4d86edd2c384adf57e09fe2fa67f7c8b8`;
- focused E16 tests: `36/36`;
- related E12-E16 tests: `104/104`;
- independent pre-state review: GO;
- independent post-result review: GO.

## Boundary

E16 uses 1,536 unique training trajectories, 256 frozen validation
trajectories, and zero held-out trajectories. It makes zero provider calls,
encoder updates, routing decisions, label inputs, or source bypasses.

The result qualifies training-package stability only for the frozen smooth
periodic scalar linear family. It does not qualify nonlinear dynamics,
particle dynamics, a new representation, or a deployment claim.

## Next arc

Open E17, not another optimizer experiment.

The smallest high-signal nonlinear contract is a 2-D periodic scalar viscous
Burgers Galerkin/closure test in the same 52-coefficient E10 latent:

- keep the encoder, coefficient order, inactive trends, routing closure, and
  identified linear generator frozen;
- construct a de-aliased nonlinear truth and independently bound spatial/time
  discretization error;
- measure whether the 49 active periodic modes form a sufficiently Markov
  projected state before blaming the operator;
- add only a constrained quadratic convection term, with the frozen linear
  model as an exact nested negative control;
- compare against projected truth and, if needed, an oracle-support ceiling;
- evaluate derivative fit, short/medium/long rollout, high-frequency error,
  mass and energy balance, semigroup defect, finite-amplitude stress, and
  cross-observation consistency; and
- keep synthetic train/validation only with zero held-out access.

Operator inference with an energy-preserving quadratic term is the preferred
first mechanism because it matches Burgers' polynomial convection while
preserving interpretability and the E10 representation. Koopman lifting,
neural ODEs, FNOs, and routing are deferred because each changes the
representation/capacity hypothesis before fixed-latent nonlinear closure has
been tested.
