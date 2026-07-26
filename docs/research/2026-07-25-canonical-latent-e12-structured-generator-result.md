# Canonical Latent E12 Structured-Generator Result

Date: 2026-07-25
Status: `structured_generator_not_qualified`

## Decision

The frozen additive continuous-time coefficient generator is not qualified as
the shared latent dynamics operator.

This is a narrow negative, not a collapse of the representation-first path.
The structured candidate improves E11's dense pretrained rollout by more than
`400x`, transfers decisively from elementary to composite dynamics, and passes
one-step, rollout, temporal, semigroup, invariance, physics, and composition
gates. It nevertheless misses two preregistered requirements:

1. composite final high-frequency NRMSE is `0.295429` versus `<=0.15`;
2. maximum decoded basis-action error is `0.0999215` before fine-tuning and
   `0.1084054` after fine-tuning versus `<=0.05`.

Do not relax either gate or add post-hoc updates. Keep the E10 encoder frozen
and routing closed. The next experiment must isolate mode-resolved generator
identifiability and optimization.

## Frozen protocol

The committed contract
`docs/research/2026-07-25-canonical-latent-e12-structured-generator-contract.md`
froze before sampled-state access:

- the qualified E10 ordered 52-coefficient representation, with 49 active
  periodic modes;
- the exact E11 states, teachers, seeds, splits, schedules, and controls;
- one representation-blind generator
  `G(v_x,v_y,nu)=v_x A_x+v_y A_y+nu D`;
- skew-symmetric advection matrices, a diagonal strictly dissipative diffusion
  matrix, a fixed constant mode, and copied inactive trend coefficients;
- exactly 2,304 learned parameters;
- combined matrix-exponential training only;
- checkpoint-identical symmetric splitting as an evaluation diagnostic only;
- analytic Fourier generators as a nonlearned oracle ceiling;
- literal generator-identification, high-frequency, transfer, temporal,
  semigroup, cross-observation, physics, provenance, replication, and boundary
  gates.

Independent pre-state review corrected underspecified training, classification,
coverage, schedule-hash, and generator-identification clauses before any
sampled state was generated.

Two later execution heads are superseded implementation attempts, not
scientific evidence:

- `33efde71e82be09ae089abd1840ad967c4d301be` stopped without a result because
  its coverage validator incorrectly assumed four deterministic grid
  realizations instead of E10's one;
- `b198f5a7a3c85c4f0e2aa1da488a3d27e0ef2831` stopped without a result because
  derived learned matrices were not detached before provenance hashing.

Both defects were corrected without changing model values, data, schedules,
metrics, thresholds, or classifications. The accepted run executed from clean
HEAD `9b9ac597ee013c23c4f0af1971e2714fb7901bd2`.

## Oracle and representation controls

| Preflight metric | Observed | Gate |
| --- | ---: | ---: |
| Minimum projection rank | `52` | `52` |
| Maximum one-step oracle decoded NRMSE | `9.87e-16` | `<=1e-10` |
| Maximum eight-step oracle decoded NRMSE | `5.24e-15` | `<=1e-10` |
| Oracle combined/splitting mismatch | `6.27e-15` | `<=1e-10` |
| Oracle combined semigroup mismatch | `1.24e-15` | `<=1e-10` |
| Oracle identification error | `0` | `<=1e-12` |

The analytic generator, Fourier teacher, projection, and exponentiation path
close near float64 precision. Every learned checkpoint preserves exact skew
symmetry, diagonal dissipation, the constant mode, inactive modes, finiteness,
and the 2,304-parameter limit by construction.

Post-operator cross-observation mismatch for the candidate is
`1.54e-14` in coefficients and `6.25e-15` decoded across all E10 grids, warped
meshes, uniform particles, and warped particles. The encoder and observation
path are not the blocker.

## Operator evidence

Composite validation under the primary combined rule:

| Arm | One-step decoded NRMSE | Eight-step decoded NRMSE | Final HF NRMSE |
| --- | ---: | ---: | ---: |
| Elementary-pretrained zero-shot | — | `0.005364` | `0.261281` |
| Pretrained plus eight-shot | `0.003167` | `0.005735` | `0.295429` |
| Scratch eight-shot | — | `1.223040` | `27.3091` |
| Full composite control | — | `0.012474` | `0.632480` |
| Persistence | — | `1.558448` | `12.2561` |
| Exact projected truth | `0` | `0` | `0` |
| Oracle combined | approximately `0` | `1.42e-14` | `8.33e-13` |

The pretrained candidate is:

- `0.004689x` scratch few-shot;
- `0.459731x` full composite training;
- `0.002337x` the accepted E11 dense pretrained result;
- zero-shot `0.003442x` persistence and `0.002184x` E11 dense zero-shot.

This is decisive positive elementary-to-composite transfer under the structured
operator. Fine-tuning retains elementary performance: macro decoded rollout
NRMSE moves from `0.020116` to `0.018876` (`0.938377x`).

Temporal extrapolation's worst coefficient/decoded NRMSE is `0.050053`;
combined semigroup mismatch is `1.01e-15`; advection mean and norm drift remain
near numerical precision; and diffusion energy monotonicity is `1.0`.

The frozen high-frequency gate still fails. Candidate composite amplitude ratio
is `1.08942`, while x/y advection amplitude ratios remain near one. The error is
therefore residual modal action and phase fidelity, not energy collapse.

## Generator identification

| Diagnostic | Elementary-pretrained | Pretrained few-shot | Gate |
| --- | ---: | ---: | ---: |
| Worst relative Frobenius error | `0.06677` | `0.06242` | `<=0.10` |
| Worst supported-entry relative error | `0.08536` | `0.07813` | `<=0.20` |
| Worst off-support leakage | `0.01719` | `0.02751` | `<=0.10` |
| Maximum diffusion-rate relative error | `1.67e-13` | `0.02263` | `<=0.20` |
| Maximum normalized commutator | `0.00559` | `0.00788` | `<=0.02` |
| Maximum decoded basis-action NRMSE | `0.09992` | `0.10841` | `<=0.05` |

Every identification subgate except literal basis action passes. Small
distributed advection-generator errors are sufficient to produce unacceptable
worst-mode phase error even while aggregate rollouts are excellent. The
full-data control also has worse generator identification and spectral error,
so more composite examples alone do not resolve the mechanism.

## Splitting diagnostic

The checkpoint-identical splitting rule reaches composite decoded rollout
NRMSE `0.005720` and high-frequency NRMSE `0.294490`. Its worst semigroup
mismatch is `5.12e-05`, still inside the frozen gate but worse than the combined
exponential. No failed combined gate becomes passing under splitting.

Splitting is therefore not a repair or alternate qualification route. The
learned generators are already close to commuting, and the remaining error is
in their identified modal action.

## Reproducibility and coverage

Raw replicates are byte-identical at SHA-256
`66390e7385fe3193ceb87bb7792e7ffd6926114cd271daf104ffeb045d7bfdfd`.
Both complete results and the top-level result are byte-identical at
`7b214639d6287ede84352bbcfe7a31a1c9e891f4080fc9c623aee5df5e9c5ccc`.
The detached manifest SHA-256 is
`a5bb4feee2ab513d050a9acd9bf6b51e3ba989e57275eec5240cb01708f0cf5a`.

Runner SHA-256 is
`8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a`;
contract SHA-256 is
`46e2fa7f68f31ccbdd88332e65d0d14d3aa39952d8e5b242d92d2b0b95985b8d`;
and config SHA-256 is
`cd428d490ad9d5505f88ead66b41fdb25e25830f45d0eb21f451c5dbea261934`.

All 13 seeded datasets independently reproduce their state, parameter, and
complete-trajectory hashes. All five frozen schedule hashes match. The result
retains 48 base cells, 48 temporal cells, 48 semigroup cells, 36 physics cells,
16 composition-gap cells, and both cross-observation rules with every E10
realization pair.

The compact artifact is
`docs/research/artifacts/canonical_latent_e12_structured_generator_result.json`
at SHA-256
`d4760ec3d69b4397cc14ffc3bb08edd3f073edcaf1f6d4dd70db070a96cab3b2`.
Independent result review rederived the hashes, seeded data, schedules,
coverage, gates, and classification.

Held-out reads, provider calls, routes, labels, and original observations after
projection are zero. This result covers only smooth synthetic periodic scalar
linear advection-diffusion. It neither qualifies nor refutes arbitrary domains,
topology, shocks, particles, coupled fields, nonlinear physics, or a universal
simulator.

## Next gate

Preregister a mode-resolved identifiability and optimization audit before any
broader physics:

1. retain E10, E11/E12 data, the analytic oracle, and every frozen scientific
   boundary;
2. record the exact worst basis index and physical parameter case, plus
   per-mode phase and amplitude error;
3. add a closed-form or sparse oracle-support recovery control to determine
   whether the data identify the correct generators independently of AdamW;
4. compare that ceiling with the current full skew-matrix parameterization
   under the unchanged high-frequency and basis-action gates;
5. do not change thresholds, add post-hoc E12 updates, reopen the encoder, or
   introduce routing.

If direct or support-constrained recovery passes, E12 isolated optimization or
overparameterized identification. If it fails under the same data, revise the
elementary excitation design before nonlinear expansion.
