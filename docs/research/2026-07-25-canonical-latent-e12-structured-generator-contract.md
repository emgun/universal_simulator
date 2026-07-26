# Canonical Latent E12 Structured-Generator Contract

Date: 2026-07-25
Status: frozen before state-level measurement

## Question

E11 showed that a generic parameter-conditioned residual MLP can fit scheduled
transitions yet fail disjoint-state rollouts, semigroup consistency, physical
structure, and elementary-to-composite transfer.

Can one representation-blind continuous-time coefficient operator learn
elementary advection and diffusion generators, compose them without task
routing, and meet the frozen E11 accuracy and transfer gates?

E12 tests operator inductive bias inside the qualified E10 coefficient space.
It does not reopen the encoder, expand the state basis, change the teacher, or
introduce family/task experts.

## Research basis

Fixed-basis coefficient-to-coefficient operator learning separates projection
error from coefficient-map error:
<https://arxiv.org/abs/2510.10350>.

Recent neural-operator work supports explicit composition or splitting of
elementary dynamics to generalize to unseen physics:
<https://arxiv.org/abs/2602.00884> and
<https://openreview.net/pdf?id=VFLwvtTwWF>.

Recent structure-preserving neural differential-equation work uses constrained
linear generators and exponential integration to improve stability:
<https://arxiv.org/abs/2503.01775>.

Energy-consistent neural operators likewise motivate making conservative and
dissipative structure architectural rather than hoping it emerges from a
generic loss:
<https://openreview.net/forum?id=8YFRJdr3CS>.

These sources identify a narrower challenger than another MLP: learn the
infinitesimal generators in the semantic latent and exponentiate their physical
combination.

## Frozen state, teacher, and splits

Reuse E11 exactly:

- ordered 52-dimensional E7/E10 basis;
- first 49 periodic modes active; constant coefficient at index `0`;
- final three nonperiodic trend coefficients fixed to zero;
- identical coefficient distribution and modal scales;
- exact float64 `64 x 64` Fourier teacher for
  `u_t + v_x u_x + v_y u_y = nu Delta u`;
- eight transitions per trajectory;
- identical state seeds, parameter seeds, physical marginals, `dt` interval,
  temporal-extrapolation `dt=0.075`, geometry seeds, and split counts.

Training and validation identities remain:

- 256 elementary trajectories per x-advection, y-advection, and diffusion;
- eight composite few-shot trajectories;
- 256 composite full-data control trajectories;
- 64 validation trajectories;
- zero reserved held-out reads.

The accepted E11 compact artifact is a historical control and is bound at
SHA-256
`d9142f94c87d5ffb0b44ff67d665b46b50e529a6d50acee703d15d20e8445f15`.
Do not retrain, reinterpret, or overwrite its dense arms.

## Frozen structured operator

For the active coefficient vector `c in R^49`, learn

`G(v_x, v_y, nu) = v_x A_x + v_y A_y + nu D`.

Constraints:

- `A_x` and `A_y` are real skew-symmetric `49 x 49` matrices;
- row and column `0` of both advection generators are identically zero, so the
  constant mode cannot move;
- each advection generator stores only the strict upper triangle over the 48
  nonconstant active modes and constructs the lower triangle by antisymmetry;
- `D` is diagonal with `D[0,0]=0`;
- the remaining diffusion diagonal is `-exp(log_rate)`, so every learned rate
  is strictly dissipative;
- the three inactive trend coefficients are copied unchanged and never enter
  a generator;
- no representation, family, task, regime, source index, resolution, or
  routing input exists.

The learned parameter count is exactly
`2 * (48 * 47 / 2) + 48 = 2304`, versus E11's 19,828 parameters.

Initialize both skew-generator vectors to zero and every diffusion log-rate to
zero. This makes initial advection persistence and initial diffusion rate one.
All learned arms start from the identical state dict. Use float64, one CPU
thread, and deterministic algorithms.

### Combined rule

The primary step is

`c(t + dt) = exp(dt G(v_x, v_y, nu)) c(t)`.

### Symmetric splitting challenger

Using the identical checkpoint, also evaluate

`exp(dt nu D / 2)`
`exp(dt v_y A_y / 2)`
`exp(dt v_x A_x)`
`exp(dt v_y A_y / 2)`
`exp(dt nu D / 2) c`.

No new splitting parameters or optimization are allowed. The ordering is
frozen as written.

### Exact oracle

Construct nonlearned real coefficient generators directly from the known
Fourier derivative and Laplacian actions in the ordered E7 basis. Record both
combined and splitting oracle arms. The oracle is an implementation and
hypothesis-class ceiling, not training data or an input to a learned model.

## Frozen preflight

Before reading any sampled state:

1. verify the learned parameter count and all exact structural identities;
2. verify the oracle matrices preserve the constant and inactive modes;
3. require oracle advection skew residual `<=1e-12`;
4. require oracle diffusion off-diagonal residual `<=1e-12` and maximum
   eigenvalue `<=1e-12`;
5. on all 49 active basis vectors and the E11 closure parameter cases, compare
   oracle combined and splitting steps with the exact Fourier teacher for one
   and eight steps;
6. require finite coefficients, projection rank `52`, decoded NRMSE
   `<=1e-10`, combined semigroup composition error `<=1e-10`, and
   oracle combined/splitting mismatch `<=1e-10`.

If preflight fails, read no training or validation state, run no optimizer, and
classify `structured_generator_not_qualified`.

## Frozen training

Reuse the exact E11 transition schedules and seeds. Generate and hash every
schedule before optimization.

All learned arms minimize the E11 modal-scale-normalized one-step coefficient
MSE with AdamW, zero weight decay, and no early stopping, validation selection,
gradient clipping, retry, seed change, or threshold change.

Arms:

1. `structured_elementary_pretrained`: balanced elementary batches, 1,500
   updates, 32 transitions per regime per update, learning rate `2e-2`.
2. `structured_pretrained_fewshot`: reset optimizer state and fine-tune arm 1
   on the eight composite trajectories for 400 updates, batch size 64,
   learning rate `5e-3`.
3. `structured_scratch_fewshot`: identical initial state and exact same 400
   composite batches as arm 2, learning rate `5e-3`.
4. `structured_full_composite_control`: identical initial state, 256 composite
   trajectories, 1,500 updates, batch size 96, learning rate `2e-2`.

Each checkpoint is evaluated under both the combined and splitting rules.
Persistence, exact projected truth, oracle combined, and oracle splitting are
nonlearned arms.

## Frozen evaluation coverage

The executable must fail before result materialization unless it retains this
complete Cartesian evidence:

- learned checkpoints: elementary-pretrained, pretrained-fewshot,
  scratch-fewshot, and full-composite;
- composition rules: combined and splitting;
- nonlearned arms: persistence, exact projected truth, oracle combined, and
  oracle splitting;
- base regimes: composite, x-advection, y-advection, diffusion;
- temporal-extrapolation regimes: the same four at `dt=0.075`;
- semigroup regimes: the same four;
- physics records: x-advection, y-advection, and diffusion for every arm.

For every evaluation arm and applicable regime, record the full E11 metric
surface: one-step and eight-step coefficient/decoded NRMSE, error by rollout
step and parameter quartile, final high-frequency NRMSE/amplitude ratio,
maximum absolute error, finite status, effective coefficient rank, temporal
extrapolation, semigroup consistency, mean-mode error, advection `L2` drift,
and diffusion monotonic-energy fraction.

Additionally record:

- learned and oracle generator SHA-256 values;
- skew, constant-mode, inactive-mode, diagonal, and dissipativity residuals;
- Frobenius error of each learned generator against its oracle;
- commutator norms `[A_x,A_y]`, `[A_x,D]`, and `[A_y,D]`;
- combined-versus-splitting coefficient and decoded mismatch for every learned
  checkpoint and regime;
- E10 cross-observation mismatch for pretrained-fewshot under both rules,
  gating every Cartesian family-realization pair;
- all model, schedule, config, contract, runner, Git, raw-result,
  complete-result, and detached-manifest hashes.

## Frozen gates

### Shared scientific gates

For the primary `structured_pretrained_fewshot` combined arm require:

1. one-step coefficient and decoded NRMSE each `<=0.03`;
2. eight-step coefficient and decoded NRMSE each `<=0.08`;
3. final high-frequency NRMSE `<=0.15`;
4. decoded rollout NRMSE `<=0.80x` structured scratch few-shot combined;
5. decoded rollout NRMSE `<=1.25x` structured full-data combined;
6. elementary-pretrained combined zero-shot decoded rollout NRMSE `<=0.20`
   and `<=0.75x` persistence;
7. post-fine-tuning elementary macro decoded rollout NRMSE `<=0.08` and
   `<=1.25x` its pre-fine-tuning value;
8. worst coefficient and decoded `dt=0.075` rollout NRMSE across all four
   regimes each `<=0.12`;
9. worst coefficient and decoded semigroup NRMSE across all four regimes each
   `<=0.05`;
10. worst cross-observation coefficient and decoded mismatch each `<=0.01`;
11. x/y advection mean-mode relative error each `<=1e-3`;
12. x/y advection relative `L2` drift each `<=0.05`;
13. diffusion nonincreasing-energy fraction `>=0.99`;
14. learned parameter count `<=2304`;
15. all structure, finiteness, provenance, coverage, replication, and boundary
    assertions pass.

The same absolute, zero-shot, temporal, semigroup, cross-observation, physics,
and structure gates apply to the splitting challenger. Its transfer ratios use
the corresponding splitting scratch and full-data arms.

Require the decoded combined-versus-splitting rollout mismatch for the
pretrained-fewshot checkpoint to be `<=0.05` on every regime.

### Classifications

Classify `structured_additive_generator_qualified` if every combined primary
gate passes.

Classify `structured_splitting_only_qualified` if the splitting challenger
passes every corresponding gate, all shared gates pass, and the combined arm
misses at least one accuracy or transfer gate.

Classify `structured_generator_capable_without_transfer` if either composition
rule passes closure, absolute accuracy, temporal, semigroup, invariance,
physics, structure, provenance, and boundary gates but misses a positive
transfer ratio.

Otherwise classify `structured_generator_not_qualified`.

## Reproducibility and boundary

Require:

- exact frozen configuration;
- contract and runner byte equality to a clean committed Git HEAD before any
  state access;
- two complete byte-identical result files containing the final decision;
- a detached manifest binding both complete hashes and execution HEAD;
- no nonfinite numeric value;
- zero held-out reads and provider calls;
- zero task/representation labels, routes, experts, source bypasses, or
  original observations after projection.

E12 covers only smooth synthetic periodic scalar linear advection-diffusion on
the normalized square. Success would qualify this structured generator only
for the frozen basis and teacher and would permit a separately preregistered
nonlinear coefficient-dynamics expansion. It would not qualify arbitrary
domains, topology, shocks, moving particles, coupled fields, or a universal
simulator.

Failure with passing oracle and closure evidence closes this learned structured
generator/training package and requires an identifiability or optimization
audit before any broader physics expansion. Failure does not reopen routing.
