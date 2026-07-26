# Canonical Latent E12 Structured-Generator Contract

Date: 2026-07-25
Status: frozen before state-level measurement

Pre-measurement revision: independent review of commit `8b1025e` found
underspecified training-rule, classification, identification, and evidence
coverage clauses. This revision supersedes that draft before any sampled state
was generated or read.

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

The E12 result must import and hash-verify the following accepted E11
comparators:

- elementary-pretrained zero-shot decoded rollout NRMSE `2.456423109290437`;
- pretrained-fewshot decoded rollout NRMSE `2.4541693117065217`;
- scratch few-shot decoded rollout NRMSE `1.5126324830652857`;
- full-data decoded rollout NRMSE `0.9381449992978823`;
- persistence decoded rollout NRMSE `1.5584481380508215`.

These are locked historical comparisons, not E12 evaluation arms or retrained
controls.

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

Every learned checkpoint is optimized through the **combined exponential rule
only**. Splitting never defines a training loss, optimizer update, checkpoint,
or selection criterion. It is checkpoint-identical post-training evaluation
of the same learned generators.

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

The literal required-cell schema is:

- `base[12 evaluation arms][4 regimes] = 48` records;
- `temporal_extrapolation[12 evaluation arms][4 regimes] = 48` records;
- `semigroup[12 evaluation arms][4 regimes] = 48` records;
- `physics[12 evaluation arms][3 regimes] = 36` records;
- `cross_observation[2 learned rules]` with every E10 family-realization pair;
- `composition_gap[4 learned checkpoints][4 regimes] = 16` records.

The 12 evaluation arms are exactly:

1. elementary-pretrained combined;
2. elementary-pretrained splitting;
3. pretrained-fewshot combined;
4. pretrained-fewshot splitting;
5. scratch-fewshot combined;
6. scratch-fewshot splitting;
7. full-composite combined;
8. full-composite splitting;
9. persistence;
10. exact projected truth;
11. oracle combined;
12. oracle splitting.

No `applicable` or optional cell is permitted inside these schemas.

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

Hash every generated coefficient, parameter, and complete trajectory tensor
for every training, validation, elementary, composite, and extrapolation
dataset. Hash every batch schedule.

Schedule tensors are generated by the accepted E11 `schedule` function with
the frozen E11 seeds and population sizes. They are C-contiguous little-endian
signed 64-bit integer arrays, and the SHA-256 input is exactly their raw array
bytes with no header or metadata. Before optimization require the exact shapes
and hashes:

- x-advection pretraining: `[1500, 32]`,
  `096659f791a0d5e728ccac7aa02801c60856bd0659a080464bb158909be3f6f7`;
- y-advection pretraining: `[1500, 32]`,
  `49401932ae228e58f2927a7695d443a17a1a41c8e80375e4346a06973b431507`;
- diffusion pretraining: `[1500, 32]`,
  `d86f8d73a236e806ff1988c274af8fd9a2daa30beeada74e8427d526d5b88487`;
- composite fine-tuning: `[400, 64]`,
  `b0de4bcb3d59866dd05489d7ecf13574dd7763ddf82a6706f72734ec701a7e32`;
- full composite control: `[1500, 96]`,
  `602ff2694d9923821782d69627c0b7ece849086abcc431f4ca085de6486519cf`.

## Frozen generator-identification diagnostics

The learned checkpoint is not qualified merely because its parameterization is
skew/dissipative or its combined exponential is a semigroup. Those properties
are architectural.

For elementary-pretrained and pretrained-fewshot checkpoints, require:

1. relative Frobenius error
   `||A-A_star||_F / ||A_star||_F` of each of `A_x`, `A_y`, and `D` against
   the analytic oracle `<=0.10`;
2. define the signed derivative support separately for each advection
   generator as `S={i,j: A_star[i,j] != 0}`; require
   `max_(i,j in S) |A[i,j]-A_star[i,j]| / |A_star[i,j]| <=0.20`;
3. define off-support as the complement of `S`, including the diagonal, and
   require
   `||A[not S]||_F / ||A_star||_F <=0.10` for each advection generator;
4. over the 48 nonconstant diagonal indices require
   `max_i |D[i,i]-D_star[i,i]| / |D_star[i,i]| <=0.20`;
5. normalized commutator norms
   `||[A_x,A_y]||_F/(||A_x||_F ||A_y||_F)`,
   `||[A_x,D]||_F/(||A_x||_F ||D||_F)`, and
   `||[A_y,D]||_F/(||A_y||_F ||D||_F)` each `<=0.02`;
6. maximum one-step decoded NRMSE over all 49 active basis vectors and all E11
   closure parameter cases `<=0.05`.

Record the same diagnostics for scratch and full-data checkpoints without
gating them. Oracle diagnostic error must be numerical zero under `1e-12`.

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
15. all generator-identification gates pass;
16. candidate decoded rollout NRMSE is `<=0.25x` the locked E11 dense
    pretrained-fewshot value, and structured zero-shot decoded rollout NRMSE is
    `<=0.25x` the locked E11 dense zero-shot value;
17. all structure, finiteness, provenance, coverage, replication, and boundary
    assertions pass.

Apply the same accuracy, transfer, temporal, cross-observation, and physics
metrics to the splitting diagnostic, but do not use it as an alternate
qualification route.

Require the decoded combined-versus-splitting rollout mismatch for the
pretrained-fewshot checkpoint to be `<=0.05` on every regime.

### Classifications

Classify `structured_additive_generator_qualified` if every combined primary
gate passes. The shared gate set is exactly closure, oracle correctness,
generator identification, parameter structure/count, source/data/schedule
provenance, Cartesian coverage, complete-result replication, and boundary.
The combined rule-specific gate set is exactly accuracy, high-frequency,
zero-shot composition, transfer versus structured scratch/full data, E11
improvement, retention, temporal extrapolation, semigroup measurement,
cross-observation invariance, physics, and combined-versus-splitting mismatch.

Classify `structured_generator_capable_without_transfer` only if every shared
gate and every combined rule-specific gate except the structured
scratch/full-data positive-transfer ratios passes.

Otherwise classify `structured_generator_not_qualified`.

If splitting passes a metric that combined misses, record
`splitting_diagnostic_improves_failed_combined=true` and the exact deltas. This
is evidence of noncommuting learned-generator or numerical-integration error,
not a scientific qualification.

## Reproducibility and boundary

Require:

- exact frozen configuration;
- before constructing, generating, or reading any sampled coefficient,
  parameter, or target tensor: exact config equality, contract and runner byte
  equality to committed Git HEAD, clean worktree, 40-character Git HEAD, and
  locked E11 artifact hash equality;
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
