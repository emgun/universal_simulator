# Canonical Latent E15 Training-Package Contract

Date: 2026-07-26
Status: preregistered after independent GO and before sampled-state access

## Decision context

E14 independently sealed E13 as
`full_parameterization_deterministic_recovery_succeeds`.
Within the frozen smooth periodic linear advection-diffusion family:

- the E10 representation closes across grids, meshes, and particles;
- the E12 elementary data have full required excitation rank;
- the 2,304-parameter full-skew generator class recovers under deterministic
  full-population L-BFGS from both neutral and E12 checkpoints; and
- the exact E12 ordered mini-batch AdamW replay fails basis-action and
  high-frequency recovery.

E13 changed optimizer, batch aggregation, and componentwise execution at once.
The elementary objective is separable by `A_x`, `A_y`, and `D`: each
elementary regime activates only its matching physical coefficient, and the
parameter tensors are disjoint. Objective separability does not make three
independent L-BFGS line searches, curvature histories, stopping decisions, or
budgets algorithmically equivalent to one joint L-BFGS run. E15 therefore
treats E13's componentwise strong-Wolfe L-BFGS routine as a frozen training
package, verifies its objective implementation, and does not call it a pure
optimizer control.

The remaining high-signal question is whether E12 failed because of:

1. ordered stochastic mini-batch gradients;
2. AdamW's optimization geometry on the deterministic objective; or
3. the modest nonuniform sample weighting induced by E12's with-replacement
   schedule.

E15 is a causal training-package audit, not a new representation, operator
family, or deployment qualification.

## Question

Using the exact E12 initialization, checkpoint, data, schedule multiset,
optimizer hyperparameters, full-skew parameterization, and recovery gates:

1. does AdamW recover when every update uses the deterministic objective
   induced by the complete frozen schedule multiset;
2. if not, does the frozen componentwise strong-Wolfe L-BFGS package recover
   on that same schedule-weighted objective; and
3. if a schedule-weighted componentwise L-BFGS arm fails while its
   initialization-matched sealed uniform-population componentwise L-BFGS
   ceiling passes, is schedule weighting the remaining distinction within
   that frozen package?

No result may be called a pure batching or optimizer effect unless the
corresponding comparison holds the other registered factors fixed.

## Research basis

Recent operator-learning research reports that spectral bias is coupled to
loss design and optimization dynamics rather than being purely
representational, and that second-order methods can materially change the
order in which frequencies are learned:
<https://arxiv.org/abs/2602.19265>.

Random-reshuffling analysis for least-squares problems shows that mini-batch
dynamics can differ from full-batch gradient dynamics beyond leading order and
can induce spectrum-dependent shrinkage:
<https://openreview.net/forum?id=XkyyvBcvA9>.
E15 does not assume that theorem transfers directly to the matrix-exponential
objective; it motivates measuring schedule-induced weighting and
frequency-resolved learning traces.

Standard deterministic BFGS curvature equations are not automatically valid
under noisy gradients, and recent stochastic BFGS work introduces a distinct
Bayesian construction to handle that case:
<https://arxiv.org/abs/2507.07729>.
E15 therefore does not invent a "mini-batch L-BFGS" corner with PyTorch's
deterministic line search.

Structured Lie-generator work motivates retaining the skew-minus-dissipative
matrix exponential rather than changing the operator while auditing training:
<https://arxiv.org/abs/2603.27442>.

These sources motivate the diagnostics and control separation. They do not
prejudge the E15 classification.

## Frozen provenance

The executable must bind and hash-verify:

- this contract;
- the E15 runner and tests;
- the E13 contract and runner;
- the E14 contract and sealer;
- the accepted E12 replay lock;
- all E12/E11/E7 implementation dependencies imported by E13;
- the committed E14 evidence bundle, compact result, detached manifest, result
  record, and steward handoff; and
- a clean committed Git HEAD containing byte-identical sources.

Require these E14 hashes:

- bundle:
  `a4886e3b3a8c678abe7b8f44907b8655af4b0ac68fb47ca9353ae2bfca677b5c`;
- compact result:
  `a26b4948a0db7fd7aa1a0c067bf29eaacd3007e4c93e1bb309a5c610cad04413`;
- detached manifest:
  `88f45edfa0c456ec19eef8da8167ed0f2bf6213f13f00ef6f32d934c4cf9257e`.

Independently reopen the E14 bundle and require its three members to match the
manifest byte counts and raw hashes. Require E14 classification
`e13_scientific_result_sealed`, underlying classification
`full_parameterization_deterministic_recovery_succeeds`, sealing HEAD
`63e23d50f2eef3ca2644100674c10c4e2aa3a5ba`, all independent sections passing,
and every zero-state boundary remaining zero.

Require exact Python `3.12.7` and PyTorch `2.7.0`. Before any tensor or sampled
state is constructed, set and verify:

- local CPU and float64 only;
- `torch.set_num_threads(1)`;
- `torch.set_num_interop_threads(1)`;
- `torch.use_deterministic_algorithms(True)`; and
- the frozen E12 model seed.

Record Python, PyTorch, device, dtype, deterministic-algorithm state, both
thread counts, and every optimizer constructor field. Any source, Git,
artifact, environment, or boundary failure stops before sampled-state access
as `e15_preflight_failed`.

## Frozen representation, teacher, data, and schedules

Reuse E12/E13 exactly:

- ordered 52-dimensional E10 coefficient basis;
- first 49 periodic coefficients active, including constant index `0`;
- last three trend coefficients inactive and copied;
- exact float64 `64 x 64` Fourier teacher;
- 256 trajectories and 2,048 transitions in each elementary regime;
- exact E12 state and parameter seeds;
- exact E12 validation data and parameter cases;
- exact E12 schedule seeds, shapes, indices, and hashes;
- full-skew `StructuredGenerator` with 1,128 `A_x`, 1,128 `A_y`, and 48
  diffusion parameters;
- no new state, parameter, geometry, or held-out seed.

Before any new optimizer runs, reproduce:

- all seven E12/E13 dataset records and five schedule records;
- initial model SHA-256
  `64c87294711c68c9bc4a9f56cb3f8a8ca23b1e1eed84493bfc18e18f3a2c9218`;
- E12 elementary checkpoint SHA-256
  `e9c17bc1871f5b2008d3899da9f59a44cf1209448f50ba576ca63f6281602e7b`;
- E12 generator hashes
  `17f70896f48f5651854746c5be9edc7dd42d2fd331c1fbc7d2c6e2edd1a46e53`,
  `726d18845ad3dc3e462fc16578adf08929594e57dd02818c1e13711eee31fda2`,
  and
  `1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370`;
- E12 first/final losses
  `0.08166621133143402` / `0.00032425428259914163`;
- 1,500 updates and 48,000 scheduled examples per regime;
- E12 basis-action NRMSE `0.09992145782297766`;
- E12 composite rollout NRMSE `0.00536402045663299`.

Any mismatch stops as `e12_reproduction_failed`.

## Frozen schedule-collapse preflight

For each elementary regime, flatten its exact `[1500,32]` schedule and count
the occurrence of every transition index in a length-2,048 int64 vector.
Serialize little-endian contiguous int64 bytes.

Require:

| Regime | Count SHA-256 | Total | Min | Max | Zeros |
| --- | --- | ---: | ---: | ---: | ---: |
| x advection | `1048116699e92f3de058114f01552707497a7fcd31e2a1bb288dbed2fdd0e5b5` | `48,000` | `10` | `46` | `0` |
| y advection | `c2208664ec391862db76668b1084b7ff8b2f08dc39a43259527f38cbcbf75b93` | `48,000` | `9` | `41` | `0` |
| diffusion | `3c62df055b01246b7a88ba8fa187f642b441645f9ce056c77f91272fa6c8a1a3` | `48,000` | `8` | `40` | `0` |

Record, but do not gate after inspecting, the frozen imbalance diagnostics:

| Regime | L1 from uniform | Coefficient of variation |
| --- | ---: | ---: |
| x advection | `0.16350781250000002` | `0.2067956586692391` |
| y advection | `0.16267708333333333` | `0.2041132583205271` |
| diffusion | `0.16257291666666668` | `0.20242090361971565` |

For transition `i`, let `e_i(theta)` be its mean squared normalized modal
error over coefficients and channels, and let `c_i` be its schedule count.
Freeze:

`L_r(theta) = sum_i c_i e_i(theta) / 48000`

and

`L_schedule(theta) = (L_x(theta) + L_y(theta) + L_D(theta)) / 3`.

Independently verify that `L_schedule` equals the arithmetic mean of all 1,500
literal E12 mini-batch losses at the neutral model, E12 checkpoint, and oracle
within `1e-14` absolute and relative tolerance. This equality check may use
chunked evaluation but may not construct a new sample.

Also compute the uniform full-population objective

`L_uniform(theta) = (mean_i e_i^x + mean_i e_i^y + mean_i e_i^D) / 3`

for every checkpoint. Schedule weighting is a diagnostic factor only; no
weights may be clipped, smoothed, balanced, or retuned.

## Frozen loss implementation and separability preflight

The executable must reuse E13's componentwise loss and optimizer routine. The
only permitted loss change inside the schedule-weighted L-BFGS arms is replacing
each uniform transition weight by its frozen schedule occurrence count and
dividing by `48,000`. The component order remains `A_x`, `A_y`, then `D`.
Within every regime, process trajectory groups in ascending trajectory index
and their eight transitions in ascending horizon order. Accumulate products and
sums in float64 with a fixed chunk size of 32 trajectory groups. Do not use
unordered or parallel reductions.

For feasibility, the registered implementation groups the 2,048 transitions
as 256 trajectories with eight transitions each. Because physical parameters
and `dt` are fixed within a trajectory, compute the generator matrix and its
matrix exponential once per trajectory and apply it to all eight source
coefficient vectors. Schedule occurrence weights remain transition-specific.

Before any training, compare this grouped implementation with a literal
flattened 2,048-transition implementation at:

- the neutral model;
- the exact E12 checkpoint;
- the oracle generator; and
- one fixed source-only parameter probe made without reading new state:
  `ax_upper[0]=0.125`, `ay_upper[47]=-0.25`, and
  `diffusion_log_rate[7]=math.log(0.375)`, with every other `ax_upper`,
  `ay_upper`, and `diffusion_log_rate` trainable coordinate zero.

Require decoded output tensors, losses, and gradients to be equal within
`1e-14` absolute and relative tolerance for uniform, schedule-weighted, and
literal-schedule reductions at the three trainable-model probes; the fixed
oracle requires output and loss equality only. Repeat with all occurrence
weights set to one and require the E15 component losses and gradients to equal
E13's uniform component losses and gradients within the same tolerance. Any
mismatch is `frozen_ceiling_or_objective_integrity_failed`.

At the neutral model and E12 checkpoint, independently compute gradients of
each regime loss with respect to all three parameter blocks.

Require:

- x-advection gradients are exactly zero for `A_y` and `D`;
- y-advection gradients are exactly zero for `A_x` and `D`;
- diffusion gradients are exactly zero for `A_x` and `A_y`;
- the accumulated gradient of `L_schedule` equals the gradient obtained from
  the literal concatenated schedule-weighted objective within `1e-14` absolute
  and relative tolerance; and
- every parameter block, loss, and gradient is finite.

Only after this passes may componentwise L-BFGS execute. This proves that its
component losses implement the registered separable objective; it does not
claim algorithmic equivalence to a joint L-BFGS optimizer.

## Frozen matched E13 ceilings

Before any E15 optimizer executes, independently recompute all E13 gates and
literal metrics for both full-skew uniform-population componentwise L-BFGS
controls from the sealed E14 evidence. Require both controls to pass and bind
the following exact checkpoints.

Neutral ceiling:

- model SHA-256:
  `b1faad552a12d9e71a2ec9788cf5b9e46547cfb8ac1fc8e7cdd706af7e46208f`;
- generator SHA-256 (`A_x`, `A_y`, `D`):
  `6e9cf3c8f18b4a6613909f8b305b6cd24638a4f005322ca685fb54028b4b506c`,
  `e051059bd996fb29d07addc914346c439e62ac6061edf3d4fc80046742e145a6`,
  and
  `8bd2aa03a087dffafdce8669d9fe2200a60c45229d6dc4e510718b6ed1b8a8d6`;
- maximum basis-action decoded NRMSE `8.173630029347496e-6`;
- relative generator Frobenius errors (`A_x`, `A_y`, `D`)
  `6.220940770884322e-6`, `6.544445921150395e-6`, and
  `8.115066223371214e-8`;
- maximum elementary one-step/eight-step decoded NRMSE
  `1.0753003105436782e-6` / `7.855289738311861e-6`;
- composite eight-step decoded NRMSE `6.544271466298818e-6`; and
- composite final high-frequency NRMSE `1.7248577530222313e-5`.

E12-checkpoint restart ceiling:

- model SHA-256:
  `2044abed51748bcda79173f0a99d84c38391d3af5553aabe102e94812c5392db`;
- generator SHA-256 (`A_x`, `A_y`, `D`):
  `86bb2d1c7834cfe641973007e940f9024309bcd720424fc7c98a82b44aea3226`,
  `476163d49bde39e561e5ae28a84a0e2880bcf5e3d31aa684ee7b84fe4239b506`,
  and
  `1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370`;
- maximum basis-action decoded NRMSE `1.0640378645420527e-5`;
- relative generator Frobenius errors (`A_x`, `A_y`, `D`)
  `7.10032611811484e-6`, `8.431646563243572e-6`, and
  `1.1554429390623648e-14`;
- maximum elementary one-step/eight-step decoded NRMSE
  `5.05464355748774e-7` / `3.203318962307899e-6`;
- composite eight-step decoded NRMSE `5.817206155737696e-7`; and
- composite final high-frequency NRMSE `2.30504938101271e-5`.

Also require every recorded E13 recovery gate, generator-identification
metric, model/generator hash, structure record, and validation leaf for these
two controls to equal the sealed evidence. The neutral E15 arms may only be
compared with the neutral ceiling, and the restart E15 arms may only be
compared with the E12-checkpoint restart ceiling.

## Frozen controls

Run exactly four new full-skew controls:

1. `schedule_weighted_adamw_neutral`;
2. `schedule_weighted_adamw_restart`;
3. `schedule_weighted_componentwise_lbfgs_neutral`;
4. `schedule_weighted_componentwise_lbfgs_restart`.

Neutral is the exact initial model. Restart is the exact E12 model checkpoint
loaded into a newly constructed optimizer with no optimizer-state carryover.
No random restart, sparse support, mode tying, Fourier hard-coding, learned
router, expert, encoder update, or additional data is allowed.

### Schedule-weighted AdamW

For both AdamW controls:

- optimize all 2,304 full-skew parameters;
- use exactly `L_schedule`;
- run exactly 1,500 optimizer updates;
- use learning rate `0.02`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay
  `0`, AMSGrad false, maximize false, capturable false, differentiable false,
  foreach `None`, and fused `None`, exactly matching the E12 constructor;
- zero gradients with `set_to_none=True`;
- evaluate the three regime losses sequentially, backpropagating each divided
  by three, then take one joint optimizer step;
- do not clip gradients, schedule the learning rate, stop early, or select a
  checkpoint after seeing validation;
- retain only the final step for recovery classification.

Construct a fresh AdamW instance for each arm. Before its first update, its
state dict must contain no per-parameter moment state; moments initialize from
zero on the first step. The restart arm does not load or reconstruct E12's
AdamW moments and is not a continuation claim. Hash and record the initial
model and complete pre-update optimizer state for each arm.

This control holds initialization, optimizer family, update count, schedule
multiset, model, and loss normalization fixed relative to E12. It replaces
ordered mini-batch gradients with the exact deterministic gradient of their
aggregate objective. It uses more examples per update and is therefore an
objective-aggregation sufficiency test, not a compute-matched deployment
benchmark.

### Schedule-weighted componentwise L-BFGS

For both L-BFGS controls:

- use each registered `L_r` component, whose arithmetic mean is exactly
  `L_schedule`;
- optimize `A_x`, then `A_y`, then `D` only after separability and grouped-loss
  equivalence pass;
- use E13 settings: learning rate `1`, maximum 250 iterations, maximum 300
  evaluations, history 100, gradient tolerance `1e-12`, change tolerance
  `1e-15`, and strong-Wolfe line search;
- use deterministic complete schedule-weighted closures;
- run every component once with no retry, warm restart, or threshold-based
  retuning.

This exactly reuses E13's component order, independent line searches,
curvature histories, stopping rules, budgets, and optimizer routine while
changing only uniform weights to schedule occurrence weights. It is a
componentwise strong-Wolfe L-BFGS package comparison, not a pure optimizer
comparison with joint AdamW.

## Frozen training and spectral traces

For AdamW record steps:

`{0,1,2,5,10,25,50,100,250,500,1000,1500}`.

At each registered step record:

- `L_schedule` and `L_uniform`;
- per-regime weighted and uniform losses;
- pre-update and post-update gradient norms and the update norm for `A_x`,
  `A_y`, and `D`;
- generator relative Frobenius errors;
- maximum decoded basis-action NRMSE;
- per-semantic-frequency decoded basis-action maxima in the complete `4 x 4`
  `(f_x,f_y)` table for `f_x,f_y in {0,1,2,3}`;
- composite rollout and final high-frequency NRMSE on the frozen validation
  set.

The one-dimensional semantic-frequency map for each seven-component axis is
`(0,1,1,2,2,3,3)`; basis index `7*x_index+y_index` maps to
`(f_x,f_y)`. Every one of the 16 cells must be present even when several basis
components share a cell.

Step `0` is measured before optimization: its update norm is exactly zero, its
pre-update gradient is the initial gradient, and its post-update gradient is
the same recorded tensor. For `k >= 1`, trace step `k` immediately after update
`k`; record the gradient used immediately before that update, the parameter
delta caused by that update, and a freshly recomputed gradient immediately
after it. Intermediate validation traces are diagnostic only and may not
select, stop, restart, or alter training.

For L-BFGS retain every closure loss and gradient norm plus the same diagnostic
summary before and after each component. Record function evaluations and
iterations exactly.

Each raw replicate must record the frozen validation cases and horizons used
for every trace row. Traces may not include wall-clock time, absolute or
temporary paths, process IDs, host names, replica labels, or other
execution-specific fields in the byte-identical scientific payload.

## Frozen recovery gates

For each new control independently retain every E13 recovery gate:

- all finite;
- exact 2,304-parameter full-skew structure;
- maximum relative generator Frobenius error `<=0.10`;
- supported-entry relative error `<=0.20`;
- off-support leakage `<=0.10`;
- diffusion-rate relative error `<=0.20`;
- normalized commutator `<=0.02`;
- decoded basis-action NRMSE `<=0.05`;
- composite final high-frequency NRMSE `<=0.15`;
- elementary one-step NRMSE `<=0.03`;
- elementary eight-step NRMSE `<=0.08`;
- composite eight-step NRMSE `<=0.20`;
- composite/persistence ratio using `1.5584481380508215` `<=0.75`.

Recompute all gates from literal metrics. Do not use training loss as a
qualification substitute.

Retain E13's mode-resolved coverage: all 49 basis indices, all 18 parameter
cases, horizons `{1,8}`, and every registered metric for all six E15
evaluation controls:

- E12 AdamW replay;
- four new schedule-weighted controls;
- oracle.

Require 10,584 unique mode keys and 30 literal argmax records. Preserve the
full-skew basis-41, `composite_c`, horizon-8 near-extinguished-mode diagnostic
without adding a post-hoc gate.

## Frozen classification

Apply this precedence:

1. source/artifact/environment preflight failure:
   `e15_preflight_failed`;
2. E12 replay mismatch:
   `e12_reproduction_failed`;
3. either sealed uniform-population ceiling does not independently reproduce
   and pass, or any E14/E13 ceiling, schedule-collapse, grouped/literal
   objective, all-ones E13-equivalence, or separability integrity check fails:
   `frozen_ceiling_or_objective_integrity_failed`;
4. execution starts but a timeout, resource exhaustion, nonfinite optimizer
   state, interrupted replica, publication failure, or missing required arm or
   trace prevents the complete preregistered matrix:
   `e15_execution_incomplete`;
5. `schedule_weighted_adamw_neutral` passes:
   `deterministic_objective_adamw_succeeds_from_neutral`;
6. neutral AdamW fails and `schedule_weighted_adamw_restart` passes:
   `deterministic_objective_adamw_restart_repairs_e12_checkpoint_only`;
7. both AdamW arms fail and
   `schedule_weighted_componentwise_lbfgs_neutral` passes:
   `componentwise_strong_wolfe_lbfgs_package_succeeds_from_neutral`;
8. both AdamW arms and neutral componentwise L-BFGS fail, but
   `schedule_weighted_componentwise_lbfgs_restart` passes:
   `componentwise_strong_wolfe_lbfgs_restart_repairs_e12_checkpoint_only`;
9. all four schedule-weighted arms fail, while the already-required
   initialization-matched uniform-population componentwise L-BFGS ceilings
   both pass:
   `uniform_population_weighting_required_under_frozen_componentwise_controls`.

Interpretation is deliberately bounded:

- step 5 establishes that deterministic aggregation of the exact E12 schedule
  objective is sufficient under AdamW from the same neutral start; it does not
  establish a compute-matched production recipe;
- step 6 establishes repairability by a fresh-moment deterministic AdamW
  restart from the E12 model checkpoint, not continuation, neutral-start
  recovery, or a pure noise cause;
- steps 7 and 8 establish sufficiency of the frozen componentwise
  strong-Wolfe L-BFGS package, not a universal optimizer requirement or
  algorithmic equivalence to joint L-BFGS; and
- step 9 assigns the distinction to schedule weighting only because both
  initialization-matched uniform ceilings are mandatory passing preconditions
  and the E15 routine differs from E13 only by the frozen weights.

No classification may qualify other equations, nonlinear physics, particles,
arbitrary geometries, public claims, or universal simulation.

Unit tests must enumerate every reachable row of this precedence table plus
every earlier failure override. They must prove that a later scientific label
cannot override preflight, reproduction, integrity, or incomplete execution,
and that no complete, integrity-valid four-arm pass/fail pattern falls through
without a registered label.

## Frozen feasibility and stop rules

The grouped implementation is required because a literal matrix exponential
per scheduled transition would make the two-replica control matrix
unnecessarily expensive. Before state-bearing execution, run source-only unit
and synthetic probes proving the grouped/literal equivalence described above.

Set a per-replica wall-clock limit of six hours and a whole-experiment limit of
14 hours. The runner may stream progress to stderr, but scientific payloads
contain no timing fields. Memory exhaustion, timeout, operator interruption,
nonfinite training state, or inability to finish both isolated replicas is
`e15_execution_incomplete`; it is not negative scientific evidence and may not
trigger a reduced arm matrix, fewer updates, smaller dataset, looser gate, or
automatic retry. A future retry requires a new clean execution using the same
committed contract and runner.

## Frozen replication and durable evidence

Run two isolated replicas from freshly constructed models and datasets.
Require byte-identical raw replicate result files.

Publish under
`docs/research/artifacts/canonical_latent_e15_training_package/` with exactly
these three top-level files:

- `canonical_latent_e15_training_package_evidence_bundle.tar.gz`;
- `canonical_latent_e15_training_package_result.json`; and
- `canonical_latent_e15_training_package_manifest.json`.

The evidence bundle must contain exactly these three regular-file members in
this exact order and no directory entries:

1. `replicate_a/result.json`;
2. `replicate_b/result.json`;
3. `complete_result.json`.

Every raw-file label must hash the literal file bytes. Canonical payload hashes
must be separately named. The combined object with its replication field
removed must equal the replicate object.

Use the same deterministic archive policy as E14: member order frozen exactly
as listed above; POSIX path separators; regular files only; tar `mtime=0`,
`uid=0`, `gid=0`, empty `uname`/`gname`, mode `0o644`; gzip `mtime=0` and no
original filename. Archive members may not contain absolute paths, symlinks,
hard links, device nodes, FIFOs, or traversal components.

The compact result must bind:

- clean execution HEAD and every source hash;
- raw and canonical replicate/combined hashes;
- bundle hash, byte count, member order, member hashes, and normalized
  metadata;
- all dataset, schedule, occurrence-count, model, generator, optimizer,
  coverage, gate, trace, and classification records; and
- zero held-out reads, provider calls, routing paths, label inputs, source
  bypasses, and encoder updates.

The detached manifest must bind the bundle's raw hash and byte count and the
compact result's raw hash and byte count. The compact result must not contain
its own raw hash or byte count. The detached manifest must not contain its own
raw hash or byte count. Any external publication receipt may report the
manifest's raw hash and byte count, but neither may be inserted into the
manifest bytes.

Build all output bytes in memory, stage in a sibling directory under an
exclusive publication lock, reopen and independently verify every declaration,
then atomically rename. Any failure must leave no durable partial evidence.
The stage name, lock metadata, and replica directory names are publication
metadata only and must not enter the canonical scientific payload.

Independent pre-state review is required before commit and execution.
Independent post-result review is required before recording a scientific
conclusion.
