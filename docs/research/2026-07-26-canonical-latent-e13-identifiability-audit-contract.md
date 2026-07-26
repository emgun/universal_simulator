# Canonical Latent E13 Identifiability Audit Contract

Date: 2026-07-26
Status: frozen after independent review and before state-level measurement

Independent pre-measurement review found that the initial draft did not
byte-lock the E12 replay, fully specify the L-BFGS closures, retain every E12
generator-identification subgate, or enforce literal mode-record coverage. This
revision repairs those issues before any E13 sampled tensor was constructed.

## Decision context

E12 kept the exact E10 projection and learned additive continuous-time
generators in the shared 52-coefficient space. It improved the E11 dense
operator by more than two orders of magnitude and passed aggregate rollout,
transfer, temporal, semigroup, cross-observation, and physics gates. It did
not qualify because:

- elementary and fine-tuned maximum decoded basis-action NRMSE were
  `0.0999214578` and `0.1084054288`, above `0.05`;
- fine-tuned final composite high-frequency NRMSE was `0.2954287579`, above
  `0.15`;
- amplitude ratios remained near one and the exact generator oracle remained
  near float64 precision.

E12 therefore leaves a narrow causal question. The E10 representation and the
structured generator hypothesis are not refuted, but the frozen E12 protocol
does not reveal whether the residual modal error comes from:

1. AdamW and its scheduled stochastic updates;
2. the 2,304-parameter full-skew parameterization relative to the frozen
   excitation;
3. missing representation-aligned support or parameter sharing; or
4. genuinely insufficient sampled excitation.

E13 is a diagnostic audit, not a replacement E12 run and not a new
qualification claim.

## Question

Using exactly the accepted E12 elementary trajectories, can a deterministic
full-population optimizer recover the full skew generators? If not, can
successively stronger but still learned E10-basis structural controls recover
the modal law?

The answer selects the next operator hypothesis without changing the encoder,
teacher, state split, thresholds, or routing boundary.

## Research basis

Continuous-time generator identification from discrete observations is
subject to system aliasing: distinct generators can induce the same sampled
transition map. Sampling bounds depend on the imaginary generator spectrum,
and structural or sparsity knowledge can reduce ambiguity:
<https://arxiv.org/abs/2204.14021> and
<https://arxiv.org/abs/1605.06973>.

Recent Lie-generator work directly learns a structured matrix and
exponentiates it, and reports that an unconstrained generator can fit dynamics
while recovering physically wrong spectra whereas skew-plus-dissipative
structure materially improves spectral recovery:
<https://arxiv.org/abs/2603.27442>.

Learning theory for Fourier linear operators separates finite-sample,
truncation, and discretization errors rather than treating aggregate prediction
error as a single cause:
<https://arxiv.org/abs/2408.09004>.

Recent spectral-refinement work further motivates isolating optimization from
representation and approximation by optimizing a restricted spectral layer
instead of retuning an end-to-end operator:
<https://arxiv.org/abs/2405.17211>.

E13 applies these ideas conservatively: first test a different deterministic
optimizer in the same full hypothesis class, then restrict only the generator
support and sharing already implied by the frozen E10 coefficient semantics.

## Frozen provenance and historical evidence

The executable must bind and hash-verify:

- this contract;
- the E13 runner;
- the accepted compact E12 artifact at
  `docs/research/artifacts/canonical_latent_e12_structured_generator_result.json`;
- the accepted E12 replay lock at
  `docs/research/artifacts/canonical_latent_e12_replay_lock.json`;
- the E12 runner and its imported E11/E10 implementation dependencies;
- a clean committed Git HEAD containing byte-identical source files.

The accepted E12 compact artifact SHA-256 is
`d4760ec3d69b4397cc14ffc3bb08edd3f073edcaf1f6d4dd70db070a96cab3b2`.
Its complete result SHA-256 is
`7b214639d6287ede84352bbcfe7a31a1c9e891f4080fc9c623aee5df5e9c5ccc`.
The E12 replay lock SHA-256 is
`0bf8f032daf95415bee401b5a90f4e6ca1598f12748151dd7310b2a6d02f8dfd`.

Before constructing any sampled tensor, require every source, config,
dependency, artifact, lock-manifest, Git-presence, and clean-worktree check to
pass. Otherwise materialize only `e13_preflight_failed` with zero training,
validation, and held-out state reads and zero optimizer updates.

Before any E13 recovery control is run, reproduce the exact E12 elementary
AdamW checkpoint and its elementary training record from the same source,
data, and schedules. Require:

- byte equality to every reused dataset record in the E12 replay lock,
  including shape plus initial-coefficient, parameter, and complete-trajectory
  SHA-256 values;
- exact equality to all five E11 schedule shapes and hashes in the replay lock;
- exact E12 config SHA-256
  `cd428d490ad9d5505f88ead66b41fdb25e25830f45d0eb21f451c5dbea261934`;
- exact initial model SHA-256
  `64c87294711c68c9bc4a9f56cb3f8a8ca23b1e1eed84493bfc18e18f3a2c9218`;
- exact elementary checkpoint SHA-256
  `e9c17bc1871f5b2008d3899da9f59a44cf1209448f50ba576ca63f6281602e7b`;
- exact elementary generator SHA-256 values
  `17f70896f48f5651854746c5be9edc7dd42d2fd331c1fbc7d2c6e2edd1a46e53`,
  `726d18845ad3dc3e462fc16578adf08929594e57dd02818c1e13711eee31fda2`,
  and
  `1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370`
  for `A_x`, `A_y`, and `D`;
- exact elementary first/final losses
  `0.08166621133143402` and `0.00032425428259914163`, update count `1500`,
  and examples per regime `48000`;
- reproduced elementary maximum basis-action decoded NRMSE within `1e-12` of
  `0.09992145782297766`;
- reproduced elementary zero-shot composite decoded rollout NRMSE within
  `1e-12` of `0.00536402045663299`;
- all reproduced model and generator hashes recorded.

Failure to reproduce E12 stops the audit as
`e12_reproduction_failed`; no E13 causal classification is permitted.

## Frozen representation, teacher, data, and thresholds

Reuse E12 exactly:

- ordered 52-dimensional E7/E10 basis;
- first 49 periodic coefficients active, including the constant at index `0`;
- last three nonperiodic trend coefficients inactive and copied unchanged;
- exact float64 `64 x 64` Fourier advection-diffusion teacher;
- 256 trajectories in each elementary regime;
- eight transitions per trajectory;
- E12 state seeds, parameter seeds, coefficient scales, parameter ranges,
  timestep range, and exact generated tensor hashes;
- E12 validation state and parameter seeds;
- zero held-out reads and zero provider calls.

No additional trajectory, state seed, parameter seed, observation realization,
teacher query, or data augmentation is allowed. The elementary recovery loss
uses all frozen elementary transitions exactly once per closure evaluation;
there is no resampling schedule.

Keep these E12 gates unchanged:

- maximum decoded basis-action NRMSE `<=0.05`;
- final composite high-frequency decoded NRMSE `<=0.15`.

Also retain the E12 structure and boundary checks. No result may qualify
nonlinear dynamics, particles, unseen equations, broader PDE families, or
public claims.

## Frozen sampling and excitation preflight

Before sampled state access:

1. construct the support-sparse and mode-tied masks solely from the static E10
   semantic basis indices, hash them, and only then construct or expose any
   oracle tensor;
2. verify the accepted oracle and all E12 closure checks;
3. compute the largest possible observed oracle advection phase from the frozen
   bounds and the oracle spectrum;
4. require that oracle phase to be strictly less than `pi` and record it only
   as a true-generator sampling-margin diagnostic. It does not exclude aliases
   in the unbounded full-skew learned class;
5. verify every recovery parameterization exactly preserves the constant and
   inactive modes, skew advection, and nonpositive diagonal diffusion;
6. verify exact parameter counts and mask hashes.

After constructing the frozen elementary datasets but before optimization,
record, without changing any decision:

- the numerical rank and condition number of each nonconstant input covariance;
- per-oracle rotation-plane `2 x 2` input Gram rank and condition number;
- the rank and singular spectrum of the 12-parameter mode-tied prediction
  Jacobian evaluated at the accepted oracle rates.

Use float64 singular values and rank tolerance
`max(shape) * eps * largest_singular_value`. These are excitation diagnostics,
not tunable gates. A rank deficiency is interpreted literally and must not be
repaired inside E13.

## Frozen recovery controls

All controls operate only on coefficients and physical parameters. None
receives family, geometry, representation, task, regime, source, or routing
labels.

Every exponential is the exact float64 `torch.matrix_exp`. Use one CPU thread
and deterministic PyTorch algorithms.

### Common deterministic-recovery executable

For controls 1-4, form one C-contiguous input tensor with shape
`[2048,52,1]`, one identically ordered target tensor of that shape, and one
C-contiguous parameter tensor with shape `[2048,4]` for each regime.
Flattening is trajectory-major and then transition-step-major, exactly as the
accepted `TrajectorySet.transitions` property. No shuffle or copy with a
different logical order is allowed.

Solve components in the fixed order `A_x`, `A_y`, then `D`. For each component:

1. create a new L-BFGS optimizer over only that component's parameter tensor;
2. call `optimizer.step(closure)` exactly once;
3. on every closure call, execute
   `optimizer.zero_grad(set_to_none=True)`, one combined-rule forward pass on
   the complete corresponding regime population, the accepted
   `normalized_loss(prediction, target)` mean reduction, `loss.backward()`,
   and return the scalar loss;
4. compute the gradient norm after backward as the float64 Euclidean norm of
   the flattened optimized parameter gradient;
5. append closure index, loss, and gradient norm in call order.

Recreate the optimizer between components; optimizer state never crosses a
component boundary. Leave each solved component in the shared control model,
assemble the three matrices through that control's frozen parameterization,
and evaluate only after all three solves finish. Because every elementary
regime zeros the other two physical coefficients, previously solved
components do not contribute to a later component's loss.

Record the optimizer's final `n_iter` and `func_evals`, total closure count,
first/final closure loss, and first/final gradient norm. Do not infer a
semantic convergence reason that PyTorch does not expose.

### Control 0: exact E12 AdamW replay

Reproduce `structured_elementary_pretrained` exactly:

- full 2,304-parameter E12 generator;
- zero/skew and unit-diffusion initialization;
- accepted E12 balanced schedules;
- 1,500 AdamW updates;
- learning rate `2e-2`;
- weight decay zero.

This control may only establish historical reproduction.

### Control 1: full-skew L-BFGS from neutral initialization

Use the identical 2,304-parameter E12 generator initialized exactly as E12.
Optimize `A_x`, `A_y`, and `D` independently on their corresponding complete
elementary trajectory population. Each closure uses all eight transitions of
all 256 trajectories in that regime.

For each independent solve use PyTorch L-BFGS with:

- learning rate `1.0`;
- `max_iter=250`;
- `max_eval=300`;
- history size `100`;
- `tolerance_grad=1e-12`;
- `tolerance_change=1e-15`;
- line search `strong_wolfe`;
- no weight decay, clipping, retries, early selection, or validation access.

The loss is the accepted E12 modal-scale-normalized one-step coefficient MSE.
The common deterministic-recovery executable fixes closure semantics,
component order, optimizer construction, and recording.

### Control 2: full-skew L-BFGS polish from E12

Use the exact same independent L-BFGS solves and settings as control 1, but
initialize from the reproduced E12 elementary checkpoint. This isolates
whether E12 reached a locally repairable point in the same hypothesis class.

### Control 3: support-sparse L-BFGS

Use the ordered E10 tensor-product sine/cosine semantics to freeze only the
nonzero support:

- each x sine/cosine pair may rotate independently for each of the seven
  y-components;
- each y sine/cosine pair may rotate independently for each of the seven
  x-components;
- every nonconstant diagonal diffusion rate remains independently learned;
- all other advection entries are exactly zero.

Learn 42 advection plane rates and 48 diffusion rates, for exactly 90
parameters. No oracle numerical entry or target rate initializes the model.
Parameterize each diffusion rate as `exp(log_rate)`, initialize advection rates
to zero and diffusion log-rates to zero, then use the same independent L-BFGS
settings and complete populations as controls 1-2.

This is an oracle-support diagnostic ceiling only in the sense that the
support is supplied by the already frozen E10 basis semantics. The numerical
law remains learned exclusively from E12 transitions. Constructing or changing
the support mask by inspecting oracle matrix values is prohibited.

### Control 4: mode-tied L-BFGS

Use the same E10 support, with additional parameter sharing implied by the
tensor-product basis:

- one learned x-rotation rate for each of x frequencies 1, 2, and 3, shared
  across all seven y-components;
- one learned y-rotation rate for each of y frequencies 1, 2, and 3, shared
  across all seven x-components;
- one learned nonnegative x-diffusion contribution for each nonzero x
  frequency and one learned nonnegative y contribution for each nonzero y
  frequency;
- each diagonal diffusion rate is the negative sum of its x and y
  contributions.

Learn exactly 12 parameters. Initialize advection rates to zero and all six
diffusion log-contributions to zero; each nonnegative contribution is
`exp(log_rate)`. Use the same independent complete-population L-BFGS settings.

The basis ordering and sharing are fixed; the analytic values `2*pi*k` and
`(2*pi*k)^2` are never supplied to the learned control.

## Frozen evaluation

Evaluate these five controls plus the exact oracle:

1. E12 AdamW replay;
2. full-skew L-BFGS from neutral initialization;
3. full-skew L-BFGS polish from E12;
4. support-sparse L-BFGS;
5. mode-tied L-BFGS;
6. exact oracle.

For every control record:

- parameter count and all structural residuals;
- model and component-matrix SHA-256 values;
- generator Frobenius, supported-entry, leakage, diffusion-rate, and
  commutator diagnostics against the oracle;
- elementary validation one-step and eight-step coefficient and decoded
  NRMSE for x-advection, y-advection, and diffusion;
- zero-shot composite validation one-step and eight-step coefficient and
  decoded NRMSE;
- final zero-shot composite high-frequency NRMSE and amplitude ratio;
- maximum decoded basis-action NRMSE over every E12 closure parameter case;
- finite status.

### Mode-resolved evidence

For every control, every 49 active basis inputs, every E12 closure parameter
case, and one- and eight-step horizons, persist:

- basis index;
- semantic x and y component labels from
  `{constant, sin1, cos1, sin2, cos2, sin3, cos3}`;
- parameter-case name and values;
- horizon;
- decoded NRMSE;
- coefficient NRMSE;
- coefficient-space angular error in radians;
- prediction-to-target amplitude ratio;
- off-target-direction residual.

Persist the literal argmax record for every metric and control. An aggregate
maximum without its identity is invalid.

The executable must reject result materialization unless it retains these
literal complete schemas:

- `generator_identification[6 controls] = 6` records;
- `validation[6 controls][4 regimes] = 24` records, where every record contains
  both one- and eight-step metrics;
- `mode_resolved[6 controls][49 bases][18 cases][2 horizons] = 10,584`
  records;
- one unique mode-resolved key
  `(control,basis_index,case_name,horizon)` for every required record;
- `mode_argmax[6 controls][5 metrics] = 30` records for decoded NRMSE,
  coefficient NRMSE, coefficient angle, absolute amplitude-ratio error, and
  off-target-direction residual;
- `recovery_training[5 learned controls] = 5` records with the exact
  component and closure histories required above.

No optional, `applicable`, averaged-away, or missing cell is permitted.

## Frozen classification

Define a learned control as `recovery_pass` only when it is finite, passes all
frozen structural checks, and satisfies every unchanged applicable E12
generator-identification subgate:

- maximum generator relative Frobenius error `<=0.10`;
- maximum supported advection-entry relative error `<=0.20`;
- maximum off-support leakage `<=0.10`;
- maximum diffusion-rate relative error `<=0.20`;
- maximum normalized commutator `<=0.02`;
- maximum decoded basis-action NRMSE `<=0.05`;
- zero-shot composite final high-frequency NRMSE `<=0.15`.

The basis-action value used for the gate is exactly E12's one-step metric at
`dt=0.04`; the eight-step mode records are required evidence but not a
substitute threshold.

To prevent a generator-recovery label from hiding predictive regression, also
require:

- maximum elementary validation one-step decoded NRMSE across x-advection,
  y-advection, and diffusion `<=0.03`;
- maximum elementary validation eight-step decoded NRMSE `<=0.08`;
- zero-shot composite eight-step decoded NRMSE `<=0.20`;
- zero-shot composite eight-step decoded NRMSE divided by the unchanged E12
  persistence comparator `1.5584481380508215` `<=0.75`.

Classify in this order:

1. `e13_preflight_failed` if any source, dependency, config, Git, artifact,
   mask, structure, oracle, or coverage preflight fails. If failure occurs
   before sampled tensors, state reads and optimizer updates must be zero.
2. `e12_reproduction_failed` if control 0 misses any locked reproduction
   requirement.
3. `full_parameterization_deterministic_recovery_succeeds` if either full-skew
   L-BFGS control is `recovery_pass`.
4. `support_restriction_required_under_frozen_solvers` if support-sparse is
   `recovery_pass` and both full-skew controls fail.
5. `mode_tying_required_under_frozen_solvers` if mode-tied is
   `recovery_pass` and support-sparse plus both full-skew controls fail.
6. `elementary_excitation_rank_deficient` if mode-tied fails and its frozen
   Jacobian or any required rotation-plane Gram is rank deficient.
7. `recovery_controls_not_qualified` otherwise.

The classification is causal only within the exact E12 synthetic linear
problem, E10 representation, frozen state population, and specified solver,
batching, and blockwise-decomposition packages. It does not attribute success
to L-BFGS alone.

## Repetition and evidence integrity

Run the complete audit twice from scratch in separate directories. Require
byte-identical complete result payloads after removing only absolute output
paths. Persist:

- raw replicate files and hashes;
- a complete combined result and hash;
- a compact reviewed artifact and detached hash manifest;
- exact Git HEAD, config, contract, runner, dependency, data, schedule, model,
  generator, and result hashes;
- independent contract and result review findings.

Any source edit, threshold edit, solver retry, seed change, extra state access,
or control added after the first sampled-state read invalidates the run and
requires a new E-number.

## Prohibited interpretations and changes

E13 must not:

- alter or retrain the E10 projection;
- add routing, experts, family labels, task labels, or source labels;
- add basis modes or nonlinear dynamics;
- use oracle numerical generator entries to initialize a learned control;
- use validation data for optimization or selection;
- relax E12 gates;
- claim that basis-semantic tying is itself a universal solution;
- treat failure of these controls as failure of shared latent physics in
  general.

## Next-action map

- `full_parameterization_deterministic_recovery_succeeds`: preregister an E14
  deterministic recovery or matched optimization-package challenger in the
  full structured generator, preserving the E10 basis.
- `support_restriction_required_under_frozen_solvers`: preregister an E14
  support-discovery or sparse-generator architecture that learns support
  without oracle access.
- `mode_tying_required_under_frozen_solvers`: preregister an E14
  representation-aligned tied generator and test whether the tying transfers
  beyond this analytic Fourier family.
- `elementary_excitation_rank_deficient`: strengthen elementary excitation in
  a new contract before changing the operator family.
- `recovery_controls_not_qualified`: audit loss geometry and sampled transition
  information before expanding the encoder or introducing specialists.

Routing remains closed in every branch.
