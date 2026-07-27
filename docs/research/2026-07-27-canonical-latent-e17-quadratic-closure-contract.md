# Canonical Latent E17 Quadratic Closure Contract

Date: 2026-07-27

## Decision question

Does the frozen E10 52-coefficient physical function space admit a stable,
representation-preserving nonlinear latent operator for a smooth 2-D periodic
viscous Burgers family, or does unresolved high-frequency state make those
coefficients materially non-Markov?

E10 established one semantic coefficient space across grids, warped meshes,
uniform particles, and warped particles. E12-E16 then established that a
structured linear generator is expressive, identifiable, and practically
recoverable in that space. E17 is the first nonlinear closure gate. It tests
the representation before adding a new encoder, task router, expert, Koopman
lift, neural ODE, or neural operator.

No E17 sampled state may be constructed until this contract, its runner, and
its tests are committed at a clean HEAD and an independent pre-state review
returns GO.

## Research basis and selected mechanism

The candidate follows quadratic operator inference: infer a reduced polynomial
vector field directly in the fixed coordinates rather than relearning the
observation representation
(<https://doi.org/10.1016/j.cma.2016.03.025>). The convection term is constrained
to preserve quadratic energy, following current structure-preserving operator
inference work
(<https://arxiv.org/abs/2401.02889>,
<https://arxiv.org/abs/2503.10824>), while stability diagnostics follow the
stable-quadratic-OpInf direction
(<https://arxiv.org/abs/2403.00646>).

The truth solver uses a Fourier pseudospectral discretization with explicit
two-thirds de-aliasing, rather than accepting aliased low-mode products
(<https://doi.org/10.1002/sapm1972513253>). Koopman lifting, neural ODEs, and
Fourier neural operators are challengers for later gates, not E17, because
they change the representation or capacity hypothesis before fixed-latent
quadratic closure has been tested
(<https://arxiv.org/abs/1510.03007>,
<https://arxiv.org/abs/1806.07366>,
<https://arxiv.org/abs/2010.08895>).

## Frozen representation and equation

The representation remains exactly:

- the ordered 52-dimensional E10 basis;
- active periodic coefficients `0:49`, with the constant at index `0`;
- inactive trend coefficients `49:52`, fixed to zero and copied unchanged;
- Fourier cutoff three on the unit torus;
- the E7/E10 basis semantics and decoder; and
- no representation, family, task, geometry, resolution, or route label as an
  operator input.

Truth fields and truth derivatives are encoded by orthogonal Fourier
projection onto the 49 periodic columns only, followed by three literal zeros.
For a primary-grid field, this is exact extraction of modes
`abs(k_x),abs(k_y)<=3` mapped into the real E7 ordering. Define
`P_49` as that operator and `u_tail = u - decode(P_49 u)`. The joint 52-column
E10 weighted least-squares projection is reserved for the registered
cross-observation ingestion control, whose source initial field is
cutoff-three. It is not used to define truth targets or high-tail closure
pairs.

The registered smooth nonlinear family is

`u_t = -v_x u_x - v_y u_y - gamma_x u u_x - gamma_y u u_y + nu Laplacian(u)`

on `[0,1)^2` with periodic boundaries. The constant mode must remain invariant.
The inviscid nonlinear term must contribute zero continuous quadratic-energy
rate. Viscosity must contribute a nonpositive energy rate.

E17 does not claim shocks, discontinuous solutions, boundaries, irregular
domains, particle dynamics, vector fluids, or arbitrary multiphysics.

## Frozen linear trunk

The primary candidate and linear-only negative control use the sealed E15
`schedule_weighted_componentwise_lbfgs_neutral` generator:

`L(v_x,v_y,nu) = v_x A_x + v_y A_y + nu D`.

The E15 package is reconstructed deterministically from its sealed data and
literal schedule only because its numeric matrices were not published as
standalone state. Before any E17 fit, the runner must reproduce the sealed
model SHA-256
`c11e1b311a6bbb12332732009e68b6efca5768943c23ca25b25cd4f28526e423`,
generator hashes `A_x=6d5be486068e5829d90dbac40a855a5242d5ebaca93aa744fb1cc1831b355cdd`,
`A_y=947f713391f7f1a308e664094379dbc336034430c385406f9e423d9f5d485d55`,
and `D=57ea9e36fa5ae2746c848061c26748bb220cf2bb3d081e788315bd8c34c1b27f`,
plus all eight E15 recovery bits. A mismatch is a preflight failure. E17 may
not update `A_x`, `A_y`, or `D`.

An exact nonlearned Fourier linear generator is used only in the Galerkin
oracle ceiling. It is not substituted into the learned candidate.

## Truth solver and independent convergence gate

The primary truth is deterministic float64 CPU Fourier pseudospectral
integration on a `144 x 144` grid:

- centered unit-torus nodes;
- `torch.fft.fftfreq(N, d=1/N)` frequency ordering and PyTorch's default
  forward/inverse FFT normalization;
- Fourier derivatives and Laplacian using signed integer wavenumbers;
- a rectangular strict two-thirds mask retaining
  `abs(k_x) < N/3 and abs(k_y) < N/3`: exactly `-47:47` on each primary axis
  and `-71:71` on each reference axis;
- the mask applied to the state before differentiation/product formation and
  to the transformed nonlinear product before the vector field is returned;
- classical RK4 with internal step `0.001`;
- observation interval `0.01`;
- 16 observed transitions, ending at `t=0.16`;
- one intra-op and one inter-op thread; and
- no adaptive time step or result-dependent retry.

Let `e_i` denote coefficient index `i` in the ordered 49-periodic-mode basis.
Before constructing the training or validation population, run these six
literal analytic calibration cases:

| Case | Nonzero coefficients | `(v_x,v_y,nu,gamma_x,gamma_y)` |
| --- | --- | --- |
| single x | `c_14=0.4` | `(0.20,0,0.008,0,0)` |
| single y | `c_2=0.4` | `(0,-0.20,0.008,0,0)` |
| two-mode x | `c_7=0.35,c_14=-0.25` | `(0.10,0,0.006,0.80,0)` |
| two-mode y | `c_1=0.35,c_2=-0.25` | `(0,-0.10,0.006,0,-0.80)` |
| mixed | `c_1=0.20,c_7=-0.25,c_8=0.30,c_16=-0.20` | `(0.15,-0.10,0.006,0.75,-0.65)` |
| stress | `c_0=0.05,c_1=-0.30,c_2=0.20,c_7=0.35,c_8=0.25,c_14=0.25,c_16=-0.15` | `(0.30,-0.30,0.004,1.20,-1.20)` |

Compare the primary solver against a `216 x 216`, RK4 step `0.0005` reference
at every observation time. Compare active coefficients by exact Fourier-mode
extraction in the E7 ordering and compare decoded fields on one fixed centered
`288 x 288` grid. All must pass:

- decoded trajectory NRMSE `<= 2e-4`;
- active-coefficient trajectory NRMSE `<= 2e-4`;
- maximum absolute constant-mode drift `<= 1e-11`;
- relative energy-trajectory mismatch `<= 5e-4`;
- nonlinear energy-rate residual divided by total field energy `<= 1e-10`;
- primary and reference outputs finite; and
- the de-alias mask and retained Fourier-index sets equal their registered
  literal values.

Failure stops before population construction or fitting.

## Frozen populations

All random streams use independent `torch.Generator` instances. Canonical
tensor hashing uses C-contiguous little-endian float64 bytes; schedules and
indices use C-contiguous little-endian int64 bytes.

### Training

- state seed `817001`;
- parameter seed `817002`;
- schedule seed `817003`;
- `192` unique trajectories, `48` in each registered regime;
- all 17 observed states and exact projected derivatives are training records;
- Fourier initial conditions are cutoff-three only;
- initial mean in `[-0.10, 0.10]`;
- nonconstant raw coefficients are iid standard normal, scaled by
  `(1 + k_x^2 + k_y^2)^-1`, then normalized to a nonconstant field RMS in
  `[0.20, 0.50]`;
- `v_x,v_y` lie in `[-0.30,0.30]`;
- `nu` lies in `[0.004,0.012]`; and
- active nonlinear magnitudes lie in `[0.40,1.00]`.

For every scalar interval `[a,b]` and each 48-record regime, draw exactly one
float64 uniform value from each half-open stratum
`[a+j(b-a)/48, a+(j+1)(b-a)/48)` for `j=0..47`, using `b` only as the closed
upper endpoint of the last stratum. Independently permute each 48-value vector
with the same parameter generator before assignment. Draw vectors in this
literal order: mean, target RMS, `v_x`, `v_y`, `nu`, `abs(gamma_x)`,
`abs(gamma_y)`. Draw the `48 x 48` raw nonconstant-normal matrix for each
regime from the state generator before moving to the next regime.

The four equal regimes are:

1. `x_only`: `gamma_y=0`, signed `gamma_x`;
2. `y_only`: `gamma_x=0`, signed `gamma_y`;
3. `mixed_same_sign`: both nonzero with the same sign; and
4. `mixed_opposite_sign`: both nonzero with opposite signs.

For `x_only` and `y_only`, assign 24 positive then 24 negative active signs.
For `mixed_same_sign`, assign 24 `(+,+)` then 24 `(-,-)`. For
`mixed_opposite_sign`, assign 24 `(+,-)` then 24 `(-,+)`. The inactive
nonlinearity is literal zero. After all four regimes are built in registered
order, apply one `torch.randperm(192)` from schedule generator `817003` to the
trajectory record order. This permutation does not change the 17-state
chronology inside any trajectory.

### Validation

- state seed `917001`;
- parameter seed `917002`;
- schedule seed `917003`;
- `64` unique trajectories, `16` per regime;
- no identity, initial coefficient, parameter tuple, or schedule overlaps
  training; and
- the same four regimes and parameter ranges.

Validation is organized as `32` registered closure pairs, eight per regime.
Both members of a pair have the same cutoff-three coefficients and parameter
tuple but distinct high-frequency tails. The pair's low field is generated by
the training rule, except that exactly two pairs per regime form the
finite-amplitude stress stratum with low-field RMS in `[0.50,0.65]` and active
nonlinear magnitudes in `[1.00,1.20]`.

Each tail:

- uses only Fourier modes satisfying
  `4 <= max(abs(k_x),abs(k_y)) <= 6`;
- is real-valued by exact conjugate symmetry and has zero mean;
- is drawn from a pair-member-specific generator derived literally as
  `917100 + 2 * pair_index + member_index`;
- has coefficients scaled by `(1 + k_x^2 + k_y^2)^-1`;
- is normalized to tail RMS `0.04`; and
- has cutoff-three projection norm `<=1e-12`.

Construct a tail in complex128 Fourier storage on the 144 grid. The independent
half-plane is
`H={(k_x,k_y): k_x>0 or (k_x=0 and k_y>0)}` intersected with the registered
tail shell, ordered lexicographically by `(k_x,k_y)`. For each index in that
order, draw exactly two float64 iid standard-normal values from its literal
member generator, first the real part and then the imaginary part. Multiply
the complex value by `(1+k_x^2+k_y^2)^-1`, assign the conjugate value at
`(-k_x,-k_y)`, and leave every other Fourier entry zero. The shell contains no
self-conjugate zero or Nyquist entry. Apply default-normalized `ifft2`, take
the exactly real field, subtract its float64 mean, then multiply the complete
field once so `sqrt(mean(u_tail^2))=0.04`.

The two members' tails must be distinct in canonical bytes while their active
52-coefficient projections agree within `1e-12` and their three inactive
trends are zero.

For each regime, draw the eight low-state raw normal vectors and scalar
parameter vectors in the same literal order as training. For the first six
pairs, stratify mean over `[-0.10,0.10]`, low RMS over `[0.20,0.50]`, velocities
over `[-0.30,0.30]`, viscosity over `[0.004,0.012]`, and active nonlinear
magnitudes over `[0.40,1.00]`, using six equal strata. For the final two stress
pairs, use two equal strata over low RMS `[0.50,0.65]` and active nonlinear
magnitudes `[1.00,1.20]`; all other intervals are unchanged and use two equal
strata. Use the same sign pattern as training, truncated to equal positive and
negative counts within the ordinary and stress subsets. After construction,
apply one `torch.randperm(32)` from schedule generator `917003` to pair order;
members remain adjacent in literal member order `0,1`.

Validation is read only after the candidate is final. It may not select a
support, constraint, regularizer, threshold, solver step, or checkpoint.

The scientific budget is `192` training trajectories, `64` validation
trajectories, and zero held-out trajectories.

## Representation-closure diagnostic

For each of the 32 validation closure pairs, evaluate the exact PDE vector
field at the two registered initial full states. The two states have the same
52-dimensional latent but distinct admissible high-frequency tails. Project
each derivative to the 49 active coefficients and define the pairwise
closure-forcing difference.

Record per-case closure-forcing NRMSE relative to the full projected derivative
and the aggregate RMS ratio. The Markov/closure gate requires:

- aggregate RMS ratio `<= 0.10`;
- 95th percentile per-case ratio `<= 0.25`;
- maximum absolute constant-mode difference `<= 1e-11`; and
- all records finite.

For pair `p`, let `f_p0,f_p1 in R^49` be the projected initial derivatives and
define

`r_p = sqrt(sum((f_p0-f_p1)^2) /
max(0.5*(sum(f_p0^2)+sum(f_p1^2)),1e-24))`.

The aggregate RMS ratio is

`sqrt(sum_p sum((f_p0-f_p1)^2) /
max(0.5*sum_p(sum(f_p0^2)+sum(f_p1^2)),1e-24))`.

Sort the 32 finite `r_p` values ascending and define the 95th percentile by the
nearest-rank rule as element `ceil(0.95*32)-1 = 30` in zero-based indexing.

Also record the divergence between the two projected truth trajectories at
horizons `{1,4,8,16}`. The candidate receives the same projected initial state
for both members and therefore produces one identical latent trajectory.

This is a representation gate, not a candidate-training metric. If its
derivative ambiguity fails, the correct conclusion is that the tested 49
active coordinates are insufficiently Markov for this registered full-state
family. Do not add a router or larger operator under the same contract.

## Registered quadratic hypothesis

For active coefficients `c in R^49`, infer

`dc/dt = L(v_x,v_y,nu)c + gamma_x Q_x(c,c) + gamma_y Q_y(c,c)`.

The candidate is fixed as follows:

- `Q_x` and `Q_y` are homogeneous quadratic maps;
- the constant output is identically zero;
- inactive trends never enter or leave either map;
- each axis uses only Galerkin-compatible triads whose analytic triple-product
  magnitude exceeds `1e-12`;
- the support is computed by exact Fourier identities and must contain exactly
  `1,329` symmetric monomial/output entries per axis, `2,658` total;
- the support exposes no oracle coefficient values to fitting;
- linear equality constraints require
  `c^T Q_x(c,c)=0` and `c^T Q_y(c,c)=0` coefficientwise as cubic
  polynomials;
- coefficients are fit once by deterministic equality-constrained least
  squares over all training derivative records;
- the residual target subtracts the frozen learned linear trunk;
- columns are normalized by their training RMS and zero-RMS columns are a
  preflight failure;
- solve the equality-nullspace problem by deterministic float64 SVD and
  minimum-norm least squares;
- SVD tolerance is `max(shape) * eps * sigma_max`;
- no ridge, sparsity penalty, early stopping, validation selection, optimizer,
  retry, or post-result support change is allowed; and
- record `rank(A)`, `cond_2(A)`, `rank(A^T A)`, and `cond_2(A^T A)`
  separately. Require full reduced-column rank, `cond_2(A) <= 1e8`, and
  `cond_2(A^T A) <= 1e16`.

The runner must verify the fitted constant-output coefficients are exactly
zero and each axis's cubic energy-constraint residual is `<= 1e-10`.

## Frozen controls

Evaluate exactly these arms:

1. `projected_truth`: the projected 144-grid truth trajectory;
2. `zero_tail_galerkin_control`: exact Fourier linear and quadratic coefficient
   actions integrated in the 49 active coordinates;
3. `frozen_linear_only`: the sealed learned E15 linear trunk with `Q=0`; and
4. `constrained_quadratic_candidate`: the same frozen learned trunk plus the
   single fitted constrained quadratic map.

There is no learned router, encoder update, dense black-box MLP, neural ODE,
Koopman lift, FNO, unconstrained post-hoc rescue, or second candidate.

## Evaluation

Use fixed-step coefficient-space RK4 with internal step `0.001` for all latent
arms. Evaluate all 64 validation trajectories without intermediate reads.

Record by arm, regime, stress stratum, and observation process:

- exact projected-derivative NRMSE;
- active-coefficient and decoded NRMSE at horizons `1`, `4`, `8`, and `16`;
- high-frequency decoded NRMSE for radius `>=3`;
- constant-mode absolute error;
- total energy and viscous energy-balance error;
- nonlinear energy-rate residual;
- finite-amplitude maximum norm and blow-up indicator;
- combined-step versus two-half-step semigroup defect at horizons `1`, `4`,
  and `8`; and
- cross-observation coefficient and decoded mismatch when the same initial
  field is ingested through the registered E10 grid, warped-mesh,
  uniform-particle, and warped-particle observation processes.

Cross-observation uses the frozen E10 geometry seeds and masses. It changes
only initial observation/projection; every arm then evolves the same 52
coefficients with no geometry label.

### Metric definitions

For arrays with the registered aggregation axes, define pooled

`NRMSE(pred,target) = sqrt(sum((pred-target)^2) / max(sum(target^2),1e-24))`.

Aggregate derivative NRMSE pools all validation trajectories, active outputs,
and the initial derivative only. Horizon metrics pool trajectories, spatial
points or active coefficients, and the named horizon only. Regime and stress
metrics use the same equation on their literal subsets; no macro average can
replace a pooled gate. Decode every coefficient state on the fixed centered
`288 x 288` grid.

For decoded spectra, use the default-normalized two-dimensional FFT on that
grid and define high frequency by integer radial wavenumber
`sqrt(k_x^2+k_y^2) >= 3`; spectral NRMSE uses complex squared magnitude in the
same pooled equation. The constant-mode error is the maximum absolute
coefficient-0 difference from the arm's own initial value.

Define field energy `E(u)=0.5*mean(u^2)` and dissipation
`R(u)=nu*mean(u_x^2+u_y^2)` on the 288 grid. Over each model interval,
accumulate `R` on every internal RK4 state with the classical RK4 quadrature
weights. The relative energy-balance defect at time `t` is
`abs(E(t)-E(0)+integral_0^t R ds) / max(E(0),1e-24)`. The nonlinear
energy-rate residual is
`abs(c^T(gamma_x Q_x(c,c)+gamma_y Q_y(c,c))) / max(c^T c,1e-24)`.

At horizon `h` observation intervals, the semigroup comparison is one call
integrating `10*h` RK4 steps of size `0.001` versus two consecutive calls of
`5*h` steps each from the same initial coefficient. Its defect is pooled
coefficient NRMSE between those results.

For every registered E10 observation realization, compare its candidate
trajectory to the candidate trajectory initialized from the exact periodic
coefficient vector. Coefficient mismatch uses pooled NRMSE at each registered
horizon. Decoded mismatch uses the same equation on the 288 grid. The
cross-observation gates use the maximum over observation family, realization,
and horizon `{1,4,8,16}`.

## Literal gates

### Zero-tail Galerkin diagnostic

Record the zero-tail Galerkin control under all learned-candidate metrics. It
is a mechanistic control, not a qualification gate or representation ceiling:
a learned deterministic closure can legitimately compensate predictable
resolved effects of truncated tails even when the zero-tail Galerkin model
cannot.

### Learned candidate

The constrained quadratic candidate must pass every gate:

- validation derivative NRMSE `<= 0.08`;
- decoded rollout NRMSE at horizons `4/8/16` `<= 0.05/0.12/0.22`;
- horizon-16 high-frequency NRMSE `<= 0.30`;
- horizon-16 decoded NRMSE `<= 0.60x` frozen-linear-only;
- no regime horizon-16 decoded NRMSE exceeds `0.30`;
- stress-stratum horizon-16 decoded NRMSE `<= 0.30`;
- maximum constant-mode error `<= 1e-10`;
- maximum nonlinear energy-rate residual divided by energy `<= 1e-9`;
- maximum relative energy-balance error `<= 0.08`;
- semigroup defect at every registered horizon `<= 2e-5`;
- all stress rollouts finite with maximum absolute field value `<= 2.0`;
- worst cross-observation coefficient mismatch `<= 0.10`;
- worst cross-observation decoded mismatch `<= 0.10`; and
- every output and metric finite.

No gate may be relaxed, averaged away across regimes, or replaced after any
validation read.

## Classification and precedence

Apply the first matching classification:

1. `preflight_failed`: source, predecessor, convergence, serialization,
   support, excitation, rank, condition, constraint, or boundary preflight
   fails;
2. `incomplete`: a required record, arm, regime, horizon, observation process,
   replica, or finiteness check is absent;
3. `latent_closure_insufficient`: the registered same-latent/different-tail
   derivative-ambiguity gate fails;
4. `quadratic_identification_failed`: representation gates pass but the learned
   candidate fails any literal candidate gate;
5. `constrained_quadratic_closure_qualified`: all preflight, representation,
   candidate, coverage, finiteness, and boundary gates pass.

Tests must exhaustively enumerate classification precedence, including every
single-gate failure and all missing-record overrides.

Only classification 5 authorizes a next contract that broadens nonlinear
families or tests particle-resolved dynamics. Classification 3 routes back to
representation work: inspect cutoff, memory/closure variables, or a learned
observation encoder without adding an under-the-hood task router.

## Provenance and evidence seal

Before state construction, verify these predecessor bytes:

| Input | SHA-256 |
| --- | --- |
| E15 evidence bundle | `3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a` |
| E15 compact result | `e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d` |
| E15 detached manifest | `1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311` |
| E16 evidence bundle | `71fc490c2bc361fbf0b26d5bfcccfc460bcf5af223b5000d1e6043672504a586` |
| E16 compact result | `6716273a3ea980f7d24462ec3e40eb37091d229d524aec5f9a0ad89bbb9d325a` |
| E16 detached manifest | `6af927037eebeebc3a9a95842d549c633279391d28715779b7fc04c05b59720f` |
| E15 runner | `943558c42d2e8a13879fc3fe6f1301142efe7c7949f51e7e4ff509a6af6ae9ca` |
| E12 runner | `8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a` |
| E10 runner | `a06486e5f6e77667fa06c65ee5dbff8c57cad6b505789b94d4596cc31515e404` |
| E7 runner | `cf81597b3909e9693508b62e595eb006a8598d186de062eaf4a8f241d4b07488` |

At launch, bind the committed E17 contract, runner, tests, all imported source
files, configuration, Python/Torch versions, environment, Git HEAD, worktree
cleanliness, and every predecessor/artifact hash.

Run two independent whole-experiment replicas in fresh processes. Require
literal canonical JSON equality after removing only the registered replication
label. Publish atomically:

- `canonical_latent_e17_quadratic_closure_evidence_bundle.tar.gz`;
- `canonical_latent_e17_quadratic_closure_result.json`; and
- `canonical_latent_e17_quadratic_closure_manifest.json`.

The deterministic archive must use zero gzip mtime, sorted fixed member order,
regular files only, normalized metadata, recorded member bytes and SHA-256
hashes, and independent reopen/recomputation before publication.

## Boundary

E17 uses synthetic training and validation only. It reads zero held-out
trajectories and makes zero provider calls. It performs no encoder update,
routing decision, task-label input, geometry-label input, source bypass, or
public/deployment claim.

The result may qualify or reject only the registered smooth periodic scalar
nonlinear closure hypothesis in the frozen E10 coordinates. Negative evidence
must be scoped to the failing layer identified by classification precedence.

## Pre-state truth-resolution erratum

The first source-only analytic calibration, after independent contract GO and
before any E17 training or validation construction, used the originally
registered `96/144/192` primary/reference/comparison grids. Five of six cases
passed. The stress case failed only the full-field convergence gate:

- full-field trajectory NRMSE `0.0019192295319917713` versus `<=2e-4`;
- active-coefficient trajectory NRMSE `3.16650803627406e-7`;
- relative energy-trajectory mismatch `5.734026434956068e-6`;
- maximum constant-mode drift `3.157196726277789e-15`; and
- nonlinear energy-rate residual `4.666255446007984e-16`.

State reads were training `0`, validation `0`, and held-out `0`. This is
negative evidence about primary-grid resolution for the registered smooth
stress trajectory, not evidence about the latent or operator. No threshold,
state distribution, equation, horizon, time step, mask rule, or classification
changed. The only correction is the preregistered one-level spatial refinement
to `144/216/288`, with strict retained sets `-47:47` and `-71:71`. The
corrected calibration must receive independent pre-run GO and pass once before
any population constructor is implemented or called.
