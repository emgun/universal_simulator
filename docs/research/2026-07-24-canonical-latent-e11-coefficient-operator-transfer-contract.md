# Canonical Latent E11 Coefficient-Operator Transfer Contract

Date: 2026-07-24
Status: frozen before state-level measurement

## Question

After E10 qualified one exact representation-blind projection, can one
coefficient-to-coefficient neural operator learn elementary periodic
advection and diffusion and then adapt to their unseen composition from only
eight target trajectories more effectively than the same model trained from
scratch?

E11 tests dynamics in the qualified coefficient space. It does not reopen the
encoder and does not introduce representation or task routing.

## Research basis

Fixed-Basis Coefficient-to-Coefficient operator learning explicitly separates
function projection from the learned coefficient map and warns that total
error is bounded by the output basis approximation:
<https://arxiv.org/abs/2510.10350>.

Recent multiphysics work reports that training on elementary forms of a target
PDE can improve data efficiency, long-term consistency, and out-of-distribution
generalization:
<https://openreview.net/forum?id=mJiPqOzc3O>.

Recent neural-operator splitting work treats unseen dynamics as compositions
of elementary learned operators:
<https://openreview.net/forum?id=vZOFjAvekt>.

E11 tests the simpler dense shared-map hypothesis first. If its transfer gate
fails while the coefficient closure and full-data controls pass, the next
identified challenger is an explicit additive-generator or Strang-splitting
coefficient operator. A hidden family router is not an allowed repair.

## Frozen function and PDE space

Retain the ordered E7/E10 52-dimensional basis and exact float64 quadrature
projection. E11 initial conditions use the first 49 periodic tensor-Fourier
modes. The final three nonperiodic trend coefficients are fixed to zero.

Draw each active coefficient independently from a zero-mean Gaussian with
standard deviation

`0.7 / (1 + k_x^2 + k_y^2)^1.5`,

except the constant mode, whose standard deviation is `0.20`. This creates a
smooth, full-rank periodic coefficient distribution while staying strictly
inside the frozen basis.

The physical teacher solves

`u_t + v_x u_x + v_y u_y = nu (u_xx + u_yy)`

on the periodic unit square. It decodes the frozen basis on a `64 x 64` grid,
applies the exact Fourier semigroup in float64, and projects the result back
with the same E7 basis. No numerical time integrator or learned model defines
the target.

## Frozen regimes and splits

All state and parameter identities are disjoint across splits.

Elementary pretraining uses 256 trajectories per regime:

- `x_advection`: `v_x` has magnitude uniformly in `[0.20, 1.00]` with
  equiprobable sign; `v_y=0`, `nu=0`;
- `y_advection`: the symmetric construction with `v_x=0`;
- `diffusion`: `v_x=v_y=0`, `nu` uniformly in `[0.01, 0.08]`.

Use initial-state seed `51001` and parameter seed `51101`. Each trajectory has
eight transitions. Its fixed `dt` is uniform in `[0.02, 0.06]`.

The composite target has all of `v_x`, `v_y`, and `nu` nonzero with the same
marginals:

- eight few-shot trajectories at state seed `52001`, parameter seed `52101`;
- 256 full-data control trajectories at state seed `53001`, parameter seed
  `53101`;
- 64 validation trajectories at state seed `61001`, parameter seed `61101`.

Elementary-retention validation reuses those 64 initial states without reading
the composite targets. Generate fresh elementary parameters at seeds `61102`,
`61103`, and `61104` for x-advection, y-advection, and diffusion,
respectively, with the same marginals and `dt` interval.

The validation split is read only after the architecture, schedules, metrics,
and gates below are frozen. Training and validation data are synthetic.
Reserved held-out reads remain zero.

For temporal extrapolation, evaluate the same 64 validation initial states and
parameters at `dt=0.075`, which is outside the training interval.

## Frozen closure preflight

Before optimization, evaluate all 49 active periodic basis vectors at
`|v|={0.20,0.60,1.00}` with both signs, `nu={0.01,0.045,0.08}`, and the fixed
composite tuples `(-1.0,0.2,0.01)`, `(0.6,-0.6,0.045)`, and
`(1.0,1.0,0.08)`. The three inactive nonperiodic trend vectors are outside the
frozen E11 initial-condition distribution and are recorded but not evolved.
For one and eight steps require:

- projected coefficient finite;
- projection rank `52`;
- truth-to-projected decoded NRMSE `<=0.01`;
- projected semigroup composition error `<=1e-10`.

If any closure condition fails, classify
`coefficient_dynamics_not_qualified`, skip every optimizer and state split,
and repair the basis before operator work.

## Frozen model and controls

Use one residual coefficient MLP:

- input: 52 normalized coefficients plus continuous
  `[v_x, v_y, nu, dt]`;
- hidden width `96`;
- two GELU hidden layers;
- output: 52 normalized coefficient increments;
- output layer initialized to zero;
- no family, representation, resolution, task, source-index, or routing input;
- coefficient normalization uses the frozen modal standard deviations above;
- physical inputs are scaled by `[1, 1, 0.08, 0.075]`;
- learned parameter count must be identical for every arm.

Freeze one initial state dict at seed `71001`.

Arms:

1. `elementary_pretrained`: train one model jointly on balanced batches from
   the three elementary regimes for 1,500 AdamW updates, learning rate
   `2e-3`, weight decay `1e-6`, 32 transitions per regime per update.
2. `pretrained_fewshot`: reset optimizer state and fine-tune the elementary
   checkpoint on the eight composite trajectories for 400 updates, learning
   rate `1e-3`, batch size 64.
3. `scratch_fewshot`: start from the identical frozen initial state and consume
   the exact same 400 composite batches as arm 2.
4. `full_composite_control`: start from the identical frozen initial state and
   train on 256 composite trajectories for 1,500 updates, learning rate
   `2e-3`, batch size 96.

All sampling schedules are generated before optimization and hashed. No early
stopping, validation selection, retry, seed change, width change, or threshold
change is allowed.

Persistence and exact projected truth are nonlearned baselines.

## Frozen metrics

For every arm and regime record:

- one-step coefficient and canonical-grid decoded NRMSE;
- eight-step autoregressive coefficient and decoded NRMSE;
- high-frequency spectral NRMSE at the final step;
- error by rollout step and by physical parameter quartile;
- maximum finite error and effective coefficient rank;
- mean-mode error;
- advection relative `L2`-norm drift;
- diffusion fraction of rollout steps with nonincreasing predicted energy;
- semigroup consistency: one `dt=0.04` prediction versus two `dt=0.02`
  predictions;
- `dt=0.075` temporal-extrapolation rollout errors.

For the few-shot validation initial states, encode the identical initial field
from high-budget E10 grid, warped-mesh, uniform-particle, and warped-particle
observations using fresh geometry seeds `40000` through `40003`. Apply only the
coefficient operator and decode on the canonical grid. Gate the maximum over
every Cartesian cross-family realization pair; never average before
comparison.

Record model, schedule, contract, runner, Git, config, checkpoint, and result
SHA-256 values. Require exact frozen configuration, source equality to a clean
committed Git HEAD, and two byte-identical complete runs.

## Frozen gates

Classify `coefficient_operator_transfer_qualified` only if:

1. closure preflight passes;
2. `pretrained_fewshot` composite one-step coefficient and decoded NRMSE are
   each `<=0.03`;
3. its eight-step coefficient and decoded NRMSE are each `<=0.08`;
4. its final-step high-frequency spectral NRMSE is `<=0.15`;
5. its macro eight-step decoded NRMSE is `<=0.80x` `scratch_fewshot`;
6. its macro eight-step decoded NRMSE is `<=1.25x`
   `full_composite_control`;
7. zero-shot `elementary_pretrained` composite eight-step decoded NRMSE is
   `<=0.20` and `<=0.75x` persistence;
8. elementary-regime decoded rollout error after fine-tuning is `<=0.08` and
   `<=1.25x` its pre-fine-tuning value;
9. `dt=0.075` coefficient and decoded rollout NRMSE are each `<=0.12`;
10. semigroup-consistency coefficient and decoded NRMSE are each `<=0.05`;
11. the maximum cross-observation coefficient and decoded mismatch is
    `<=0.01`;
12. advection mean-mode relative error is `<=1e-3`, advection relative
    `L2`-norm drift is `<=0.05`, and at least `99%` of diffusion rollout steps
    have nonincreasing predicted energy;
13. every provenance and boundary assertion passes.

Classify `coefficient_operator_capable_without_transfer` if closure,
pretrained-fewshot absolute accuracy, temporal, invariance, physics, provenance,
and boundary gates pass but either transfer ratio gate fails.

Classify `coefficient_dynamics_not_qualified` if closure, absolute accuracy,
temporal, invariance, physics, provenance, or boundary fails.

## Boundary

- synthetic smooth periodic scalar fields only;
- linear advection, diffusion, and their composition only;
- no nonlinear dynamics, shocks, boundaries, irregular domains, moving
  particles, vector fields, coupled physics, or arbitrary topology;
- no representation, family, or task label enters the model;
- no original observations remain available after exact projection;
- no reserved held-out data, provider call, router, expert, or public/claim
  promotion;
- success permits only the next preregistered coefficient-dynamics expansion;
  it does not establish a universal simulator.
