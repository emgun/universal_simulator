# Canonical Latent E11 Coefficient-Operator Transfer Result

Date: 2026-07-24
Status: `coefficient_dynamics_not_qualified`

## Decision

The frozen dense residual coefficient MLP does not qualify as the first shared
latent dynamics operator. It misses absolute rollout, transfer, temporal,
semigroup, high-frequency, retention, and physics gates.

This is not evidence against the E10 encoder or its ordered coefficient
representation. Exact 49-mode periodic closure passes; the exact projected
teacher passes every applicable metric; and independently encoding grids,
warped meshes, uniform particles, and warped particles changes post-operator
outputs only at float64 numerical scale.

Close the dense-map hypothesis without adding seeds, updates, width, or relaxed
thresholds. Keep routing closed. Next test an explicitly compositional
continuous-time coefficient generator under the same scientific controls.

## Frozen protocol

The committed contract at
`docs/research/2026-07-24-canonical-latent-e11-coefficient-operator-transfer-contract.md`
froze before state-level measurement:

- the first 49 periodic modes of the E7/E10 ordered 52-coefficient space;
- an exact float64 Fourier teacher for periodic linear advection-diffusion;
- 768 elementary pretraining trajectories, eight composite few-shot
  trajectories, a 256-trajectory full-data control, and 64 validation states;
- one 19,828-parameter residual MLP receiving only coefficients and continuous
  `[v_x, v_y, nu, dt]`;
- identical initialization and shared few-shot batches for pretrained and
  scratch arms;
- no representation/task label, route, expert, original-observation bypass,
  provider call, or held-out read.

Two implementation defects were repaired without changing this contract. A
first attempt stopped without a result when constant elementary physical axes
collapsed generic quartile bins. Independent review then required literal
arm-by-regime coverage and replication of the complete decision, not just raw
metrics. Artifacts from execution HEADs `ed655b74...` and `f5204417...` are
superseded.

The accepted executable at clean Git HEAD
`52318903d77fd763c10927c71f56300f69e1f1dc` records every arm across composite,
x-advection, y-advection, and diffusion where applicable; explicitly gates
finite closure values; writes complete replicated decisions; and produces a
detached hash manifest.

## Representation, closure, and oracle controls

| Closure metric | Observed | Gate |
| --- | ---: | ---: |
| Minimum projection rank | `52` | `52` |
| Maximum truth-to-projection decoded NRMSE | `1.2422e-14` | `<=0.01` |
| Maximum eight-step composition error | `4.0974e-15` | `<=1e-10` |
| Finite projected coefficients and errors | yes | required |

Worst post-operator cross-observation mismatch is `1.5049e-14` for
coefficients and `7.0218e-15` decoded, versus the `0.01` gate.

Exact projected truth has zero rollout and temporal-extrapolation error in
every regime, maximum semigroup mismatch below `8.5e-16`, maximum advection
norm drift `4.16e-14`, and diffusion monotonicity `1.0`. The representation,
teacher, and measurement path close orders of magnitude below learned errors.

## Operator evidence

Composite validation:

| Arm | One-step decoded NRMSE | Eight-step decoded NRMSE | Final HF NRMSE |
| --- | ---: | ---: | ---: |
| Elementary-pretrained zero-shot | `0.29050` | `2.45642` | `31.6525` |
| Pretrained plus eight-shot | `0.28894` | `2.45417` | `16.0113` |
| Scratch eight-shot | `0.26572` | `1.51263` | `10.9284` |
| Full composite control | `0.28182` | `0.93814` | `2.23206` |
| Persistence | `0.24735` | `1.55845` | `12.2561` |
| Exact projected truth | `0` | `0` | `0` |

The candidate's decoded rollout error is `1.62245x` scratch and `2.61598x`
full-data. Zero-shot pretraining is `1.57620x` persistence. Pretraining provides
no positive composition transfer under this model form.

The full-data control also misses absolute accuracy. The negative is broader
than an elementary-pretraining failure: this dense residual MLP and schedule
are not an adequate operator even for the bounded linear gate.

## Stability and physics

- elementary macro decoded rollout error worsens from `0.82252` before
  fine-tuning to `1.08216` after it (`1.31566x`);
- composite `dt=0.075` decoded rollout NRMSE is `3.64030`;
- composite decoded semigroup mismatch is `0.17979`;
- x/y advection mean-mode errors are `0.0020123` / `0.0025845`;
- x/y advection relative `L2` drifts are `1.01032` / `0.80531`;
- diffusion energy is nonincreasing on only `37.6953%` of predicted steps.

The model fits scheduled transitions—few-shot final normalized loss is
`2.7357e-4` after pretraining and `2.2148e-4` from scratch—but fails to
generalize the state-independent phase rotation and contraction law.

## Reproducibility and boundary

Both complete results, including the decision and reproducibility record, are
byte-identical at SHA-256
`0cbb025eef9be73a0a6015ba6d247be929bcd4a2b7fb9e9c27439077dc85929e`.
The detached manifest SHA-256 is
`623aaae76067bc4866709a9b6f58bb598c056484b5106a99a4defd5f9849e483`.

The runner SHA-256 is
`720e2ad33b92faee49fcfbdee84c66c023b40bf1f50427f874f231ab555483eb`;
contract SHA-256 is
`c48e580357e2baddf35311cad243ceb921b5754fc9beb85bbaf44f732f360328`;
and config SHA-256 is
`224d433fbc8389141cb078ec64aebc2096962274ad331b7621289aa917247209`.
All sources byte-match clean execution HEAD. The artifact contains no
nonfinite numeric value.

The compact artifact is
`docs/research/artifacts/canonical_latent_e11_coefficient_operator_transfer_result.json`.
Held-out reads, provider calls, routes, labels, and original observations after
projection are zero.

This result covers only smooth synthetic periodic scalar linear
advection-diffusion. It does not qualify or refute nonlinear, irregular-domain,
particle, coupled, or universal physics dynamics.

## Next gate

Preregister E12 as a model-form test, not an encoder restart:

1. retain the E10 projection and E11 state/split/control definitions;
2. learn shared continuous-time generators for x-advection, y-advection, and
   diffusion from elementary data;
3. combine them additively and exponentiate the combined generator for
   zero-shot composition, with a Strang-splitting control;
4. constrain advection toward skew symmetry and diffusion toward negative
   semidefiniteness;
5. retain the dense MLP, scratch, full-data, persistence, exact-teacher,
   rollout, temporal, physics, invariance, provenance, and transfer controls.

If structured composition closes the gap, E11 isolated operator inductive
bias. If it fails while the oracle passes, audit generator identifiability and
optimization before expanding the function space. Neither outcome justifies
hidden routing.
