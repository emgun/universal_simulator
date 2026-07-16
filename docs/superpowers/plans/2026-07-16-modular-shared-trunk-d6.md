# D6 Modular Shared-Trunk Validation Plan

Date: 2026-07-16

Status: implementation scope frozen; executable artifact plan must be generated
from the committed implementation before any remote run

## Run contract

- Run tag: `strat-v1-modular-shared-trunk-d6`
- Branch: `codex/modular-shared-trunk`
- Question: can small task-routed latent adapters remove D5's cross-task
  interference while retaining one shared conditioned operator trunk and the
  existing shared physical codecs?
- Sole architectural variable: paired zero-initialized residual adapters
  immediately before and after the shared operator core. Each task receives
  one bottleneck-16 input adapter and one bottleneck-16 output adapter, routed
  densely by the existing frozen `task_id`. The encoder, decoder, conditioning,
  and PDE-Transformer trunk remain shared and unchanged.
- Primary metric: equal-task macro of Advection h16 rollout NRMSE, Burgers h16
  rollout NRMSE, and Darcy coefficient-to-solution NRMSE.
- Local harness: focused unit and integration tests plus a synthetic routed
  codec forward/backward and checkpoint round-trip.
- Remote harness: one hash-bound, validation-only seed-17 D6 run after the
  implementation and executable plan are committed.

## Arms

1. `joint-modular`: three paired task adapters and one shared conditioned trunk.
2. `ablation-advection1d`: the identical module graph trained only on
   Advection.
3. `ablation-burgers1d`: the identical module graph trained only on Burgers.
4. `ablation-darcy2d`: the identical module graph trained only on Darcy.

All arms retain the complete three-task adapter inventory and universal
conditioning vocabulary. Routing selects the active task expert while the
identical full module graph keeps architecture and checkpoint accounting
comparable.

## Mutable implementation surface

- the latent operator's optional routed-adapter module;
- identical adapter construction in `scripts/train.py` and
  `scripts/evaluate.py`;
- one D6 config, planner, runner, independent materializer, and their tests;
- compact D6 plan and result artifacts after their respective evidence exists.

## Fixed surfaces

- the six-object `strat-v1` training lock and all object identities;
- train/validation membership, transformations, regime composition, and
  `strat-v1.1` metrics;
- seed 17, latent dimension/tokens, operator dimensions/depths/heads, stage
  epochs, batch sizes, optimizers, rollout horizons, and selection behavior;
- loss-dependent early stopping is disabled so the frozen stage schedule gives
  every joint task and matched ablation equal scheduled source exposure;
- out-of-memory events fail the run instead of silently skipping a sample or
  batch and corrupting exposure parity;
- D5 result and specialist reference values;
- parameter/task/equation/boundary/resolution conditioning schema and the
  shared grid encoder/decoder;
- data loading, staging, evaluator, checkpoint hashing, signed B2 transfer, and Vast
  teardown contracts.

## Forbidden moves

- any test-role object, measurement lock, or held-out path;
- task-private capacity beyond the frozen bottleneck-16 input/output adapters;
- Python or argmax task routing, ambiguous/missing routes, or adapters enabled
  with nonzero semigroup loss;
- new dependencies, dataset changes, normalization changes, extra seeds,
  epoch extensions, threshold relaxation, or replacement runs;
- Poseidon, additional PDEBench families, or The Well during D6;
- comparing a joint modular arm with the old D5 specialists as if they were
  matched U2 controls.

## Gates

U1 compares `joint-modular` with the frozen D5 references:

- macro NRMSE `<= 0.7584231366` (`1.10x` the D5 specialist oracle);
- Advection `<= 0.6388001070`, Burgers `<= 0.6895805941`, and Darcy
  `<= 0.1403299866`;
- beat persistence and corrected regime spread `<=1.5` on every task;
- shuffled-parameter degradation `>=5%`;
- fewer checkpoint bytes than the frozen three-specialist ensemble, plus a
  lower initialized tensor-element count than the three matched D6 ablations;
- held-out reads exactly zero.

U2 compares `joint-modular` with the three new matched ablations:

- joint macro `<=1.05x` the equal-task ablation macro;
- no joint task metric `>1.10x` its matched ablation;
- exact scheduled per-task source exposure and rollout-weighted compute must
  match; total scheduled optimizer steps, initialized tensor elements,
  checkpoint bytes, runner wall time, and process-memory high-water marks must
  be reported separately as consolidation evidence rather than mislabeled as
  per-task optimizer parity or GPU memory.

## Baseline and stop conditions

The baseline is the immutable D5 negative result; it will not be retrained.
Shared macro NRMSE was `0.7972321757`, versus the frozen specialist oracle
`0.6894755787`, with held-out reads zero.

If U1 fails, close modular shared-model research at this scale and narrow the
product to a unified interface over family-specific models. If U1 passes but
U2 fails, sharing remains negative and also stops. Only a joint U1/U2 pass may
authorize a separately preregistered U3/U4 experiment. D6 never authorizes
held-out access by itself.

## Evidence artifacts

The executable run must produce self-hashed `plan`, `stage`, `summary`, and
independently materialized `result` artifacts. They must bind source commit,
source-file hashes, config, training lock, staged objects, module ownership,
exposure parity, metrics, resource accounting, and held-out reads.
