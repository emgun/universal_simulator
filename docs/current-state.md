# Current State

Updated: 2026-07-14

## Project

Universal Physics Stack (UPS) is research software for latent-space neural
simulation of PDE-style physical systems. The canonical repository is
`/Users/emerygunselman/Code/universal_simulator`; Codex worktrees may be used
for execution, but status and roadmap decisions should resolve against that
workspace when paths disagree.

## North Star

Build one shared simulator that approaches credible task-specialist accuracy
on known systems and earns its shared design through positive transfer, lower
adaptation data, or material operational consolidation. Evidence must use
trajectory-disjoint, regime-honest protocols, validation-only selection,
reserved held-out tests, and reproducible provenance through result artifacts.

The protocol-integrity gate and A4 pipeline-calibration wall are complete.
Claim-grade specialist recipes and universal candidates remain validation-only.
Held-out access is deferred until the regime-metric erratum, recipe adequacy,
candidate selection, and universal-value gates are frozen. Active sequencing is
defined by `docs/superpowers/plans/2026-07-13-universal-value-roadmap-amendment.md`;
the hash-bound 2026-07-09 roadmap remains historical evidence.

## Current Protocol Interpretation

`light-v1` and `medium-v1` are frozen legacy mixed protocols:

- Burgers test trajectories occur in training; this is training-set
  reproduction, not held-out trajectory generalization.
- Advection reuses the same initial conditions across splits while beta changes
  from train to validation to test; this measures regime extrapolation on
  reused initial conditions.
- Darcy is trajectory-disjoint and is the only clean generalization task in
  these protocols.

Matched-protocol comparisons remain internally scoped because every candidate
used the same construction. They must not be presented as broad generalization
evidence. Source: `docs/research/2026-07-09-split-integrity-audit.md`.

## Frozen Foundation: `strat-v1`

The replacement protocol requires zero cross-split trajectory overlap and every
physical regime represented in every split. Provenance, manifests, hashes,
integrity audits, and regime-composition audits are build gates.

The only supported construction path is `scripts/make_light_hdf5_shards.py`.
It requires explicit provenance, regime, and field semantics; validates the
complete train/validation/test allocation before writing outputs; and persists
the gate result in the manifest. There is no legacy construction mode. Initial
conditions may repeat across regimes inside one split, but the same initial
condition may not cross split boundaries.

Publishing is bound to the manifest's exact artifact set, byte sizes, SHA-256
digests, task splits, and passed gates. The old positional-slicing experiment
wrappers and their launchers are archived and exit before doing work. The demo
queue can no longer schedule new `light-v1` or `medium-v1` experiments. The
historical three-task smoke pipeline remains retired; new work resolves an
exact immutable `strat-v1` lock instead of selecting a tier nickname.

The runtime data plane now applies the same integrity contract universally:

- gated shard construction emits runtime source and protocol manifests;
- `ups-data resolve` produces a deterministic, self-verifying run lock;
- `ups-data plan/stage/verify` stages exact objects into a content-addressed
  local cache and a zero-copy source-relative run view;
- training locks cannot contain test bytes, and loaders enforce the lock's
  role and exact filenames when `DATA_LOCK` is set;
- PDEBench HDF5 is lazy and worker-local; normalization is disabled unless
  train-only statistics match the lock and selection hashes;
- The Well has a source-neutral lazy adapter that preserves native HDF5,
  fields, scalars, boundaries, grids, time, and window identity. Its exact
  27-object inventory is pinned to a package commit and Hugging Face dataset
  commit. The locked train/validation pilot has passed checksum staging, a real
  lazy read, and a spawned-worker read.

W&B remains the checkpoint, metric, and evidence plane. Dataset tar artifacts,
nickname registries, fuzzy B2 discovery, direct remote HDF5 training, and
heuristic split fallbacks are retired. B2 or another object store may still be
an exact mirror URI in a source manifest when that mirror is operationally
worth maintaining.

- **Advection complete:** the forward 256/64/64 root has explicit beta,
  disjoint provenance, exact balance across eight regimes, immutable B2
  objects, and locked release controls. The original mutable publication is
  historical only.
- **Burgers complete:** all 12 official viscosity files were checksum-verified
  on ephemeral Vast scratch, the raw schema was frozen, and balanced
  `288/72/72` shards are durably published and fetch-verified.
- **Darcy complete:** all five official beta files passed exact lock, size, and
  MD5 verification. The canonical coefficient-to-solution root and balanced
  `260/65/65` shards pass provenance, coefficient-regime uniqueness, five-regime
  balance, and zero cross-split coefficient-overlap gates. Separate train-only
  input and target statistics and a real spawned-worker read also pass. See
  `docs/research/2026-07-13-strat-v1-darcy-root.md`.
- **Contract frozen:** the immutable three-task release and A3 rules are in
  `docs/research/2026-07-13-strat-v1-contract.md`. The universal training lock
  fetched 427,029,641 bytes into an empty training cache; its separate
  measurement lock fetched 77,222,552 reserved test bytes into a different
  cache. All nine objects re-verified by SHA-256 and no model ran.

## `strat-v1` Pipeline-Calibration Wall

The 2026-07-13 A4 validation run used the universal training lock only. Vast
staged all six train/validation objects directly from immutable B2 keys, ran
one universal persistence evaluation and 11 applicable task/model runs, copied
the evidence back, and destroyed the instance. No measurement-lock or test
object was staged or opened.

| Model | Advection | Burgers | Darcy | Three-task macro |
| --- | ---: | ---: | ---: | ---: |
| Persistence | `0.673296` | `0.640048` | `0.972114` | `0.761819` |
| FNO | `0.832979` | `0.306547` | `0.897273` | `0.678933` |
| UNO | `0.815870` | `0.320993` | `0.896466` | `0.677776` |
| U-Net | `0.837644` | `0.748463` | `0.992262` | `0.859457` |
| CNO1d | `2.822913` | `0.587078` | not applicable | not applicable |

UNO is the current three-task calibration wall at `0.677776`; per task, the
calibration walls are persistence for Advection, FNO for Burgers, and UNO for
Darcy. These learned rows are matched small three-epoch architecture
reproductions, not the full official-paper recipes and not the B2 claim
threshold. CNO is scoped to the two 1D tasks because the audited implementation
is CNO1d-only. The checksum-bound scorecard is
`docs/research/artifacts/strat_v1_a4_validation_scorecard.json`.

The Darcy denominator defect is repaired by the frozen metric-only
`strat-v1.1` addendum. It preserves the original task metrics and `1.5x`
threshold but normalizes every regime by one task-level validation target
scale. The validation-only reprojection touched no test object and reconstructs
every frozen task primary within `3e-9`. All Advection and Burgers calibration
rows pass the corrected spread gate. Every Darcy calibration row still fails:
persistence `2.221`, FNO `2.022`, UNO `2.089`, and U-Net `2.198`. The old
hundreds-scale values were a denominator artifact, but the corrected result
shows a real Darcy regime imbalance that candidates must address. See
`docs/data/protocols/strat_v1_1_metric_addendum.yaml` and
`docs/research/artifacts/strat_v1_1_validation_regime_diagnostics.json`.

## `strat-v1.1` Reference-Recipe Adequacy

The pre-registered validation-only FNO/UNO ladder completed on one Vast RTX
4090 using only the six-object training lock. Both continuous trajectories
used seed 17 and validation rungs `3/6/12/24/48`; no measurement lock or test
object was staged or read.

Both learning curves satisfy the declared plateau rule by epoch 24, with their
best macro validation checkpoints at epoch 6: FNO `0.651839` and UNO
`0.658227`. Neither recipe is eligible for confirmation, however, because the
selected checkpoints fail the corrected Darcy regime-spread gate: FNO
`1.9108` and UNO `1.9176` versus the frozen `<=1.5` limit. Seeds 29/43 were
therefore not run and no specialist was selected. This is a negative R0 result,
not authorization to weaken the gate or spend held-out access.

The follow-up D0 identifiability diagnostic changes the mechanism
interpretation without changing those recorded numbers. The frozen Darcy
operator depends on coefficient field and beta, but R0 passed coefficient only.
Across 13 validation provenance groups the coefficient input is byte-identical
at all five betas. Even the minimum-MSE beta-blind oracle has NRMSE `0.869025`
and maximum corrected spread `1.990764`, which closely matches the learned R0
failure. R0 is therefore an unconditioned-interface negative, not evidence that
FNO/UNO are generically inadequate. A matched validation-only coefficient-only
versus beta-conditioned Darcy FNO ablation has now completed.

That D1 ablation confirms the mechanism but does not pass the specialist gate.
Explicit beta conditioning improves Darcy validation NRMSE from `0.876806` to
`0.189475` (`78.39%`). Shuffled beta degrades the conditioned result to
`1.358415`, and counterfactual beta strongly changes its predictions, so the
gain represents real parameter use. However, maximum corrected spread is still
`2.170477` because beta `100` dominates absolute error, and the conditioned
curve is still improving at epoch 24. Treat parameter value/presence as a
universal interface requirement, but do not promote this checkpoint or access
held-out data.

The matched D2 affine-head follow-up also completed and is negative under its
frozen gate. Extending both arms to epoch 192 improved the direct conditioned
control to `0.122444`; the affine basis further improved primary validation
NRMSE to `0.104063` (`15.01%`) and beta-100 global-scale NRMSE from `0.270295`
to `0.231361`. However, corrected spread remained `2.223289` versus `<=1.5`,
and neither arm plateaued. Parameter-aware capacity is useful, but a linear
beta-affine output basis does not solve the high-beta error concentration or
optimization imbalance. No held-out data was read. The exact result and repair
provenance are committed under
`docs/research/artifacts/strat_v1_darcy_fno_affine_head_ablation_*.json`.

The complete 15.3 MiB run bundle is immutable at B2 SHA-256
`bc917695cdec16a517995036576933628a8d9a3136ad2f1fb1bffaaa2e5b78b7`.
Compact plan, stage, and selection evidence is committed under
`docs/research/artifacts/strat_v1_1_reference_recipe_adequacy_*.json`; the run
receipt and exact object key are in
`docs/research/2026-07-14-strat-v1-1-reference-recipe-adequacy-result.md`.

## Latest Model Evidence

The pre-registered model-side beta-parameter transport-head measurement
completed once on 2026-07-11 under key
`9c028afbfb85328fd21fc7de4cffb277fbde274aa042ad63e6499abc562addc3`:

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Overall decoded rollout NRMSE | `0.11122069865007121` | `0.12976493407013082` |
| Advection | `0.0017868130908052495` | `0.0011108774108008665` |
| Burgers | `0.14738121412908425` | `0.17446879896821743` |
| Darcy | `0.188979512124482` | `0.20909553062258152` |

All registered gates passed. This confirms that explicit beta conditioning can
handle the diagnosed Advection regime shift, but it uses a scoped mixed root
and a different inference contract. It is not a replacement for the primary
result, is not comparable to the public matched-protocol table, and must not be
rerun. See `docs/research/2026-07-11-beta-head-heldout-result.md`.

## Boundaries

- Do not rerun any completed held-out measurement key.
- Do not mix legacy `light-v1`/`medium-v1`, scoped beta-head, and `strat-v1`
  numbers in one comparison claim.
- Do not promote against A4 calibration rows or the defective Darcy regime
  gate. Freeze the `strat-v1.1` metric addendum, a claim-grade specialist
  recipe, and universal-value contracts before held-out access.
- Preserve the recorded Darcy provenance and paired-regime coefficient gates;
  do not reinterpret its steady operator samples as trajectories.
- Do not treat provider setup, available credit, or artifact hydration as
  authorization to weaken data, test-access, gate, or claim boundaries.
- Keep raw datasets, checkpoints, provider logs, and large run bundles in
  external artifact storage rather than Git.

## Active Work and Next Move

1. Keep parameter value/presence universal in specialist and shared candidate
   contracts; D1 and D2 confirm causal parameter use.
2. Pre-register one validation-only regime-balanced objective or sampling
   ablation that explicitly targets beta-100 error and requires plateau by cap.
   Do not extend the rejected affine-head family unchanged.
3. If balancing still misses the spread gate, test one stronger steady-operator
   architecture before any held-out access. Only after a specialist clears the
   gate, repair Poseidon and `tier_b` conditioning/evaluation
   contracts and measure shared-versus-single-task interference, low-data transfer, regime
   handling, and consolidation economics. Select exactly one specialist and
   one shared candidate before drafting any held-out sequence.

## Reopen Triggers

Revisit this snapshot when a baseline or candidate is measured under the
frozen `strat-v1` contract,
new evidence changes the legacy split interpretation, or a request would
broaden held-out access, public claims, experiment budget, or task-family scope.

## Orientation Files

- `docs/experiments/ledger.md`
- `docs/superpowers/plans/2026-07-09-universal-baseline-experimentation-roadmap.md`
- `docs/research/2026-07-09-split-integrity-audit.md`
- `docs/research/2026-07-09-strat-v1-advection-root.md`
- `docs/research/2026-07-13-strat-v1-darcy-root.md`
- `docs/research/2026-07-13-strat-v1-burgers-root.md`
- `docs/research/2026-07-13-strat-v1-contract.md`
- `docs/research/2026-07-11-beta-head-heldout-result.md`
- `docs/claim_evidence/universal_sota_roadmap.md`
