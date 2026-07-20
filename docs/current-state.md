# Current State

Updated: 2026-07-20

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

The D3 regime-balanced-objective follow-up is also negative and closes the
nearby FNO loss/sampling branch. With deterministic regime-complete batches,
the mean-loss control reached primary NRMSE `0.116942`, beta-100 global-scale
NRMSE `0.257621`, spread `2.202992`, and plateaued at epoch 384. The matched
`0.5 mean + 0.5 worst-regime` candidate worsened primary NRMSE to `0.123019`
and beta-100 NRMSE to `0.270654`, barely changed spread to `2.200100`, and did
not plateau. Causal beta diagnostics remained strong and no held-out data was
read. The remaining imbalance is not solved by equal regime counts or a simple
worst-regime penalty; do not continue FNO objective or sampling variants.

The D4 conditioned-UNO architecture comparison is negative and closes the
nearby single-task specialist branch. At epoch 384, UNO reached primary NRMSE
`0.141998`, beta-100 global-scale NRMSE `0.315154`, and corrected spread
`2.219424`, all worse than the frozen D3 conditioned-FNO control (`0.116942`,
`0.257621`, and `2.202992`). It did not plateau. Shuffled-beta degradation
(`8.4713x`) and counterfactual sensitivity confirm causal parameter use, so the
failure is not an ignored-conditioning artifact. No held-out data was read.

The complete 15.3 MiB run bundle is immutable at B2 SHA-256
`bc917695cdec16a517995036576933628a8d9a3136ad2f1fb1bffaaa2e5b78b7`.
Compact plan, stage, and selection evidence is committed under
`docs/research/artifacts/strat_v1_1_reference_recipe_adequacy_*.json`; the run
receipt and exact object key are in
`docs/research/2026-07-14-strat-v1-1-reference-recipe-adequacy-result.md`.

## `strat-v1.1` Shared `tier_b` D5

The one allowed validation-only native `tier_b` retrial completed on Vast and
is negative. The shared candidate's macro NRMSE was `0.797232` versus the
frozen specialist oracle's `0.689476`, a `1.1563x` ratio that fails the
preregistered `<=1.05` U1 gate. It regressed Advection and Burgers, improved
Darcy relative to its weak frozen specialist, but still failed Darcy corrected
regime spread at `2.1931 > 1.5`. Shuffled conditioning degraded macro NRMSE by
only `4.4947%`, just below the required `5%` parameter-use signal.

The checkpoint consolidated `67,752,971` specialist bytes into `23,688,828`
bytes, about a 65% reduction, and held-out reads remained exactly zero. Those
two passing gates do not offset the failed accuracy, per-task, regime, and
conditioning gates. Close the native monolithic `tier_b` branch at this scale;
do not run D5b, extra seeds, relaxed thresholds, U3/U4, or held-out measurement.
See `docs/research/2026-07-15-strat-v1-shared-tier-b-d5-result.md`.

## Modular Shared-Trunk D6

D6 completed once under the hash-bound v5 recovery contract and failed both U1
and U2. Joint macro rollout NRMSE was `0.809862` versus `0.662876` for the
matched single-task ensemble and `0.689476` for the frozen D5 specialists.
Joint-to-ablation ratios were `1.336892`, `1.465869`, and `1.010490` for
Advection, Burgers, and Darcy. Darcy corrected spread failed at `2.173540`, and
parameter shuffling changed macro NRMSE by only `1.184208e-6` relative.

Checkpoint and initialized-tensor consolidation passed, as did exact update
parity, but these do not offset failed accuracy, negative-transfer, regime, and
parameter-use gates. Held-out reads were exactly zero. The immutable archive
SHA-256 is `3e58f7fea593f46e05389c9260a13ac33f60eca44e157cdb06234a9c1eaf9bcc`,
and Vast reports zero active instances.

The 2026-07-17 encoder-contract audit narrows what D6 establishes. D6 is a
negative end-to-end **grid** candidate, not an isolated test of a universal
latent operator and not a grid/mesh/particle representation test. The
12-epoch operator stage used a materialized encoder outside its optimizer, the
6-epoch decoder stage froze that encoder, and only the final 4-epoch joint
stage optimized codec and dynamics together. No codec-only, latent-geometry,
paired-discretization, cross-decoding, or resampling-invariance metric was
recorded. Codec-versus-dynamics causality is therefore unresolved and hidden
family routing is paused. See
`docs/research/2026-07-17-universal-latent-encoder-audit-plan.md` and
`docs/research/artifacts/strat_v1_d6_universal_latent_contract_audit.json`.

E1 now isolates a substantial codec-path contribution without invoking the
operator. Joint-to-matched global reconstruction NRMSE ratios are `2.4922x`
Advection, `1.2396x` Burgers, and `2.9357x` for the Darcy coefficient. The
Darcy solution target is poorly reconstructed by both codecs (`0.9038` joint,
`0.8336` matched global NRMSE) because standalone decoder training supervised
the coefficient but not the solution. Joint/matched latent CKA is above
`0.998` on every task, while cross-decoding fails, showing co-adapted latent
bases rather than an interchangeable common space. The result is
`docs/research/2026-07-17-universal-latent-encoder-e1-result.md`.

The UPT lineage audit now identifies the deeper implementation gap. UPT uses
one point-set encoder with geometry-aware supernodes, transformer processing,
and learned Perceiver queries. UPS instead had separate grid and mesh paths;
the mesh path used storage-order chunk means and adaptive average pooling, not
the referenced hierarchy. An opt-in canonical point-set encoder plus paired
latent/cross-decoding metrics is implemented and locally verified. This is E2
mechanics evidence only: no codec training or scientific paired benchmark has
run, so the canonical latent basis remains `not_qualified`. See
`docs/research/2026-07-17-universal-latent-backbone-lineage.md` and
`docs/research/2026-07-17-canonical-latent-codec-qualification-contract.md`.

The 2026-07-20 measure-aware analytic E2 is now complete and negative for the
learned-query Perceiver encoder. Paired identity is strong (`1.0` retrieval,
CKA `0.9984`) and physical rank is preserved, but shared canonical-query NRMSE
is `0.3024` from grids and `0.3069` from meshes versus `0.2824`/`0.2761`
matched controls and `0.0905`/`0.0987` direct interpolation. Grid/mesh output
mismatch worsens `2.633x` under refinement. The codec is common but lossy and
non-convergent; it remains `not_qualified`. See
`docs/research/2026-07-20-canonical-latent-e2-measure-aware-result.md`.

The compact RIGNO-style regional-interaction E3 is also complete and negative.
After repairing deterministic regional-set ordering into fixed geometric slot
semantics, paired retrieval is `1.0`, CKA is `0.9978`, order invariance is
exact, and high-resolution grid/mesh mismatch is only `0.570x` the low-
resolution mismatch. Information preservation is worse: shared grid/mesh
NRMSE is `0.3856`/`0.3865` versus matched controls `0.2782`/`0.2637` and
interpolation `0.0905`/`0.0987`. E2 and E3 controls both remain roughly `3x`
worse than interpolation, so another encoder swap is not identified. Next run
a specialist codec capacity-identifiability ladder over latent length plus a
no-compression/direct-query ceiling. Reopen shared-encoder work only if a
bounded codec passes the absolute gate. See
`docs/research/2026-07-20-canonical-latent-e3-regional-interaction-result.md`.

The specialist E4 capacity-identifiability ladder closes latent-token count as
the active blocker. Compound Perceiver scaling from `8/24` to `16/48` and
`32/96` latent-token/supernode pairs does not improve reconstruction: grid/mesh
NRMSE is `0.2824`/`0.2761`, `0.2900`/`0.3283`, and `0.2994`/`0.3091`. A learned
no-compression ceiling that retains all `196` high-resolution source points
improves to `0.2620`/`0.2430` and remains stable at an unseen resolution, but
still fails the absolute `2x` interpolation gate at `2.894x`/`2.463x`.
High-frequency spectral NRMSE remains about `1.0` despite preserved gross
magnitude. Compression contributes, but neither token count nor another
encoder is the identified next lever. Pause encoder, shared-latent, operator,
and routing work; next isolate explicit relative-coordinate decoder locality
against the unchanged global `AnyPointDecoder`, using the no-compression tokens
and frozen gates. See
`docs/research/2026-07-20-canonical-latent-e4-capacity-identifiability-result.md`.

E5 now isolates decoder locality as causal. With the identical all-point
encoder, data, exposure, and gates, a fixed-radius relative-coordinate,
quadrature-aware local integral decoder improves grid NRMSE from `0.2620` to
`0.0652` (`75.12%`) and mesh from `0.2430` to `0.0758` (`68.79%`). Both beat
their interpolation baselines and pass the absolute and unseen-resolution
gates. High-frequency spectral NRMSE improves from about `1.0` to
`0.3161`/`0.3591`. The local decoder uses fewer parameters, is source-order
invariant, has no neighborhood truncation, and reproduces byte-identically.
This qualifies local decoding on the no-compression ceiling, not the universal
latent: the positive arm still reads every source token. See
`docs/research/2026-07-20-canonical-latent-e5-decoder-locality-result.md`.

E6 now tests that locality mechanism across a strict eight-token regional
bottleneck with no original-source bypass and is negative. The frozen E3 global
controls reproduce exactly. Local decoding worsens grid NRMSE from `0.2782` to
`0.3047` and mesh from `0.2637` to `0.3237`; high-frequency error worsens by
`17.58%`/`20.26%`, and both absolute and unseen-resolution gates fail. Coverage,
positive mass, order invariance, zero truncation, and parameter-budget checks
pass, while effective latent rank is only about `3.9`. Close the compact
regional-token codec. Preserve the common-latent objective, but define the next
candidate as coefficients of one physical function space and establish its
deterministic approximation sufficiency before training another encoder. See
`docs/research/2026-07-20-canonical-latent-e6-compressed-locality-result.md`.

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

1. Keep parameter value/presence universal in every candidate contract; D1-D5
   show that an available conditioning channel is necessary but not sufficient.
2. Close both the Darcy specialist search at D4 and the native monolithic
   `tier_b` branch at D5. Do not add seeds, extend epochs, relax gates, or open
   held-out access for either branch.
3. D6 failed U1 and U2 end to end. Do not add seeds, extend training, relax
   gates, run U3/U4, or open held-out access. Do not infer an operator-only
   failure or add family routing from that result.
4. E1-E6 isolate the codec failure. E5 proves physical-space decoder locality
   repairs the all-point ceiling, while E6 shows the compact eight-token
   regional state does not preserve enough field information. Close that codec
   and do not attach original source tokens around a future operator. Next
   freeze one deterministic function-space latent sufficiency test: project
   arbitrary coordinates into a shared multiresolution physical basis and
   qualify reconstruction before training a universal encoder. Do not
   instantiate an operator or router first.

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
- `docs/research/2026-07-15-strat-v1-shared-tier-b-d5-result.md`
- `docs/research/2026-07-16-d6-vast-infrastructure-failure.md`
- `docs/research/2026-07-16-strat-v1-modular-shared-trunk-d6-result.md`
- `docs/research/2026-07-17-universal-latent-encoder-audit-plan.md`
- `docs/research/2026-07-17-universal-latent-encoder-e1-result.md`
- `docs/research/2026-07-17-universal-latent-backbone-lineage.md`
- `docs/research/2026-07-17-canonical-latent-codec-qualification-contract.md`
- `docs/research/2026-07-20-canonical-latent-e2-measure-aware-result.md`
- `docs/research/artifacts/canonical_latent_e2_measure_aware_result.json`
- `docs/research/2026-07-20-canonical-latent-e3-regional-interaction-result.md`
- `docs/research/2026-07-20-canonical-latent-e4-capacity-identifiability-result.md`
- `docs/research/artifacts/canonical_latent_e4_capacity_identifiability_result.json`
- `docs/research/2026-07-20-canonical-latent-e5-decoder-locality-result.md`
- `docs/research/artifacts/canonical_latent_e5_decoder_locality_result.json`
- `docs/research/2026-07-20-canonical-latent-e6-compressed-locality-result.md`
- `docs/research/artifacts/canonical_latent_e6_compressed_locality_result.json`
- `docs/research/artifacts/strat_v1_d6_universal_latent_codec_audit.json`
- `docs/research/artifacts/strat_v1_d6_universal_latent_contract_audit.json`
- `docs/superpowers/plans/2026-07-16-modular-shared-trunk-d6-v5.md`
- `docs/superpowers/plans/2026-07-16-modular-shared-trunk-d6.md`
- `docs/research/2026-07-11-beta-head-heldout-result.md`
- `docs/claim_evidence/universal_sota_roadmap.md`
