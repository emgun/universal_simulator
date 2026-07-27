# Canonical Latent E16 Multi-Realization Robustness Contract

Date: 2026-07-27
Status: preregistered after independent GO and before E16 sampled-state access

## Decision context

E10 qualified one exact 52-coefficient physical function space across the
registered grid, warped-mesh, uniform-particle, and warped-particle observation
processes. E12 then failed to recover its smooth periodic linear
advection-diffusion generator under one ordered stochastic AdamW package. E13,
E14, and E15 narrowed that failure:

- the complete elementary populations have the required excitation rank;
- the full-skew generator is expressive enough to recover the oracle;
- schedule-weighted AdamW repairs the E12 checkpoint when its moments are
  restarted, but fails from the neutral generator; and
- schedule-weighted componentwise strong-Wolfe L-BFGS recovers from both the
  neutral generator and the E12 checkpoint.

Those conclusions bind one sampled training realization. E15 explicitly
requires a compact multi-realization robustness gate before nonlinear
expansion. E16 therefore changes only the training states, elementary
parameters, and schedule draws. It is not a new representation, optimizer
search, operator family, or physics claim.

## Question

Across the sealed E15 realization and two frozen independently generated
training realizations:

1. does deterministic schedule-weighted AdamW with fresh moments repair each
   realization's own ordered-AdamW E12 checkpoint; and
2. does schedule-weighted componentwise strong-Wolfe L-BFGS recover from the
   canonical neutral generator?

A practical recovery rule is stable only if one of those packages passes every
unchanged E12/E15 recovery gate on all three realizations. Per-realization
success, mean success, best-of-realizations selection, and the E15 precedence
label are not substitutes for the complete conjunction.

## Registered realizations

The E15 result is immutable realization `r0_sealed_e15`; it is not rerun.
E16 samples exactly the following two new training realizations:

| Realization | Training-state seed | Training-parameter seed | x schedule | y schedule | diffusion schedule |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r1` | `151001` | `151101` | `172001` | `172002` | `172003` |
| `r2` | `251001` | `251101` | `272001` | `272002` | `272003` |

The six schedule seeds in the table are the literal generated seeds. The
implementation retains E12's `schedule_seed + regime_index` mechanism, with
base seeds `172001` and `272001`, but the derived values are frozen above and
must be checked rather than treated as an open offset rule. Each realization
regenerates all three 256-trajectory elementary populations and all three
`[1500, 32]` ordered schedules. State and parameter seeds are shared across the
three elementary regimes exactly as in E12; the regime identity changes only
which physical coefficient is nonzero. Each state sampler, parameter sampler,
and schedule sampler must construct its own local
`torch.Generator().manual_seed(literal_seed)`; no generator state may be shared
across axes. The validation states and parameter cases remain the single
E12/E15 frozen validation population with seeds `61001`, `61101`, `61102`,
`61103`, and `61104`.

The E12 model seed remains `71001`. The exact neutral generator makes the model
seed observational rather than a hidden realization axis, but it remains
recorded and fixed.

No further realization, seed, resampling, or replacement is allowed after
sampled-state access. A failed realization remains in the conjunction.

## Frozen representation, physics, and thresholds

Reuse without modification:

- the ordered 52-dimensional E10 coefficient space;
- active periodic coefficients `0:49` and copied inactive trend coefficients
  `49:52`;
- the exact float64 `64 x 64` Fourier linear teacher;
- the E12 smooth scalar 2-D periodic x-advection, y-advection, diffusion, and
  composite validation family;
- the 2,304-parameter full-skew additive generator;
- the combined matrix-exponential evaluation rule;
- eight-step rollout length;
- the E12/E15 validation cases and mode semantics; and
- the exact eight-bit E15 recovery vector:
  `structure`, `generator_identification`, `high_frequency`,
  `elementary_one_step_nonregression`, `elementary_rollout_nonregression`,
  `zero_shot_rollout`, `zero_shot_to_persistence`, and `finite`.

The exact nested metric thresholds are:

- relative generator Frobenius error `<= 0.10`;
- supported-entry relative error `<= 0.20`;
- off-support leakage `<= 0.10`;
- diffusion-rate relative error `<= 0.20`;
- normalized commutator `<= 0.02`;
- decoded basis-action NRMSE `<= 0.05`;
- composite final high-frequency NRMSE `<= 0.15`;
- elementary one-step NRMSE `<= 0.03`;
- elementary eight-step NRMSE `<= 0.08`;
- composite eight-step NRMSE `<= 0.20`; and
- composite/persistence ratio, using persistence
  `1.5584481380508215`, `<= 0.75`.

The broader E12 physics, semigroup, composition, and cross-observation reports
are not rerun because they are not members of E15's literal recovery vector.
Their previously frozen boundaries remain unchanged and E16 makes no new claim
about them.

No threshold may be relaxed, averaged across realizations, or replaced by a
training loss. E16 reads no held-out state.

## Realization preflight

After provenance passes and before optimization, each new realization must
record canonical hashes for:

- all three complete trajectory tensors, initial-state tensors, and parameter
  tensors;
- all three ordered schedules;
- all three schedule-occurrence count vectors; and
- the complete realization descriptor.

For float tensors, serialize a CPU, C-contiguous copy as little-endian float64
bytes in C order. For schedules and occurrence counts, serialize a CPU,
C-contiguous copy as little-endian int64 bytes in C order. Record shape,
`torch.dtype`, C-contiguity, byte-order assertion, and SHA-256 of those exact
bytes. Descriptor hashes use UTF-8 JSON with sorted keys and separators
`(",", ":")`, with no NaN. Per-regime names are literal `x_advection`,
`y_advection`, and `diffusion`.

Require:

- exactly 256 trajectories, nine states, and 2,048 transitions per elementary
  regime;
- exactly 48,000 schedule occurrences per regime;
- every transition occurs at least once;
- finite tensors and parameters within the frozen E12 ranges;
- full rank for every 48-dimensional nonconstant input covariance;
- full rank for every registered 2-D rotation-plane Gram matrix; and
- full rank for the 12-column mode-tied oracle Jacobian.

Rank uses the E13 numerical tolerance
`max(shape) * eps(float64) * largest_singular_value`. Record condition numbers
without imposing a post-observation condition threshold. Because the generator
and data law are unchanged, a rank failure classifies the realization as a
preflight failure rather than authorizing resampling.

After each realization's E12 checkpoint is constructed but before either E16
package runs, repeat E15's objective-integrity preflight at the neutral model,
that realization's checkpoint, the oracle, and the frozen synthetic source
probe. Require grouped-versus-literal output, weighted-loss, literal-schedule
loss, all-ones E13 loss, and trainable-gradient equality at `1e-14` absolute
and relative tolerance. Require exact-zero cross-block gradients and equality
between the grouped joint gradient and the arithmetic mean of the three
literal component gradients. Failure stops before E16 package optimization.

## Frozen packages

Each new realization constructs its own E12 checkpoint by replaying the exact
ordered elementary AdamW training:

- x-advection, y-advection, and diffusion schedules with 1,500 updates each;
- batch size 32 per regime;
- learning rate `0.02`;
- AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay;
- the E12 elementary update ordering and all other constructor fields
  unchanged.

The checkpoint is a prerequisite and diagnostic, not a third E16 candidate.
E16 then evaluates exactly two packages:

### `deterministic_adamw_restart`

- initialize from that realization's E12 checkpoint;
- discard every optimizer moment;
- construct fresh AdamW with learning rate `0.02`, betas `(0.9, 0.999)`,
  epsilon `1e-8`, weight decay `0`, AMSGrad false, maximize false, foreach
  `None`, capturable false, differentiable false, and fused `None`;
- require zero per-parameter optimizer-state entries before the first update;
- minimize the literal schedule-occurrence-weighted objective for 1,500
  deterministic full-objective updates;
- process x-advection, y-advection, then diffusion, backpropagating each
  component loss divided by three before one joint optimizer step;
- call zero-grad with `set_to_none=True`;
- retain trace steps
  `{0,1,2,5,10,25,50,100,250,500,1000,1500}` and the full E15 diagnostics;
  and
- classify only the final step, with intermediate validation prohibited from
  selecting, stopping, restarting, or changing training.

### `componentwise_lbfgs_neutral`

- initialize from the canonical all-zero skew blocks and unit diffusion rates;
- minimize the same schedule-occurrence-weighted objective;
- optimize `A_x`, then `A_y`, then `D`;
- use learning rate `1.0`, at most 250 iterations and 300 evaluations per
  component, history size 100, gradient tolerance `1e-12`, change tolerance
  `1e-15`, and strong-Wolfe line search; and
- retain every closure loss and gradient norm.

Each L-BFGS component uses one deterministic complete schedule-weighted
closure and runs once. Record the constructor, independent line-search state,
iterations, function evaluations, complete closure history, model/generator
hashes, and before/after diagnostics. No validation result may select a
closure, retry, polish, or alter a stopping rule.

There is no neutral deterministic AdamW arm, restart L-BFGS arm, joint L-BFGS
arm, altered weighting, new initialization, retry, polish, or best-checkpoint
selection. Those either failed, were already bounded by E15, or would reopen
optimizer archaeology instead of testing replication.

## Recovery vector and classifications

For each new realization and package, record the full E15 evaluation object and
its literal `recovery_pass` bit. Also bind these sealed E15 values:

| Package | E15 realization recovery |
| --- | --- |
| `deterministic_adamw_restart` | `true` |
| `componentwise_lbfgs_neutral` | `true` |

The aliases are literal:

- `deterministic_adamw_restart` maps to E15 control
  `schedule_weighted_adamw_restart`; and
- `componentwise_lbfgs_neutral` maps to E15 control
  `schedule_weighted_componentwise_lbfgs_neutral`.

E16 must reparse those exact E15 evaluation objects and require their
`recovery_pass` values and all eight member gate bits to be true. The sealed
E15 evidence retains its 10,584 unique mode keys and 30 literal argmax cells.
For each new realization, E16 evaluates the E12 checkpoint, the two E16
packages, and the oracle over all 49 basis indices, 18 parameter cases, and
horizons `{1,8}`. That four-control scope requires exactly 7,056 unique mode
keys and 20 literal argmax cells per realization; it is coverage, not a third
or fourth recovery package.

Define:

`stable(package) = pass(r0_sealed_e15) and pass(r1) and pass(r2)`.

Apply classification precedence:

1. `e16_preflight_failed` if provenance, environment, the E15 seal, or a new
   realization preflight fails;
2. `e16_execution_incomplete` if an optimizer, evaluation, coverage, finiteness,
   or reproducibility requirement is incomplete;
3. `both_practical_recovery_packages_stable` if both package conjunctions pass;
4. `componentwise_lbfgs_neutral_stable_only` if only neutral componentwise
   L-BFGS passes;
5. `deterministic_adamw_restart_stable_only` if only AdamW restart passes; and
6. `no_practical_recovery_package_stable` otherwise.

Only classifications 3-5 authorize preregistering the nonlinear shared-latent
gate. Classification 6 stops nonlinear expansion and records the failing
realization/package/gates without adding seeds or changing thresholds.

Direct tests must exhaust all 16 complete outcomes of
`r1/r2 x {deterministic_adamw_restart, componentwise_lbfgs_neutral}` with the
sealed `r0` bits fixed true, recompute both stability conjunctions, and verify
the resulting complete classification. Separately enumerate every preflight
and execution-completeness override so no scientific label can supersede an
earlier failure.

## Provenance, execution, and evidence

The immutable E15 source and artifact records are:

| Object | Raw SHA-256 |
| --- | --- |
| E15 contract | `5a3c826d29d65de549fbcfb4a186f6dcced7fd39c51e5169c4d5217d04bdbece` |
| E15 runner | `943558c42d2e8a13879fc3fe6f1301142efe7c7949f51e7e4ff509a6af6ae9ca` |
| E15 direct tests | `191640579e037c1e0165c1a736b0991ebbab762f59ee0d2c4c1ac0ef6cc2c8d2` |
| E15 evidence bundle | `3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a` |
| E15 compact result | `e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d` |
| E15 detached manifest | `1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311` |

The live inherited implementation must also equal these sealed E15 source
hashes, not merely the current commit:

| Source | Raw SHA-256 |
| --- | --- |
| E13 runner | `f95b1e50f409fc939c62120d06f2eafa89a864de148f9309fb38d481c01310c2` |
| E12 runner | `8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a` |
| E11 runner | `720e2ad33b92faee49fcfbdee84c66c023b40bf1f50427f874f231ab555483eb` |
| E7 runner | `cf81597b3909e9693508b62e595eb006a8598d186de062eaf4a8f241d4b07488` |
| Latent evaluation | `e2bb0fb86ac464aa6b96221d706f71aad4fd8fb48992613ead6d5b94e1943994` |

Reopen the E15 bundle in this exact order:

| Member | Bytes | Raw SHA-256 |
| --- | ---: | --- |
| `replicate_a/result.json` | `7,037,400` | `f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed` |
| `replicate_b/result.json` | `7,037,400` | `f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed` |
| `complete_result.json` | `7,038,050` | `ba2294304bdb8f97add96e4a6d39869f044a90067df41c4d682608a5d3820429` |

Require raw replicate identity and canonical payload SHA-256
`eec9fbc5bca20fc7c94600217dd13cc55d18fc779cae2b4990e0aed2be191758`.

Before sampled-state access:

- this contract, the E16 runner, and its direct tests must be committed;
- the Git worktree must be clean;
- working source bytes must match the committed HEAD;
- the imported E7, E11, E12, E13, E15, and latent-evaluation sources must be
  individually hashed, recorded, and match the committed HEAD;
- the exact E15 result, evidence bundle, manifest, contract, runner, and direct
  tests must match their registered raw SHA-256 hashes;
- the E15 classification must equal
  `deterministic_objective_adamw_restart_repairs_e12_checkpoint_only`;
- the sealed E15 recovery bits used by E16 must independently reparse as true;
- Python must equal `3.12.7` and PyTorch must equal `2.7.0`; and
- execution must use CPU float64, deterministic algorithms, one intra-op
  thread, and one inter-op thread.

The runner performs two independent whole-experiment replicas in fresh model
objects. Their canonical payloads and raw JSON bytes must be identical. The
published directory is replaced atomically only after verification and
contains:

- `canonical_latent_e16_multi_realization_robustness_evidence_bundle.tar.gz`;
- `canonical_latent_e16_multi_realization_robustness_result.json`; and
- `canonical_latent_e16_multi_realization_robustness_manifest.json`.

The deterministic gzip archive must bind source records, environment, the E15
seal, realization descriptors and hashes, schedule counts, excitation reports,
E12 checkpoint diagnostics, objective-integrity reports, optimizer
constructors and histories, complete evaluations, recovery vectors,
classification, state reads, and boundary.
Each replica has an eight-hour timeout and the whole experiment has an
eighteen-hour timeout. This is a preregistered static work-count bound: E15 used
one E12 replay plus four packages per replica under six hours; E16 uses two E12
replays plus four packages, so the per-replica ceiling is increased by one
third. It is not calibrated from E16 sampled states. Timeout or interrupt
publishes only an explicit incomplete status; it does not publish a partial
scientific classification.

Before commit, run only source-level and synthetic feasibility checks:

- Python compilation of the runner and tests;
- the focused E16 unit suite;
- the E12-E16 related unit suites; and
- tiny hand-constructed tensor tests for canonical serialization,
  occurrence-count logic, grouped/literal objective equality, classification
  precedence, deterministic bundle construction, and atomic publication.

These checks may not call `build_frozen_datasets`, `sample_coefficients`,
`sample_parameters`, or the E16 scientific runner. A failed feasibility check
blocks launch; it does not authorize timeout or scientific-contract tuning.

An independent zero-state review must reopen the bundle, verify every raw hash
and byte count, recompute the recovery conjunction and classification, inspect
coverage and finiteness, confirm raw and canonical replica identity, and verify
the boundary before the result is promoted.

## State accounting and boundary

The unique scientific state budget is:

- training trajectories: `1536` (`2 realizations x 3 regimes x 256`);
- validation trajectories: `256` (one frozen four-regime population reused
  across realizations and packages);
- held-out trajectories: `0`.

E16 makes no provider call, encoder update, routing decision, task-label or
representation-label input, source bypass, public claim, or held-out read. It
qualifies neither nonlinear nor particle dynamics. The particle statement
remains limited to E10's observation-side projection.

## Next gate

If and only if E16 finds at least one stable practical recovery package, stop
linear optimizer archaeology. The next contract should be E17: a
representation-preserving nonlinear closure test in the same 52 coefficients.
The minimal registered candidate should nest the frozen linear generator
inside a constrained quadratic convection term for a de-aliased 2-D periodic
viscous Burgers family, with a linear-only control and an explicit
non-Markov/latent-truncation diagnostic.

Do not introduce a task router, new encoder, Koopman lift, black-box neural ODE,
or neural operator in that first nonlinear gate. Those would change the
representation or capacity hypothesis before the fixed semantic latent has
been tested for quadratic closure.
