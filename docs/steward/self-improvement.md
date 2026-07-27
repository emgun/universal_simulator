# Steward Self-Improvement

This file records project-specific lessons for recurring steward ticks. It is
not claim evidence.

## 2026-06-23

- Aggregate validation gates are not sufficient for Phase 2 external-backbone
  work. Option A and Option B Poseidon runs both looked acceptable on aggregate
  but failed or missed transport/advection protection. Future GPU/provider
  plans must include explicit advection, advection h16 when available, Burgers,
  and Darcy gates before launch.
- Do not escalate adapter capacity unless the proposed change names the
  transport mechanism it is expected to fix. Small task modulation improved
  Burgers/Darcy and aggregate score, but did not repair advection.
- Vast containers used by this project may fail auto-shutdown even after the
  remote wrapper exits successfully. Any future Vast run should explicitly
  verify instance teardown and destroy the instance manually if needed.
- Strong transport-sidecar validation can still be scoped rather than
  claim-promotable if it depends on a different data/provenance contract. Future
  ticks should separate "mechanism is validated" from "public protocol can
  promote it" before suggesting held-out or claim-evidence work.

## 2026-06-24

- When a sidecar mechanism is strong but scoped, the next productive design
  should name the exact model-side insertion point and default-off contract
  before implementation. For this repo, decoder-side first is lower risk than
  latent-operator displacement because the validated signal is already decoded
  phase/displacement.
- Treat a new evaluator hook plus validator stub as mechanics evidence, not as
  validation evidence. Future ticks should require an end-to-end summary/schema
  smoke before drafting any provider plan for the model-side transport head.
- A synthetic smoke that clears all gates with trivial data proves only schema
  wiring. Future ticks should not treat it as model progress; the next decision
  needs a real-shard validation plan before any provider launch.
- For beta-conditioned transport-head work, standard `data/pdebench` advection
  shards are not enough. Future ticks must preflight `source_file_index`,
  `source_paths`, and checkpoint availability before proposing or running
  validation.
- When an audit invalidates a construction protocol, preserve its immutable
  result artifacts but remove its executable default. Compatibility is not a
  sufficient reason to let contaminated split logic remain available to new
  workflows; active callers must adopt the same fail-closed gates.
- Derive target split sizes from the number of regimes before freezing a
  protocol. Exact balance makes `256/64/64` impossible for 12-regime Burgers
  and five-regime Darcy; use `288/72/72` and `260/65/65` instead of weakening
  the gate.
- If local disk cannot fit even sequential official-data hydration, route the
  work to remote scratch rather than weakening the beta-provenance requirement.
  The remote route must keep held-out stages disabled and publish only small
  evidence artifacts, not hydrated data.
- Vast may return a contract even when the response reports failure or the
  instance never reaches usable uptime. Future launch ticks must poll sanitized
  instance status immediately, destroy stuck contracts, and relaunch with an
  explicit alternate offer rather than waiting indefinitely.
- Remote wrappers must not depend on ignored local report JSONs unless they
  explicitly hydrate or regenerate them on the instance. For this repo,
  `official_advection_hydration_plan.json` is an ignored planning artifact, so
  remote model-side transport-head runs must generate it from tracked manifest
  sources before sequential hydration.
- A model-side run that produces a strong-looking aggregate metric is still not
  accepted evidence if the required summary schema is missing. Future remote
  launches for this branch must prove locally that `scripts/run_light_experiment.py`
  summaries carry `extra.model_side_transport_head` and
  `extra.model_side_transport_head_metrics` through the same path that the
  remote wrapper uses.
- When decoded evaluation emits validator-owned metadata, do not only preserve it
  behind a generic `decoded_*` prefix. Keep backward-compatible prefixed keys,
  but also surface the validator contract keys at top-level `summary.extra` and
  test the exact summary writer path.
- If a repaired Vast relaunch immediately returns a stopped contract on an
  explicit offer, destroy it and avoid reusing that same offer without a fresh
  offer search. Treat this as an instance/offer failure, not as experiment
  evidence.

## 2026-06-25

- A model-side mechanism can still change the public inference contract if it
  requires PDE metadata not present in the frozen public protocol. Before
  drafting held-out or claim-evidence work, map whether the candidate is a
  primary-contract replacement or a scoped variant with separate language.
- Do not let a pretest contract quietly reuse validation-only data tooling.
  For beta-provenance work, the validation root builder correctly refuses
  `split=test`; a held-out path needs an explicit guarded pretest-root wrapper
  before any irreversible test command.
- For scoped held-out preparation, separate three gates: contract validation,
  test-root materialization, and held-out metric execution. The root builder may
  materialize test data inside the registered workflow, but it must not write
  the held-out ledger or run the metric command.
- Remote held-out wrappers should be separate from validation wrappers unless a
  guarded mode is obviously safer. For this beta-head branch, a dedicated
  dry-run-first wrapper keeps `split=test`, ledger writes, beta provenance, and
  artifact publication visible and fail-closed.
- Treat a provider credit failure as a hard stop before retrying, even when the
  offer and wrapper preflight are clean. Record the selected offer and dry-run
  evidence, then wait for explicit top-up or reroute confirmation.

## 2026-07-12

- For regime-stratified protocols, distinguish an initial-condition group from
  a sample identity. The same initial condition may intentionally appear under
  several regimes inside one split; require unique `(initial condition,
  regime)` pairs and unique provenance identities, while forbidding the raw
  initial condition from crossing train, validation, or test boundaries.
- Treat protocol construction as a fail-closed build, not a later audit. Verify
  provenance, within-split identities, regime balance, and cross-split overlap
  before writing shards, then persist the gate result in the manifest.
- Keep `docs/current-state.md` as a compact decision snapshot. Move completed
  chronology to the experiment ledger and evidence notes instead of allowing
  stale provider blockers to obscure the active protocol gate.

## 2026-07-17

- Do not infer that a shared latent operator failed when an end-to-end
  joint-versus-specialist comparison leaves encoder, decoder, and operator
  ownership entangled. Before routing by task or family, require codec-only
  metrics and a freeze-based causal split.
- Equal token counts and latent dimensions are an interface contract, not a
  universal representation result. Grid/mesh/particle claims require paired
  physical states, cross-decoding, and discretization-invariance evidence.
- Refuse convenient local data when its protocol differs from the frozen run.
  For D6 codec recovery, the old `pdebench.oct2025_backup` validation files are
  not substitutes for the exact locked `strat-v1` validation objects.
- For steady coefficient-to-solution systems, codec qualification must cover
  both domains. Training a decoder only on coefficient `fields` leaves the
  solution output codec entangled with operator training and invalidates an
  operator-only failure interpretation.
- Preserve pre-joint checkpoints under distinct immutable names before a joint
  stage overwrites compatibility paths. File-name variants with identical
  tensor values do not provide a before/after causal comparison.
- Audit named reference architectures at the mechanism level before treating
  repo labels as implementation evidence. Here, "supernode" and
  "Perceiver-style" concealed storage-order chunk averaging and adaptive
  pooling, while the UPT reference uses geometry selection, transformer
  processing, and learned cross-attention queries.
- Prefer one shared point-set encoder over independently trained modality
  adapters when the north star is a canonical physical latent basis. If
  modality adapters become necessary later, require paired cross-decoding and
  explicit alignment rather than assuming equal tensor shape is enough.
- Relative shared-versus-control gates can produce a false pass when every
  codec is weak. Canonical codec gates must include a deterministic direct
  interpolation baseline and absolute information-preservation requirement.
- High paired retrieval and CKA can coexist with a lossy, non-convergent common
  code. Always measure the same physical query set across a resolution ladder;
  treat worsening discretization mismatch as an encoder failure even when
  cross-decoding ratios look symmetric.
- A deterministic regional-node set is not automatically a semantically
  ordered latent sequence. Before tokenwise alignment, compare matched-slot
  distance with set distance; if the set matches but order does not, assign
  explicit geometry-bound slots and discard pre-repair metrics.
- When materially different encoders and their specialist controls all miss
  the same absolute interpolation gate, stop swapping encoders. Identify the
  codec bottleneck with a latent-capacity ladder and no-compression ceiling
  before attributing the failure to another representation architecture.
- Reproducing the frozen base rung's checkpoint hashes inside a new ladder is a
  strong no-drift control: it distinguishes a real capacity result from an
  accidental change in data order, initialization, exposure, or evaluation.
- If removing fixed-token compression eliminates cross-resolution instability
  but not absolute reconstruction error, do not keep enlarging the tokenizer.
  Hold the all-point source representation fixed and test decoder locality;
  stable gross magnitude with high-frequency spectral NRMSE near `1.0` is a
  direct signal that the query path is failing to recover local detail.
- A positive no-compression decoder result can still violate the universal-
  latent objective if the decoder reads original source tokens around the
  latent operator. Treat the all-point arm as a causal ceiling only; the next
  gate must compress into spatially anchored evolving tokens and prohibit a
  source-feature bypass.
- For mesh/grid decoder locality, freeze support in physical coordinates and
  normalize learned aggregation with quadrature weights. Report physical
  neighbor coverage and cap truncation explicitly; a nominally local k-nearest
  implementation can otherwise change its receptive field under refinement.
- When a challenger decisively passes with fewer parameters, reproduce the
  exact control checkpoints, complete result JSON, and challenger checkpoints
  byte-for-byte before changing the roadmap. This makes an inductive-bias
  conclusion stronger than a raw architecture score comparison.
- Spatial anchors and a local decoder do not guarantee an information-bearing
  latent. When a no-bypass compressed codec fails while coverage, quadrature,
  invariance, and implementation controls pass, close that tokenization rather
  than hiding the loss with routing or a source skip.
- Before asking a learned universal encoder to discover a common space, test
  whether the proposed function space itself can preserve the field through a
  deterministic coordinate- and quadrature-defined projection. This separates
  representation sufficiency from encoder optimization and gives a future
  operator a semantic state to evolve.
- When the deterministic common space passes by orders of magnitude while
  several learned tokenizers fail, treat the basis coefficients as supervised
  semantic targets for the next encoder. Do not ask latent alignment losses to
  invent coefficient meaning indirectly, and do not interpret the failed
  tokenizers as evidence for family routing.
- A quadrature moment vector can support accurate amortized projection on a
  narrow geometry set while failing on a new sampling measure. When the exact
  basis projection remains full-rank and well-conditioned but the learned
  encoder fails, expose geometry sufficient statistics such as the weighted
  basis Gram matrix to the shared correction before changing the latent space.
  Test positive transfer on held-out sampling geometries; two fixed families
  may be too similar for mixed training to provide a measurable advantage.
- Condition-number notation is part of the frozen scientific contract. For
  `G=A^T A`, `cond(G)=cond(A)^2`; never substitute a weighted-design threshold
  for a Gram threshold without a separately preregistered repair. Compute and
  record both, and stop before state reads when either frozen gate fails.
- Cross-geometry semantics must be gated on every realization pair. Averaging
  encodings before comparison can cancel sampling-dependent errors and create
  a false invariance pass; report pair count, mean, maximum, and the worst
  realization identities.
- A tiny, stable exact geometry solve can be the universal encoder rather than
  merely a teacher. Do not add an amortized inverse, routing layer, or learned
  preconditioner when a cacheable 52-dimensional factorization is accurate,
  deterministic, and cheaper scientifically. Reopen approximation only when
  basis scale or conditioning is measured as the bottleneck.
- Parameter-quartile reporting must treat a frozen physical axis as one
  explicit full-population stratum; generic quantiles create empty bins.
- “Every arm and regime” is a Cartesian evidence contract. Enforce the full
  shape in the executable for temporal, semigroup, and applicable physics
  records rather than relying on one candidate aggregate.
- A reproducibility flag is not replication evidence. Serialize the complete
  decision into each replicate, compare those bytes, and bind their hash in a
  detached manifest.
- When closure, oracle, and invariance pass but both few-shot and full-data
  dense controls fail, preserve the representation and change operator
  inductive bias before width, updates, seeds, or routing.

## 2026-07-25

- Architectural conservation and semigroup properties are not learned-physics
  evidence when the parameterization guarantees them. Gate the identified
  generator against the analytic action, including a literal worst
  basis-vector/parameter case, rather than relying on aggregate rollout.
- Small matrix-level errors can hide behind excellent global NRMSE. In E12,
  roughly `6%` generator Frobenius error and small off-support leakage coexist
  with `0.0057` rollout NRMSE but about `0.10` worst basis-action error and
  `0.295` high-frequency NRMSE. Preserve mode-resolved and spectral gates.
- A checkpoint-identical splitting rule is a useful diagnostic, not an
  alternate qualification path. When combined and splitting both miss the same
  spectral gate and differ by only `4.3e-05` decoded, audit generator
  identification before numerical integration.
- Cartesian coverage must reflect the frozen source construction exactly.
  E10 has one deterministic grid realization and four stochastic realizations
  for each other family, so grid-family pair counts are four while
  stochastic-family pair counts are sixteen.
- Hash derived parameter tensors only after detaching them from autograd. A
  provenance serializer must not change values or gradients, but it must be
  exercised on trained, gradient-tracked modules before launching replicated
  measurement.
- When an experiment is an aggregate success but a strict mode-resolved
  negative, do not weaken the gate or reopen the representation. First compare
  the learned optimizer with a direct or oracle-support-sparse recovery ceiling
  under the identical data and thresholds.

## 2026-07-26

- Exercise every frozen model API used by a preflight, including non-trainable
  oracle adapters, before the clean execution commit. `nn.Module` inheritance
  does not imply that an older frozen helper implements `forward`; a direct
  literal-oracle regression would have caught E15's zero-evidence launch
  failure.
- When a sealed invocation fails before publication, preserve the scientific
  boundary: emit an incomplete result, reproduce only enough to expose the
  engineering traceback, record a narrow pre-execution erratum, obtain fresh
  review, and rerun from a new clean HEAD. Do not silently patch the registered
  implementation or reinterpret an incomplete run.
- A precedence classification is a decision label, not the complete causal
  result. E15 names the AdamW restart because it is the first passing arm, while
  both componentwise L-BFGS arms also pass. Always record the full recovery
  vector before choosing the next gate.
- Deterministic population weighting did not rescue AdamW from neutral, and
  nearly uniform versus schedule-occurrence weights both permit L-BFGS neutral
  recovery. Do not attribute E12 to stochastic noise or weighting alone.
  Initialization basin and the componentwise second-order package remain
  coupled; resolve robustness once, then stop optimizer archaeology.

## 2026-07-27

- A zero-tail Galerkin truncation is a mechanistic control, not an upper bound
  on every deterministic closure: a learned quadratic map may compensate
  predictable resolved effects of discarded modes. Attribute representation
  failure only with registered same-latent/different-tail states or another
  genuine conditional-ambiguity bound.
- For periodic closure experiments, define truth targets with the periodic
  orthogonal projection and append inactive trends as zeros. Do not silently
  use a joint periodic-plus-nonperiodic least-squares projection on fields with
  unresolved Fourier tails; reserve that full projection for the qualified
  observation-ingestion control.
- Freeze de-aliasing at the signed-index level. At resolutions divisible by
  three, strict `abs(k)<N/3` avoids the ambiguous endpoint whose self-sums can
  alias. Record FFT ordering, normalization, comparison grid, and literal
  retained sets before state construction.
- Low-mode and conservation convergence can pass while the full nonlinear
  truth field is under-resolved. E17's first two spatial rungs had
  coefficient error below `4e-7` and energy mismatch below `6e-6` but failed
  the full-field gate. Preserve a full-field reference check, use a
  preregistered refinement/hard-stop rule, and never reinterpret a
  truth-discretization failure as latent or operator evidence.
