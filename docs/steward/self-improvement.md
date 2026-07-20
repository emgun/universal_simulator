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
