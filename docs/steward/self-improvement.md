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
