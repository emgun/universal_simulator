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
