# Universal Baseline Experimentation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` for concrete tasks. Selection is validation-only; held-out test runs require a pre-registered contract and ledger key, exactly as before. Do not change datasets, splits, gates, or promotion rules during an experiment unless this file is updated first.

**Goal:** A solid, working universal/general simulation baseline: one system, trained once across task families, that beats persistence and the strong external baselines on held-out test for every task under an honestly-constructed protocol — with regime handling (parameter conditioning + online inference) as a first-class, documented capability rather than an accident.

**Architecture:** Fix the protocol first (regime-stratified splits), then re-select the two strongest existing candidates under it (Poseidon channel-lift family; in-house tier_b core, which earned a retrial), and make regime conditioning explicit. Breadth (universal-v1) comes only after the three-task baseline is solid.

**Tech Stack:** Existing UPS harness (`run_light_experiment.py`, external-baseline runners, Vast/B2/W&B plumbing), the official beta-provenance hydration pipeline, `scripts/diagnose_advection_split_shift_distributions.py` for regime auditing.

---

## 1. What changed and why this plan exists

The 2026-07-08 regime diagnosis and 2026-07-09 split-integrity audit proved
that `light-v1` and `medium-v1` are mixed protocols: Advection reuses initial
conditions across single-regime splits (train beta 0.1, val beta 4.0, test beta
7.0), Burgers test trajectories occur in training, and Darcy is disjoint.
Consequences that reshape the experiment queue:

1. **Legacy Advection failures cannot isolate model failure from regime shift.** The Poseidon channel-lift candidate (val 0.3578, G2a passed) and the model-side candidate were selected on beta 4.0 and tested on beta 7.0 after training only on beta 0.1. The scoped beta-head result later confirmed that explicit beta conditioning can transfer tightly, but under a different mixed-root inference contract.
2. **P1's "structural rollout collapse" verdict is confounded for Advection.** The in-house operator trained exclusively on near-static beta-0.1 data. Its model-drift observation remains useful, but Burgers test contamination and Advection regime shift limit broader conclusions. The core earns one cheap validation retrial on `strat-v1` before the "explore-only" demotion is final.
3. **Val-selection only means something on stratified splits.** Until the protocol is fixed, further candidate spend is structurally capped.

Frozen protocols stay frozen: `light-v1`/`medium-v1` comparisons remain valid
only within their matched mixed construction. Advection is a scoped
transport-speed extrapolation track; Burgers is training-set reproduction; and
Darcy is disjoint generalization. None is promoted as a uniform broad
generalization benchmark.

## 2. Definition of "solid working universal baseline" (exit criteria)

All of the following, under the new stratified protocol (`strat-v1`):

- **B1 — beats trivial physics:** held-out `decoded_rollout_nrmse` better than persistence on every task family and overall, with no estimator assistance.
- **B2 — beats strong baselines:** held-out overall better than the best re-measured external baseline (FNO/UNO/U-Net/CNO under strat-v1).
- **B3 — regime-general in-distribution:** per-regime held-out breakdown (e.g., per-beta) shows no regime catastrophically worse; reported in the evidence.
- **B4 — legacy mixed protocol reported separately:** `light-v1`/`medium-v1` task-level numbers retain their explicit construction labels (Burgers reproduction, Advection regime extrapolation on reused initial conditions, Darcy disjoint generalization) and are never mixed into B1-B3.
- **B5 — one system:** a single trained artifact with documented conditioning inputs (task id, physical parameters when available, optional online context inference mode), not per-task specialists.

## 3. Track A — Protocol foundation: `strat-v1` (prerequisite, mostly CPU)

- [x] A1: Audit existing roots. Done 2026-07-09 (`docs/research/2026-07-09-split-integrity-audit.md`): Burgers test is fully contained in train (light and medium tiers); advection shares all initial conditions across splits (regime-only differences); Darcy is clean. This escalates A2 from improvement to requirement.
- [x] A2: Stratified shard builder and release. **Completed 2026-07-13.** Advection `256/64/64`, Burgers `288/72/72`, and Darcy `260/65/65` are checksum-bound, balanced, provenance-disjoint, and published at immutable B2 keys. The combined training lock fetched six objects (427,029,641 bytes) into an empty training cache; the separately authorized measurement lock fetched three reserved test objects (77,222,552 bytes) into a different empty cache. All nine passed independent size/SHA-256 verification. No model evaluation ran. Exact releases are under `docs/data/releases/strat-v1/`.
- [x] A3: Frozen 2026-07-13 in `docs/research/2026-07-13-strat-v1-contract.md` before any `strat-v1` candidate or baseline number existed. It fixes tasks, splits, temporal and steady metrics, validation-only selection, physical-parameter disclosure, held-out ledger rules, and separate primary/legacy reporting tracks.
- [ ] A4: Baselines under strat-v1: persistence plus the four external baselines re-measured (val + one held-out each, pre-registered as baseline measurements). **Validation completed 2026-07-13:** all 12 applicable summaries are frozen in `docs/research/artifacts/strat_v1_a4_validation_scorecard.json`; UNO is the three-task macro wall at `0.677776`. CNO1d is correctly scoped to Advection/Burgers. No test object was staged or opened. Remaining DoD: pre-register and run the baseline-only held-out sequence, then freeze the final B2 wall. (~2-3 GPU-hours originally estimated)

## 4. Track B — Backbone candidate: Poseidon channel-lift family (primary bet)

Rationale: already the best validated no-estimator model measured here (0.3578
legacy `light-v1` validation, 13-43 trainable parameters, pretrained interface
intact). The mixed legacy protocol prevents a clean attribution of its held-out
failure. Under `strat-v1`, validation selection can finally measure
regime-general competence on disjoint trajectories.

- [ ] B1: Re-run channel_lift Option A and task-modulated Option B on strat-v1 train/val. Selection on validation with the per-regime breakdown required (no regime > 1.5x the mean). (~1 GPU-hour)
- [ ] B2: If validation clears the strat-v1 gate (better than best external baseline val), pre-register one held-out contract for the selected variant and run it once. (DoD) First strat-v1 held-out candidate result. (~0.5 GPU-hour)
- [ ] B3 (conditional): LoRA Option C only if A/B validation is within 10% of the gate but not past it. Budget <= 3 GPU-hours, pre-registered kill criterion.

## 5. Track C — Regime handling as a capability (parallel, cheap)

- [x] C1: **Scoped beta-head held-out diagnostic complete.** The registered measurement key `9c028afb...` ran exactly once on 2026-07-11 and passed its gates: held-out overall `0.12976493407013082` versus validation `0.11122069865007121`. It confirms the regime diagnosis under the scoped beta-provenance mixed root, but is not a primary result or public-table comparable. Do not rerun the key. See `docs/research/2026-07-11-beta-head-heldout-result.md`.
- [ ] C2: Parameter-conditioning interface: promote `param:beta`-style conditioning into the strat-v1 candidate contract as an allowed, documented input (metadata present in strat-v1 shards by construction). Candidates may use it; the evidence must say so.
- [ ] C3: Inferred-parameter mode: a small head that estimates the regime parameter from early context and feeds the conditioning interface (merges the proven online-context robustness with the parameter interface; the May inferred-transport work is the prior art). Train/val only on strat-v1; promote only through the standard gates. (~2 GPU-hours)

## 6. Track D — In-house core retrial (one shot, then decide)

- [ ] D1: tier_b (759K params) retrained on strat-v1 medium-tier train with the P1.2 recipe, evaluated on strat-v1 val with per-regime breakdown. One run, ~0.5 GPU-hour. 
  - If advection val improves dramatically (persistence-competitive), the "structural" verdict was data-confounded: re-open a bounded capacity/recipe ladder under strat-v1 (budget 10 GPU-hours).
  - If advection improves but Burgers/Darcy drift persists, P1's drift finding stands: keep the core as explore-track only and let Track B carry the baseline.
  - Either way this resolves an honesty debt in the P1 record; the result is appended to the P1 research notes.

## 7. Track E — Breadth (after B1-B5 hold on three tasks)

Unchanged from the north-star roadmap Phase 3, with one hard amendment: **universal-v1 shard prep must stratify every task's splits by its regime parameter(s) and record the composition in the manifest.** Target 8-12 PDEBench families plus one mesh and one particle task; baselines measured before candidates; the strat-v1 machinery generalizes directly.

## 8. Sequencing and budget

| Step | Depends on | Cost | Decision it feeds |
|---|---|---|---|
| C1 beta-head pretest (complete) | registered contract | CPU measurement; two failed remote hydration attempts | Scoped regime-conditioning diagnostic only |
| A1 split audit (complete) | none | CPU | Scope of A2 |
| A2-A3 strat-v1 build + contract (complete) | A1 | CPU + durable object storage | Everything downstream |
| A4 baselines | A2-A3 | ~$1 | The B2 wall |
| D1 core retrial | A2-A3 | ~$0.25 | Core: retry vs explore-only |
| B1 Poseidon re-selection | A2-A3 | ~$0.50 | Primary candidate |
| B2 held-out | B1 gate | ~$0.25 | **The baseline claim** |
| C3 inferred-parameter mode | A2 | ~$1 | The "general system" story |
| Track E breadth | B1-B5 met | ~$30-60 | universal-v1 |

Total to a defensible three-task universal baseline: **roughly $5 of GPU** plus CPU work — because all the expensive lessons (what fails and why) are already paid for.

## 9. Kill criteria / honesty rules

- Any candidate whose strat-v1 validation per-regime spread exceeds 1.5x mean does not get a held-out contract; fix the regime gap on validation first.
- Track D gets exactly one retrial run before its verdict is final; no incremental sweeps without the D1 signal.
- Legacy light/medium-v1 numbers are never mixed with strat-v1 numbers in a single claim sentence.
- Every remote run: destroy the instance after collection (storage fees accrue on stopped instances); merge automation only on green CI.

## 10. Relationship to existing plans

- Extends `docs/superpowers/plans/2026-06-09-universal-simulator-north-star-roadmap.md`: Phase 2 continues through Track B; Phase 3 (universal-v1) is amended with the stratification requirement; the Phase 1 verdict is partially reopened via Track D per the 2026-07-08 diagnosis.
- The claim ledger discipline (`docs/claim_evidence/universal_sota_roadmap.md`) is unchanged and governs all held-out language.
