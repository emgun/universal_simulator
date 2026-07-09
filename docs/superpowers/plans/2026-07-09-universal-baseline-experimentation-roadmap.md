# Universal Baseline Experimentation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` for concrete tasks. Selection is validation-only; held-out test runs require a pre-registered contract and ledger key, exactly as before. Do not change datasets, splits, gates, or promotion rules during an experiment unless this file is updated first.

**Goal:** A solid, working universal/general simulation baseline: one system, trained once across task families, that beats persistence and the strong external baselines on held-out test for every task under an honestly-constructed protocol — with regime handling (parameter conditioning + online inference) as a first-class, documented capability rather than an accident.

**Architecture:** Fix the protocol first (regime-stratified splits), then re-select the two strongest existing candidates under it (Poseidon channel-lift family; in-house tier_b core, which earned a retrial), and make regime conditioning explicit. Breadth (universal-v1) comes only after the three-task baseline is solid.

**Tech Stack:** Existing UPS harness (`run_light_experiment.py`, external-baseline runners, Vast/B2/W&B plumbing), the official beta-provenance hydration pipeline, `scripts/diagnose_advection_split_shift_distributions.py` for regime auditing.

---

## 1. What changed and why this plan exists

The 2026-07-08 diagnosis (`docs/research/2026-07-08-advection-split-regime-diagnosis.md`) proved that light-v1 and medium-v1 advection splits are disjoint single-regime slices (train beta 0.1, val beta 4.0, test beta 7.0). Consequences that reshape the experiment queue:

1. **Every held-out advection failure was regime extrapolation, not model failure.** The Poseidon channel-lift candidate (val 0.3578, G2a passed) and the model-side candidate both died on a 75%-faster transport regime that validation could not select for.
2. **P1's "structural rollout collapse" verdict is confounded for advection.** The in-house operator trained exclusively on near-static beta-0.1 data. Its Burgers/Darcy drift finding stands; its advection finding does not. The core earns one cheap retrial on stratified data before the "explore-only" demotion is final.
3. **Val-selection only means something on stratified splits.** Until the protocol is fixed, further candidate spend is structurally capped.

Frozen protocols stay frozen: light-v1/medium-v1 claims remain valid and scoped, and those benchmarks are reinterpreted as zero-shot transport-speed extrapolation tests (a feature, kept as a separate reporting track).

## 2. Definition of "solid working universal baseline" (exit criteria)

All of the following, under the new stratified protocol (`strat-v1`):

- **B1 — beats trivial physics:** held-out `decoded_rollout_nrmse` better than persistence on every task family and overall, with no estimator assistance.
- **B2 — beats strong baselines:** held-out overall better than the best re-measured external baseline (FNO/UNO/U-Net/CNO under strat-v1).
- **B3 — regime-general in-distribution:** per-regime held-out breakdown (e.g., per-beta) shows no regime catastrophically worse; reported in the evidence.
- **B4 — extrapolation measured separately:** the legacy light/medium-v1 extrapolation numbers reported side-by-side as a scoped capability, not mixed into B1-B3.
- **B5 — one system:** a single trained artifact with documented conditioning inputs (task id, physical parameters when available, optional online context inference mode), not per-task specialists.

## 3. Track A — Protocol foundation: `strat-v1` (prerequisite, mostly CPU)

- [x] A1: Audit existing roots. Done 2026-07-09 (`docs/research/2026-07-09-split-integrity-audit.md`): Burgers test is fully contained in train (light and medium tiers); advection shares all initial conditions across splits (regime-only differences); Darcy is clean. This escalates A2 from improvement to requirement.
- [ ] A2: Stratified shard builder. Generalize the official-advection hydration pattern (which already carries `source_file_index` provenance and the stratified block policy used by the beta-head pretest root) into `strat-v1` roots for all three tasks: every split contains every regime, split by sample index within regime, provenance datasets mandatory, manifest records per-regime counts. **Build gate:** `scripts/audit_split_integrity.py` must report zero cross-split overlap for every task (the 2026-07-09 audit found light/medium-v1 Burgers test fully contained in train and advection sharing all initial conditions across splits; Darcy was the only honest split), and the regime diagnostic must confirm every-regime-in-every-split; both audit artifacts are committed with the manifests before any candidate runs. Sizes: train 256+ / val 64 / test 64 per task (light tier) and 512/128/128 (medium tier). (DoD) Builder + manifests + hashes; shards published to B2 under `strat-v1/`; test shards built but flagged reserved.
- [ ] A3: Freeze the `strat-v1` contract doc: tasks, splits, metric (16-step decoded rollout NRMSE + per-task/regime/horizon), selection rules (validation only), held-out ledger rules, and the two reporting tracks (in-distribution primary; legacy extrapolation scoped). (DoD) Contract merged before any candidate numbers exist.
- [ ] A4: Baselines under strat-v1: persistence plus the four external baselines re-measured (val + one held-out each, pre-registered as baseline measurements). (DoD) Baseline scorecard committed; this is the B2 wall. (~2-3 GPU-hours)

## 4. Track B — Backbone candidate: Poseidon channel-lift family (primary bet)

Rationale: already the best validated no-estimator model ever measured here (0.3578 light-v1 val, 13-43 trainable params, pretrained interface intact); its only held-out failure mode is now explained by the protocol. Under strat-v1, val-selection can finally reward regime-general competence.

- [ ] B1: Re-run channel_lift Option A and task-modulated Option B on strat-v1 train/val. Selection on validation with the per-regime breakdown required (no regime > 1.5x the mean). (~1 GPU-hour)
- [ ] B2: If validation clears the strat-v1 gate (better than best external baseline val), pre-register one held-out contract for the selected variant and run it once. (DoD) First strat-v1 held-out candidate result. (~0.5 GPU-hour)
- [ ] B3 (conditional): LoRA Option C only if A/B validation is within 10% of the gate but not past it. Budget <= 3 GPU-hours, pre-registered kill criterion.

## 5. Track C — Regime handling as a capability (parallel, cheap)

- [ ] C1: **Fire the pending beta-head held-out pretest** the moment Vast credit exists (~$0.50, fully pre-registered, measurement key `9c028afb...`). Its result doubles as confirmation of the regime analysis and banks the strongest scoped number (0.111 val) if it transfers.
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
| C1 beta-head pretest | Vast credit | ~$0.50 | Confirms regime analysis; banks scoped result |
| A1 regime audit (Burgers/Darcy) | none | CPU | Scope of A2 |
| A2-A3 strat-v1 build + contract | A1 | CPU + B2 storage | Everything downstream |
| A4 baselines | A2 | ~$1 | The B2 wall |
| D1 core retrial | A2 | ~$0.25 | Core: retry vs explore-only |
| B1 Poseidon re-selection | A2 | ~$0.50 | Primary candidate |
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
