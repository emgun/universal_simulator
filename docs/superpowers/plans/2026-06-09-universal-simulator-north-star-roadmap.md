# Universal Simulator North-Star Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement concrete tasks from this plan. Use the light-experiment loop (`docs/light_experiment_loop.md`) for explore-track probes. Do not change datasets, splits, promotion rules, or held-out test selection during an experiment unless this file is explicitly updated first. All claim-facing evidence still flows through `docs/claim_evidence/universal_sota_roadmap.md`; this plan governs strategy and sequencing, not claim language.

**Goal:** Move UPS from "narrow, well-audited light-v1 claim" to a credible universal physics simulator: a learned multi-physics, multi-modality core with conservation guards, data assimilation, and control — every capability backed by frozen-protocol evidence.

**Architecture:** Two-track portfolio on one shared harness. The **exploit track** raises the capability floor: bank the existing claim, get a learned operator above persistence at credible scale, transplant a pretrained foundation backbone, freeze a breadth protocol (`universal-v1`), then demonstrate the differentiators (modality breadth, guards, DA, control). The **explore track** runs time-boxed architecture bets against a reference baseline with pre-registered gates and kill criteria. The backbone is treated as a swappable cartridge; durable architecture investment goes into everything that survives a backbone swap (conditioning, I/O stems, objectives, correctors, physics primitives).

**Tech Stack:** PyTorch, `src/ups` modules, PDEBench HDF5 shards on B2, W&B artifacts, Vast.ai GPU workers, `scripts/run_light_experiment.py`, `scripts/audit_universal_sota_status.py`, claim-evidence validators under `scripts/validate_*`, Hugging Face Hub checkpoints (Poseidon/ScOT, DPOT).

---

## 1. North Star, Stated Precisely

Two nested goals, in order:

1. **Claim north star (existing, kept):** the strongest defensible universal simulation claim under frozen, auditable protocols (`light-v1`, `medium-v1`, and later `universal-v1`), improving and comparing `decoded_rollout_nrmse` with strict validation/test separation.
2. **System north star (this plan):** a universal physics simulator — one latent interface across grids, meshes, and particles; a learned core that beats trivial and strong baselines across many PDE families; conservation-gated rollouts; latent data assimilation; steady-state solving; and safe control — where each capability has its own frozen evidence package.

The system north star is reached through the claim north star, never around it. No capability is "done" until it has a protocol, baselines, and committed evidence.

## 2. Current State (evidence snapshot, updated 2026-06-20)

What is already strong:

- Primary held-out claim: `ups_light_shared_context_transport_guarded` test `decoded_rollout_nrmse = 0.4165820594268877`, beating all five measured external baselines under the same protocol (FNO `0.6391747076887233`, UNO `0.5560551396226746`, PDEBench U-Net `0.6095843876848097`, CNO1d `0.5918753212407414`, repo physical Fourier `0.5636730976415197`) and persistence (`0.5701633411507036`) by `26.9%`.
- Medium confirmation: `ups_medium_shared_context_transport` test `0.30616533327650614` vs persistence `0.5725109200102603` (`46.5%` improvement).
- Scoped variants: CT1 online transport-context `0.20177292896682064`; data-conditioned context-phase `0.1808155304023394`; P2 parameter-conditioned canonical-root validation `0.11122069865007121` with advection at `0.0017868130908052495`.
- The full UPS component stack (M0–M12) is implemented and unit-tested; only M13 packaging remains.
- Evidence discipline (pre-test contracts, ledgers, hashes, fail-closed validators) is mature and is itself a strategic asset.

What blocks the north star:

- **The learned operator is below persistence.** Standard-root validation: operator `0.7077811986610774` vs persistence `0.3685752310100123` (persistence per task: advection `0.5140255043059492`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`). Every headline win uses `roll_persistence` transport mechanisms, not the neural core.
- **Advection transport is saturated** (P2 validation `~0.0018`); remaining light-v1 headroom is Burgers/Darcy, which only a genuinely learned operator can move.
- **Phase 1 scale probes did not rescue the in-house core.** Medium-v1 GPU smoke, capacity, rollout-stability recipe, and data-budget sweeps all completed validation-only with verified artifacts. Best capacity result was tier_b `0.7449043873888164`, best recipe result was `r_hpower = 0.7620583413339258`, and best data-budget result was `n128 = 0.7905242942784613`; all are far worse than medium-v1 persistence `0.38260034902058476`.
- **Breadth is thin:** three tasks, 1D-heavy, grid-only in the claim protocol despite multimodal code.
- **The in-house core failure mode is now specific, not speculative:** one-step prediction can be competitive, but decoded autoregressive rollouts inject drift, especially on Burgers/Darcy where persistence is naturally strong. Raw parameter count, longer-horizon training pressure, semigroup weighting, longer training, and more train samples did not reverse that.

Phase 1 execution artifacts:

- Capacity sweep: `docs/research/2026-06-10-p1-capacity-sweep-results.md`, artifact `b2://pdebench/remote-runs/capacity-sweep/capacity_sweep_medium-v1_20260610T235516Z.tar.gz`.
- Rollout-stability recipe sweep: `docs/research/2026-06-11-p1-recipe-sweep-results.md`, artifact `b2://pdebench/remote-runs/recipe-sweep/recipe_sweep_medium-v1_20260611T185755Z.tar.gz`.
- Data-budget sweep: `docs/research/2026-06-20-p1-data-budget-sweep-results.md`, artifact `b2://pdebench/remote-runs/data-budget-sweep/data_budget_sweep_medium-v1_20260621T020525Z.tar.gz`.

## 3. Strategy

1. **Exploit/explore portfolio, ~70/30 effort split.** Exploit raises the floor (Phases 0–4 below); explore hunts breakthroughs (Section 9) with pre-registered gates and kill criteria so failed bets are cheap and documented.
2. **Backbone-as-cartridge.** Do not compete with Poseidon/DPOT on pretraining scale. Transplant a pretrained backbone into the UPS harness and differentiate on what they lack: discretization-agnostic I/O, physics guards, DA/control, equation-aware conditioning, and audit-grade reproducibility.
3. **Minimum credible scale before architecture science.** Architecture comparisons at a scale where the operator loses to persistence measure noise. Establish a reference model that beats persistence first; all bets then fight that reference at matched parameters and compute.
4. **Separate harnesses, one discipline.** Claim-track work keeps the full evidence machinery. Explore-track work runs on a lighter `research-v1` scorecard (validation-only, no claim-evidence JSONs) and only graduates winners into the claim pipeline.

## 4. Phase 0 — Bank the Existing Claim (days; CPU only)

Objective: `audit_universal_sota_status.py` returns `sota_ready=true` from a clean checkout, and the advection transport track is formally closed.

- [ ] P0.1: Make the official transport objective status durable. Regenerate or restore `reports/research/sota_loop/transport_objective_status.json` (and its dependency `official_hydrated_transport_shift_gate.json`) from committed evidence, or commit them as durable artifacts under `docs/claim_evidence/artifacts/` with the audit pointed at the durable path. (DoD) Audit check `official_transport_objective_achieved` passes on a fresh worktree.
- [ ] P0.2: Make the transfer scorecard durable. Same treatment for `reports/research/sota_loop/inferred_transfer_scorecard/scorecard.json` (`partial_transfer_validated`, 2 tasks). (DoD) Audit check `transfer_signal_present` passes on a fresh worktree.
- [ ] P0.3: Make the persistence baseline row durable. Ensure `persistence_light_v1_test` (`0.5701633411507036`) is present where the audit reads it, so `light_v1_min_improvement` evaluates to `26.9% >= 20%` instead of `null`. (DoD) Audit check passes on a fresh worktree.
- [ ] P0.4: Run the full audit and commit its output as the canonical status artifact. (DoD) `sota_ready=true`; roadmap updated with the closure entry.
- [ ] P0.5: Close the advection transport track. Write the scoped-claim language for the P2 parameter-conditioned canonical-root result (validation `0.11122069865007121`) per existing decision notes; mark the warp/sidecar/gate exploration line as closed in the roadmap. No further advection sidecar PRs unless a Phase gate reopens it. (DoD) Roadmap entry merged; deprioritized-paths list updated.
- [ ] P0.6: Branch archaeology. Audit the divergent `codex/foundation-performance-roadmap` line in the main checkout (TTC/UPT/FSDP work): salvage anything useful for Phases 1–2 (especially trained checkpoints, FSDP/optimizer fixes, UPT analyses), then archive or merge it so reports stop fragmenting across checkouts. (DoD) Written summary of salvaged/retired items; main checkout no longer sits on a stale branch.

Exit gate G0: audit green from clean checkout; transport track closed; one consolidated "state of the claim" roadmap entry.

## 5. Phase 1 — Minimum Credible Scale: Operator Beats Persistence (completed negative)

Objective: a learned UPS operator — no roll-shift estimators, no context oracles, standard `data/pdebench` root — whose decoded rollout beats persistence on validation. This model becomes the **reference architecture** for all explore-track bets.

- [x] P1.1: Stand the GPU pipeline back up. Completed with a remote smoke run and verified B2 artifact.
- [x] P1.2: Scale a capacity sweep on medium-v1 (512 train / 128 val). Completed; best tier_b `decoded_rollout_nrmse = 0.7449043873888164`, worse than persistence `0.38260034902058476`.
- [x] P1.3: Data-budget sweep. Completed; best run `n128 = 0.7905242942784613`, larger data budgets did not improve the curve.
- [x] P1.4: Reference-model scale line. Closed negative: existing capacity/data/recipe levers did not produce a learned operator near the G1 persistence gate, so no reference model is promotable.
- [ ] P1.5: Light-v1 mapping. Blocked by failed G1; do not spend a held-out measurement from the in-house core.
- [ ] P1.6: Re-attach the transport signal. Deferred until there is a successor learned model; do not re-open transport sidecars against the failed fixed-tier_b core.

Exit gate G1 (hard): learned-operator-only validation beats persistence on the standard root. **Status: missed.** The best completed Phase 1 learned-operator result is still roughly 2x worse than persistence. The declared fallback is now active: skip directly to Phase 2 adapter/transplant work and treat the current in-house core as explore-track only.

## 6. Phase 2 — Backbone Transplant, Path B (active fallback, 2–6 weeks)

Objective: a pretrained foundation backbone running inside the UPS harness (UPS encoders/decoder/guards around it), fine-tuned on repo splits, beating both the Phase 1 reference and the best external baseline under the claim protocol.

- [ ] P2.1: Adapter design doc. Specify how scalar/light-v1 fields map into Poseidon/ScOT's native 4-channel input embedding and back (learned projection or channel replication + learned mix), and the equivalent for DPOT. Explicitly avoid the failed scalar-layer-replacement path (measured and stopped at validation `0.5453508470039229`). (DoD) Design doc with parameter counts, frozen/trainable split, and provenance/hash plan.
- [ ] P2.2: Poseidon frozen-backbone fine-tune (roadmap Gate 1, done properly). Train adapter + heads on train split only; validate under light-v1. (DoD) **Gate G2a:** validation `decoded_rollout_nrmse <= 0.363424243629033` (best external validation baseline, UNO) with no task collapsing near `1.0`. Continue-zone `0.363–0.5` authorizes P2.3; above `0.5` after a clean run kills the Poseidon path.
- [ ] P2.3: Controlled unfreeze / low-rank adaptation (roadmap Gate 2) if G2a is in the continue zone. LoRA or top-k block unfreezing, same split discipline. (DoD) Validation improvement over P2.2 or documented stop.
- [ ] P2.4: DPOT parallel probe (cheap). Same adapter contract for a DPOT checkpoint; one fine-tune run to compare backbone families. (DoD) One-page comparison; pick a primary backbone.
- [ ] P2.5: Held-out measurement (roadmap Gate 3). Pre-test contract, ledger key, single guarded test run for the winning transplant candidate. (DoD) **Gate G2b:** held-out `decoded_rollout_nrmse` beats the current primary claim `0.4165820594268877`. Update external-baseline mapping docs in the same PR.
- [ ] P2.6: Medium-v1 confirmation of the transplant candidate, mirroring the existing medium confirmation pipeline. (DoD) Medium evidence JSON validated.

Exit gate G2: a learned (non-heuristic) candidate is the new primary claim at light and medium scale, with the backbone documented as swappable behind the UPS interface.

## 7. Phase 3 — `universal-v1`: Make "Universal" Mean Something (3–8 weeks, overlaps Phase 2)

Objective: a frozen breadth protocol that the claim machinery can audit, covering many PDE families and — critically — the modalities that are this repo's moat.

- [ ] P3.1: Protocol contract. Freeze `universal-v1`: task list, splits, sample budgets, rollout horizon, metrics (`decoded_rollout_nrmse` + per-task/family/horizon + spectral), baselines required per task (persistence + at least one strong neural baseline), and ledger rules. Target ~8–12 PDEBench families spanning transport, diffusive, elliptic, and compressible regimes in 1D and 2D (e.g., advection, Burgers, Darcy, diffusion-sorption, reaction-diffusion, shallow-water 2D, compressible NS 2D), **plus at least one mesh task and one particle task** from the existing M2 generators or The Well. (DoD) Contract JSON + doc merged before any candidate numbers exist for it.
- [ ] P3.2: Data plumbing. Convert/stage the new families via `convert_pdebench_multimodal.py` and the dataset registry; B2 artifacts with hashes; hydration scripts proven on Vast. (DoD) All `universal-v1` splits hydrate from registry on a clean machine.
- [ ] P3.3: Baseline sweep. Persistence + the measured external baselines (FNO/UNO/U-Net/CNO where applicable) on every `universal-v1` task, validation and held-out, recorded under the contract. (DoD) Baseline scorecard committed; this is the wall candidates must beat.
- [ ] P3.4: Modality stems (architecture Bet 3, exploit-side landing). Bring the mesh/particle encoders into the evaluated path with per-modality stems feeding the shared latent backbone. (DoD) One mesh and one particle task evaluated end-to-end under `universal-v1`.
- [ ] P3.5: Candidate runs. Phase 1 reference and Phase 2 transplant candidates trained/fine-tuned on the `universal-v1` mix; validation selection; single held-out run for the winner. (DoD) **Gate G3:** held-out mean beats persistence by ≥ 20% and beats the strong baseline on the majority of families, with no family catastrophically regressed; first `universal-v1` claim entry written.
- [ ] P3.6: Extend the audit. `audit_universal_sota_status.py` (or a sibling) gains a `universal_v1` section with the same fail-closed checks. (DoD) Audit green from clean checkout for the new protocol.

## 8. Phase 4 — Differentiator Demonstrations (after G2/G3; each is its own evidence package)

These convert dormant, already-implemented capabilities (M6–M11) into headline claims no grid-only foundation model can match. Each gets a mini-protocol: frozen splits, a baseline, validation gates, one held-out run.

- [ ] P4.1: Conservation-gated rollouts. Long-horizon (≥ 64-step) rollouts with physics guards on vs off; metrics: stability horizon, conservation-budget violation, NRMSE. Baseline: unguarded model and persistence. (DoD) Evidence that guards extend stable horizon at matched accuracy.
- [ ] P4.2: Latent data assimilation. Sparse-observation trajectory recovery using the latent DA module; baseline: interpolation + persistence. (DoD) DA evidence package under a frozen observation-budget protocol.
- [ ] P4.3: Steady-state solving. Steady prior vs iterative solve cost/accuracy on Darcy-class tasks. (DoD) Steady evidence package.
- [ ] P4.4: Few-shot transfer. Hold out one full PDE family from `universal-v1` training; measure k-shot fine-tune curves (k ∈ {0, 8, 32, 128}) for the transplant backbone vs from-scratch. (DoD) Transfer-curve evidence; this is the single most "foundation-model" shaped claim available.
- [ ] P4.5: Safe control demo on one controllable task, using the existing control module with the guarded simulator in the loop. (DoD) Control evidence package.
- [ ] P4.6: M13 packaging. Export, repro scripts, model cards, and a public-facing results README that states exactly what is and is not claimed. (DoD) M13 checklist complete; a third party can reproduce the headline numbers from artifacts.

## 9. Explore Track — Architecture Bets (post-P1 fallback)

Post-P1 update: because G1 missed after the capacity, recipe, and data-budget axes were measured, the explore track is now allowed only for architecture-changing bets. Do not continue fixed-tier_b scaling, alpha schedules, more data-budget sweeps, or minor recipe tweaks against the failed core.

Rules of engagement:

- Runs on a lighter `research-v1` harness: `scripts/run_light_experiment.py` + a research scorecard directory separate from claim evidence. Validation-only. No claim-evidence JSONs, no validators, no held-out access — ever — from this track.
- Every bet pre-registers: hypothesis, opponent (the current reference model at matched params/compute), gate, budget, and kill criterion, as a short contract JSON before the first run.
- Winners graduate: a bet that beats its gate gets re-implemented on the exploit track with full claim-evidence discipline.
- Default budget per bet: ≤ 20 GPU-hours and ≤ 2 weeks wall clock before a keep/kill decision.

Bets, in priority order (from `docs/cutting_edge_architecture_research_2026-04-07.md` and the 2026-06-04 literature queue, re-ranked for the post-G1 world):

- [ ] E1: **Semigroup/horizon training objective** (Bet 1). Compositional consistency `(t0→t1→t2)` vs `(t0→t2)`, variable horizons, random temporal skips. Gate: ≥ 15% validation h16 improvement over reference at matched capacity. Kill: < 5% after tuning.
- [ ] E2: **Spectral refiner corrector** (Bet 5 / P3-refiner). 1–3 step deterministic high-frequency refinement on decoded outputs; report runtime cost alongside accuracy. Gate: ≥ 10% overall validation improvement on the reference with < 2× inference cost. Kill: improvement only on tasks where persistence already dominates.
- [ ] E3: **Physics-primitive library in the core** (new; generalizes the repo's own transport discovery). Differentiable gated primitives — semi-Lagrangian warp, local stencil residual head, spectral filters — routed by task/field/horizon features (literature P4 is the embryo). Hypothesis: explicit primitives recover what attention fails to learn at accessible scale, as proven for advection (NRMSE `0.002` via warp vs `~0.5` learned). Gate: beats reference on advection+Burgers validation without Darcy regression > 2%, with primitives demonstrably routed (not always-on). This is the repo's most credible novel-contribution candidate alongside E4.
- [ ] E4: **Equation-graph conditioning** (Bet 2). PDE-form encoder producing equation tokens; cross-attention conditioning of operator and decoder (beyond one-hot task IDs). Requires `universal-v1` breadth to be testable (needs ≥ 6 families to show generality). Gate: matches per-family specialists within 5% while one model serves all families, and improves zero-shot family transfer (ties into P4.4). Highest upside: equation-aware × any-point multimodal latent × guards is a configuration none of Poseidon/DPOT/PDEformer-2 have.
- [ ] E5: **Continuous-time RHS head** (Bet 4). Latent derivative prediction with explicit integration; valuable for irregular-time data and DA/control. Scheduled after P4.2 begins. Gate: matches discrete stepping on `universal-v1` while enabling variable-`dt` evaluation.
- [ ] E6: **Backbone-scale probe** (conditional). Only if Phase 2 transplant under-delivers: scale the in-house PDE-Transformer to ~10⁷ params on a PDEBench + The Well mix. This is the Path A fallback and needs explicit budget sign-off.

Standing exclusions (carried over from existing docs, still binding): more alpha sweeps on old checkpoints; fixed train-fitted shift regularization; scalar-only Poseidon transfer; full-stack replacement with PhysicsNeMo/DeepXDE; MoE scaling; text multimodality; published-table comparisons without protocol mapping; any held-out access from the explore track.

## 9.1 Post-P1 Next-Direction Fork (2026-06-20)

The completed Phase 1 evidence narrows the next move to a real architecture fork, not more hardening:

1. **Backbone transplant first (recommended exploit path).** Write P2.1 as an adapter design and provenance spec, then run a validation-only Poseidon/ScOT or DPOT adapter with the failed scalar-only path explicitly excluded. This has the best claim leverage because it can produce a learned, non-heuristic candidate under the existing harness without pretending the in-house core will scale itself out of the hole.
2. **Physics-primitive library first (recommended explore path if we want novel UPS-side architecture).** Pre-register a small research-v1 bet with gated differentiable primitives: semi-Lagrangian warp, local stencil residual, and spectral filter/corrector. This attacks the measured failure directly: attention-only rollouts drift, while explicit transport primitives solved advection when allowed. The risk is scope: it can become another sidecar treadmill unless the contract requires shared routing across advection+Burgers without Darcy regression.
3. **`universal-v1` protocol first (infrastructure path).** Freeze the broader protocol and data plumbing before another model bet. This improves future claim quality and prevents overfitting to three PDE families, but it is unlikely to improve the immediate learned-operator metric by itself.

Recommendation: run two short specs in sequence. First, P2.1 backbone-transplant adapter design because the roadmap fallback explicitly points there and it is mostly CPU/design work. Second, an E3 physics-primitive research contract as the UPS-native counter-bet, capped tightly so it cannot become more transport-only drift.

## 10. Decision Gates Summary

| Gate | Condition | Unlocks |
|---|---|---|
| G0 | Audit `sota_ready=true` from clean checkout; transport track closed | Phase 1 GPU spend |
| G1 | Learned operator (no estimators) validation `< 0.3685752310100123` on standard root | **Missed on 2026-06-20; fallback to Phase 2 transplant is active** |
| G2a | Transplant validation `<= 0.363424243629033` | Held-out pre-test contract (G2b) |
| G2b | Transplant held-out `< 0.4165820594268877` | New primary claim; Phase 3 candidate runs |
| G3 | `universal-v1` held-out: ≥ 20% over persistence, beats strong baseline on majority of families | Phase 4 demos; "universal" claim language |
| E-gates | Per-bet pre-registered thresholds vs reference | Graduation to exploit track |

## 11. Budget and Cadence (planning estimates, not commitments)

- Phase 0: CPU only, ~2–4 working days.
- Phase 1: ~50–100 GPU-hours (capacity + data sweeps + reference training).
- Phase 2: ~30–80 GPU-hours (adapter fine-tunes are cheap; unfreezing is the variable).
- Phase 3: ~100–200 GPU-hours (breadth training + baseline sweep dominate).
- Phase 4: mostly evaluation-scale; ~20–50 GPU-hours total.
- Explore: ≤ 20 GPU-hours per bet, ~30% of ongoing effort after G1.

Cadence: keep the small-PR, one-evidence-surface-at-a-time discipline. Weekly: re-run the audit, update the roadmap worklog, and re-rank explore bets. Any change to gates or protocols requires editing this file first.

## 12. Risks and Mitigations

- **G1 fails at accessible scale.** Mitigation: pre-declared fallback to Phase 2 transplant as the core; in-house architecture moves to explore-track only (E6 decision point).
- **Transplant adapters can't bridge modalities/channels.** Mitigation: DPOT parallel probe (P2.4) gives a second backbone family; E6 is the final fallback; the `universal-v1` protocol is backbone-agnostic and survives either outcome.
- **Breadth dilutes per-task quality.** Mitigation: G3 requires majority-of-families wins with no catastrophic regression, mirroring the existing per-task guard pattern; per-family adapters are allowed if documented.
- **Explore track scope creep** (the advection-sidecar failure mode). Mitigation: pre-registered kill criteria, hard per-bet budgets, and the standing-exclusions list; the 70/30 split is reviewed at each phase gate.
- **Evidence machinery slows exploration to a crawl.** Mitigation: `research-v1` harness is deliberately lightweight; only graduated winners pay the full evidence tax.
- **Metric gaming via persistence-shaped tasks** (Burgers h16 persistence ≈ `0.0046` makes late horizons nearly free). Mitigation: `universal-v1` must report per-horizon and per-family metrics and include families where persistence is weak; gates reference per-task floors, not only the mean.

## 13. Relationship to Existing Plans

- Supersedes the sequencing in `2026-05-11-universal-physics-sota-improvement-plan.md` (its G0–G2.5 goals are complete or closed by Phase 0 here); its benchmark-integrity rules (G0) remain binding.
- The 2026-06-07 causal transport phase estimator plan is closed by P0.5; its outputs (P2 sidecar, decoded-evaluator hooks) are consumed by P1.6.
- `docs/claim_evidence/universal_sota_roadmap.md` remains the append-only claim ledger; this plan feeds it but never bypasses it.
- The 2026-04-07 architecture research doc and 2026-06-04 literature landscape are the source rankings for Section 9; re-rankings happen here.
