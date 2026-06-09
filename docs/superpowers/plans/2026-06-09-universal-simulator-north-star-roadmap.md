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

## 2. Current State (evidence snapshot, 2026-06-09)

What is already strong:

- Primary held-out claim: `ups_light_shared_context_transport_guarded` test `decoded_rollout_nrmse = 0.4165820594268877`, beating all five measured external baselines under the same protocol (FNO `0.6391747076887233`, UNO `0.5560551396226746`, PDEBench U-Net `0.6095843876848097`, CNO1d `0.5918753212407414`, repo physical Fourier `0.5636730976415197`) and persistence (`0.5701633411507036`) by `26.9%`.
- Medium confirmation: `ups_medium_shared_context_transport` test `0.30616533327650614` vs persistence `0.5725109200102603` (`46.5%` improvement).
- Scoped variants: CT1 online transport-context `0.20177292896682064`; data-conditioned context-phase `0.1808155304023394`; P2 parameter-conditioned canonical-root validation `0.11122069865007121` with advection at `0.0017868130908052495`.
- The full UPS component stack (M0–M12) is implemented and unit-tested; only M13 packaging remains.
- Evidence discipline (pre-test contracts, ledgers, hashes, fail-closed validators) is mature and is itself a strategic asset.

What blocks the north star:

- **The learned operator is below persistence.** Standard-root validation: operator `0.7077811986610774` vs persistence `0.3685752310100123` (persistence per task: advection `0.5140255043059492`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`). Every headline win uses `roll_persistence` transport mechanisms, not the neural core.
- **Advection transport is saturated** (P2 validation `~0.0018`); remaining light-v1 headroom is Burgers/Darcy, which only a genuinely learned operator can move.
- **Model and compute scale are toy:** few-KB checkpoints, CPU training, 32-sample caps.
- **Breadth is thin:** three tasks, 1D-heavy, grid-only in the claim protocol despite multimodal code.
- **Audit bookkeeping:** `audit_universal_sota_status.py` reports `sota_ready=false` from a clean checkout only because three report surfaces are not durable committed evidence (transport objective status, transfer scorecard, persistence baseline row).

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

## 5. Phase 1 — Minimum Credible Scale: Operator Beats Persistence (1–3 weeks; first GPU spend)

Objective: a learned UPS operator — no roll-shift estimators, no context oracles, standard `data/pdebench` root — whose decoded rollout beats persistence on validation. This model becomes the **reference architecture** for all explore-track bets.

- [ ] P1.1: Stand the GPU pipeline back up. One Vast.ai smoke run end-to-end (hydrate datasets from B2, train, eval, push W&B artifact). Budget: < 5 GPU-hours. (DoD) A medium-v1 training run completes remotely and its summary lands locally.
- [ ] P1.2: Scale a capacity sweep on medium-v1 (512 train / 128 val). Sweep latent dim, depth, and token count from the current toy size up through ~1M–20M params, with the existing decoded-rollout loss. Selection on validation only. (DoD) Capacity-vs-validation curve committed; best candidate identified.
- [ ] P1.3: Data-budget sweep. The light caps (32 samples) and even medium (512) are small; measure validation NRMSE vs train-sample count using full available PDEBench trajectories for the three families. (DoD) Data-scaling curve committed; chosen operating point recorded.
- [ ] P1.4: Train the reference model at the chosen capacity/data point with the existing training levers already in `scripts/train.py` (decoded field loss, rollout pressure, horizon weighting). (DoD) **Gate G1:** standard-root validation `decoded_rollout_nrmse < 0.3685752310100123` (persistence) with no roll-shift/context estimators, and per-task no worse than persistence by more than 5% on any task.
- [ ] P1.5: Light-v1 mapping. Evaluate the reference model under the frozen light-v1 validation contract; if it clears the existing phase-gate thresholds, write a pre-test contract and spend one held-out measurement. (DoD) Either a new clean primary claim or a documented gap analysis.
- [ ] P1.6: Re-attach the transport signal. With the reference model in place, enable the already-integrated `param:beta` / data-conditioned hooks and measure the combined candidate on validation. (DoD) Combined-candidate validation evidence; promotion decision recorded.

Exit gate G1 (hard): learned-operator-only validation beats persistence on the standard root. **No Phase 2 GPU spend on transplant fine-tuning and no explore-track bets start until G1 passes**, with one exception: P2.1 (adapter design) is paper/code work and may proceed in parallel.

Fallback: if G1 fails after the capacity and data sweeps (~50–100 GPU-hours), that is strong evidence the current core architecture is the problem — skip directly to Phase 2 (transplant) and treat the in-house core as explore-track only.

## 6. Phase 2 — Backbone Transplant, Path B (2–6 weeks)

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

## 9. Explore Track — Architecture Bets (continuous, ~30% of effort, starts after G1)

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

## 10. Decision Gates Summary

| Gate | Condition | Unlocks |
|---|---|---|
| G0 | Audit `sota_ready=true` from clean checkout; transport track closed | Phase 1 GPU spend |
| G1 | Learned operator (no estimators) validation `< 0.3685752310100123` on standard root | Phase 2 fine-tuning; explore track opens |
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
