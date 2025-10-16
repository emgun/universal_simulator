# Reward-Model-Driven Test-Time Compute (TTC) Integration Plan

Paper: “Towards Reasoning for PDE Foundation Models: A Reward-Model-Driven Inference-Time-Scaling Algorithm” (arXiv:2509.02846)

This plan describes how to implement a reward-model-driven inference-time-scaling (TTC) algorithm into UPS with minimal disruption, clear milestones, and measurable definitions of done (DoD).

## Objectives
- Add TTC to improve rollout accuracy by sampling multiple candidates per step and selecting via reward models.
- Provide both physics-analytic rewards (ARMs) and learned process reward models (PRMs).
- Keep costs controllable via budgeted search (best-of-N, beam with short horizon) and efficient decoding.

## High-Level Architecture
- Candidate generation: stochastic next-step proposals from current latent state.
- Reward evaluation: score candidates using ARM and/or PRM.
- Selection policy: choose best candidate (greedy) or maintain a beam (short-horizon lookahead).
- Orchestration: TTC rollout wrapper around existing operator/diffusion pipeline.

```
LatentState z_t --(K stochastic proposals)--> {ẑ_{t+1}^k}
   -> decode minimal fields for ARM or embed latents for PRM
   -> reward scores r_k
   -> select best (or beam) => z_{t+1}
```

## Status (Oct 12, 2025)
- ✅ Implemented `src/ups/eval/reward_models.py` with analytical mass/momentum/energy scoring.
- ✅ Added `src/ups/inference/rollout_ttc.py` (best-of-N + beam-search TTC, stochastic τ/latent noise, reusable builder).
- ✅ Wired TTC into CLI/configs:
  - `configs/inference_ttc.yaml`
  - `scripts/evaluate.py` & `scripts/infer.py` auto-switch when `ttc.enabled=true`.
- ✅ Unit coverage via `tests/unit/test_ttc.py` (mass reward monotonicity, best-of-N, beam lookahead).
- ✅ Evaluation HTML/plots now include TTC reward tables + trajectories; per-step logs saved alongside reports.
- ⏳ Next: learned PRM critic, richer TTC dashboards/metrics, beam-budget sweeps.

## Implemented Interfaces
- `RewardModel.score(prev_state, next_state, context)` base protocol.
- `AnalyticalRewardModel` with configurable decoder grid, weights, negativity penalties.
- `TTCConfig` fields: `steps`, `dt`, `candidates`, `beam_width`, `horizon`, `tau_range`, `noise_std`, `noise_schedule`, `residual_threshold`, `device`.
- `ttc_rollout(...) -> (RolloutLog, List[TTCStepLog])` supporting beam planning; logs per-step rewards/indices/total reward.
- `build_reward_model_from_config(ttc_cfg, latent_dim, device)` helper (shared by eval/infer).
- `evaluate_latent_operator(..., ttc_config, reward_model)` returns TTC step logs in report details.
- CLI integration: `scripts/evaluate.py`, `scripts/infer.py`, and starter preset `configs/inference_ttc.yaml`.

## Components to Implement

1) TTC Orchestrator (new)
- File: `src/ups/inference/rollout_ttc.py`
- API:
- `@dataclass TTCConfig { steps:int, dt:float, candidates:int=4, beam_width:int=1, horizon:int=1, device: str|torch.device="cpu", reward: dict, sampler: dict, budget: dict }`
  - `ttc_rollout(initial: LatentState, operator: LatentOperator, corrector: Optional[DiffusionResidual], decoder: Optional[AnyPointDecoder], cfg: TTCConfig, reward_model: RewardModel) -> RolloutLog`
  - Policies:
    - Best-of-N (beam=1, horizon=1)
    - Beam search (beam>1) with short horizon (horizon>=2) using cumulative reward.
  - Budget: cap candidate×horizon, early-stop if reward margin exceeds threshold.

2) Reward Models (new)
- File: `src/ups/eval/reward_models.py`
- Base Protocol:
  - `class RewardModel(nn.Module): def score(self, prev_state: LatentState, next_state: LatentState, context: dict) -> torch.Tensor`
- Analytical Reward Model (ARM)
  - Constructor takes: `decoder: AnyPointDecoder`, `fields: {name->channels}`, `weights: {mass,momentum,energy,penalty}`, `grid: (H,W)`, optional boundary/domain metadata.
  - For consecutive states, decode required fields on a fixed uniform grid; approximate integrals with area-weighted sums.
  - Rewards (examples):
    - `r_mass = -|∫ρ(t+1) - ∫ρ(t)|`
    - `r_mom = -||∫(ρu)(t+1) - ∫(ρu)(t)||_1/2`
    - `r_energy = -|∫E(t+1) - ∫E(t)|`
    - Penalties: negativity (ρ, p), BC violations; optional divergence.
  - Return weighted sum (higher is better).
- Process Reward Model (PRM)
  - File: `src/ups/models/prm.py` (or colocate in reward_models if small)
  - Input: short latent window `z[t:t+H]`, `cond`; optional probe-point decodes.
  - Output: scalar reward. Train via:
    - Regression to short-horizon rollout error (MSE/nRMSE) targets, or
    - Contrastive/pairwise preference over candidate sets (InfoNCE/triplet).

3) Stochastic Candidate Generation (reuse + light tweaks)
- Use `LatentOperator` with optional `DiffusionResidual` corrector to add diversity:
  - Random τ per proposal; optional Gaussian latent noise; small `dt` jitter.
  - Optional dropout within operator/diffusion for stochasticity (config-gated).

4) Config & CLI
- New: `configs/inference_ttc.yaml`
  - `operator, decoder, run` blocks; plus
- `ttc: {enabled, candidates, beam_width, horizon, dt, reward: {analytical_weight, weights, grid: [H,W], critic: {weight, hidden_dim, dropout}}, sampler: {tau, noise_std, noise_schedule, dt_jitter}, budget: {max_prop: int, early_margin: float} }`
- Update `scripts/infer.py` and `scripts/evaluate.py` to accept TTC config and print TTC metrics (rewards, choices).

5) Logging & Metrics
- Extend `RolloutLog` to record: chosen_k, reward values per step, time per step, decode points used.
- W&B keys: `ttc/reward`, `ttc/selected_k`, `ttc/beam_width`, `ttc/horizon`, `ttc/runtime_ms`, and evaluation deltas vs baseline.

## Phased Milestones (with DoD)

M1 — MVP: ARM + Best-of-N (1–2 days)
- Implement `AnalyticalRewardModel` with mass (generic scalar), bounds penalties.
- Implement `ttc_rollout` (best-of-N only) and config parsing.
- DoD:
- On Burgers1D and Advection1D subsets, TTC reduces rollout nRMSE ≥5% with candidates∈{4,8} and ≤2× inference time.
  - Unit tests for ARM shape/finite outputs and TTC selection correctness.

M2 — Beam + Short-Horizon Scoring (2–3 days)
- Add beam search with horizon H=2–4; compute cumulative (possibly discounted) reward.
- Efficiency: cache decoded u(t); sub-sample grid for ARM; expose `decode_points` config.
- DoD:
  - Additional ≥2% nRMSE gain over best-of-N at similar compute budget on at least one task.
  - Reproducible runs with fixed seeds; budget caps respected.

M3 — PRM Training Pipeline (3–5 days)
- Add `ProcessRewardModel` and `scripts/train_reward_model.py`:
  - Generate candidate sets on ≤12.5% training data; build supervision (regression or contrastive pairs).
  - Train PRM to correlate with short-horizon error (r ≥ 0.6 Spearman on val).
- TTC can switch ARM→PRM or blend (weighted sum).
- DoD:
  - PRM-augmented TTC beats ARM-only by ≥3% nRMSE on at least one OOD split under fixed budget.

M4 — Optional RL/Planning (future)
- PRM-guided multi-step planning (value-function-like); explore constrained fine-tuning.

## API Sketches

```python
# src/ups/eval/reward_models.py
class RewardModel(nn.Module):
    def score(self, prev_state: LatentState, next_state: LatentState, context: dict) -> torch.Tensor:
        raise NotImplementedError

class AnalyticalRewardModel(RewardModel):
    def __init__(self, decoder: AnyPointDecoder, fields, weights, grid_hw=(64,64)):
        ...
    def score(...):  # returns scalar reward (higher is better)
        # decode fields, compute conservation deltas, penalties
        return reward
```

```python
# src/ups/inference/rollout_ttc.py
@dataclass
class TTCConfig: ...

def ttc_rollout(initial, operator, corrector, decoder, cfg: TTCConfig, reward_model: RewardModel) -> RolloutLog:
    state = initial.to(cfg.device)
    for t in range(cfg.steps):
        candidates = []
        for k in range(cfg.candidates):
            proposal = operator(state, torch.tensor(cfg.dt, device=...))
            if corrector and cfg.sampler.get("use_corrector", True):
                # stochastic corrector (random tau, noise)
                ...
            r = reward_model.score(state, proposal, context={"step": t})
            candidates.append((r.item(), proposal))
        state = max(candidates, key=lambda x: x[0])[1]
        ...
    return log
```

## Data & Compute Considerations
- Decoding cost: use coarse grid (e.g., 64×64), point subsampling, or task-specific fields.
- Cache current u(t) decode when scoring u(t)→u(t+1).
- Parallelize candidate scoring across candidates; torch.no_grad during TTC.

## Testing Strategy
- Unit tests:
  - ARM computes finite rewards; monotonic with synthetic conservation perturbations.
  - TTC picks higher-reward synthetic candidates deterministically.
- Integration tests:
  - Small Burgers/Advection runs comparing TTC vs baseline rollouts.

## Risks & Mitigations
- Weak rewards on some tasks: start with ARM, then train PRM on small curated sets.
- Latent→field mismatch: ensure decoder outputs required fields via config mapping.
- Runtime blowup: enforce budget caps; prefer best-of-N as default.

## Example Config Snippet (inference)

```yaml
include: inference_transient.yaml

run:
  mode: transient

operator:
  latent_dim: 512
  pdet:
    input_dim: 512
    hidden_dim: 1024
    depths: [1,1,1]
    group_size: 8
    num_heads: 4

# New TTC block
ttc:
  enabled: true
  dt: 0.1
  candidates: 6
  beam: 1
  horizon: 1
  reward:
    type: ARM
    fields: {rho: 1}    # task-dependent
    weights: {mass: 1.0, momentum: 0.0, energy: 0.0, penalty: 0.1}
    grid: [64, 64]
  sampler:
    use_corrector: true
    tau: [0.3, 0.7]
    noise_std: 0.0
    dt_jitter: 0.0
  budget:
    max_proposals: 64
    early_margin: 0.05
```

## Expected Outcomes
- Immediate: ≥5–10% rollout error reduction on select tasks with modest compute overhead.
- Medium term: PRM yields further gains, especially OOD and low-data settings.

## Next Actions
1) Implement M1 (ARM + Best-of-N), add config and minimal CLI support.
2) Validate on Burgers/Advection subsets, log TTC metrics; iterate grid size / candidates.
3) Scope M2 beam horizon and profiling; proceed if budget-effective.
4) Plan M3 PRM training data generation and losses.
