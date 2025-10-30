# Test-Time Conditioning (TTC) System Implementation

## Overview

The Test-Time Conditioning (TTC) system in the Universal Physics Stack (UPS) is a greedy inference controller that improves latent operator predictions by evaluating multiple candidate trajectories at each rollout step and selecting the best one according to physics-based reward models. This document describes the actual implementation, not the idealized design.

**Current Status**: TTC is fully implemented but shows minimal improvement (~0-2%) in practice on Burgers1D test cases.

---

## Architecture

### Core Components

#### 1. TTC Rollout (`src/ups/inference/rollout_ttc.py`)

The main TTC inference loop that orchestrates candidate generation, scoring, and selection.

**Key Types**:

```python
@dataclass
class TTCConfig:
    steps: int                          # Number of forward steps
    dt: float                           # Time step size
    candidates: int = 4                 # Number of candidates per step
    beam_width: int = 1                 # Beam width for lookahead
    horizon: int = 1                    # Lookahead depth
    tau_range: Tuple[float, float]      # Range for diffusion tau sampling
    noise_std: float = 0.0              # Standard deviation of latent noise
    noise_schedule: Optional[Sequence[float]] = None  # Per-step noise schedule
    residual_threshold: Optional[float] = None        # Budget constraint
    max_evaluations: Optional[int] = None             # Hard limit on evaluations
    early_stop_margin: Optional[float] = None         # Stop if gap > margin
    gamma: float = 1.0                  # Discount factor for lookahead
    device: torch.device | str = "cpu"

@dataclass
class TTCStepLog:
    rewards: List[float]                # Immediate rewards for each candidate
    totals: List[float]                 # Total rewards (with lookahead if enabled)
    chosen_index: int                   # Index of selected candidate
    beam_width: int                     # Effective beam width used
    horizon: int                        # Effective horizon used
    noise_std: float                    # Noise applied at this step
    reward_components: Dict[str, float] # Detailed reward breakdown (mass_gap, etc.)
```

**Main Function**: `ttc_rollout()`

```python
def ttc_rollout(
    *,
    initial_state: LatentState,
    operator: LatentOperator,           # PDE-Transformer operator
    reward_model: RewardModel,          # Analytical or learned reward model
    config: TTCConfig,
    corrector: Optional[DiffusionResidual] = None,  # Optional diffusion corrector
) -> Tuple[RolloutLog, List[TTCStepLog]]:
```

**Algorithm**:

For each step in 1..steps:
1. Generate base prediction from operator
2. Sample N candidates by:
   - Starting from base prediction
   - Applying optional diffusion correction (tau ~ Uniform[tau_range[0], tau_range[1]])
   - Adding optional latent noise (std from noise_schedule or noise_std)
3. Score each candidate using reward_model
4. (Optional) Perform lookahead beam search to depth `horizon`
5. Select candidate with highest (total) reward
6. Log rewards, chosen index, and reward components
7. Advance state to chosen candidate

**Key Features**:

- **Evaluation budget tracking**: `_EvalBudget` class limits total forward passes to `max_evaluations`
- **Early stopping**: If best-vs-second reward gap exceeds `early_stop_margin`, skip lookahead
- **Lookahead**: Recursive `lookahead()` function expands top `beam_width` candidates to depth `horizon-1`
- **Reward component logging**: Captures per-sample values from reward model (mass_gap, energy_gap, etc.)

---

#### 2. Reward Models (`src/ups/eval/reward_models.py`)

Abstract base class and two concrete implementations for scoring candidate trajectories.

**Base Class**:

```python
class RewardModel(nn.Module):
    """Abstract reward model. Higher scores indicate better candidates."""
    
    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
```

#### 2.1 Analytical Reward Model (ARM)

Physics-inspired scoring based on conservation law violations and positivity.

**Configuration**:

```yaml
reward:
  analytical_weight: 1.0        # Weight in composite model
  grid: [64, 64]                # Evaluation grid resolution
  mass_field: rho               # Field name for mass density
  momentum_field: []            # Field names for momentum (optional)
  energy_field: e               # Field name for energy density
  weights:
    mass: 1.0                   # Mass conservation weight
    momentum: 0.0               # Momentum conservation weight
    energy: 1.0                 # Energy conservation weight
    penalty_negative: 0.5       # Penalty for negative values
```

**Implementation**:

```python
class AnalyticalRewardModel(RewardModel):
    def __init__(
        self,
        decoder: AnyPointDecoder,
        grid_shape: Sequence[int],
        weights: AnalyticalRewardWeights,
        mass_field: Optional[str] = None,
        momentum_field: Optional[Sequence[str]] = None,
        energy_field: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
```

**Scoring Logic**:

1. Decode latent states to physical space using `AnyPointDecoder`
2. Compute reward penalties:
   - **Mass gap**: `|sum(rho_next) - sum(rho_prev)|`
   - **Energy gap**: `|sum(e_next^2) - sum(e_prev^2)|`
   - **Momentum gap**: Per-component gap for each momentum field
   - **Negativity penalty**: `sum(max(-rho, 0))` for negative density
3. Total reward = -(sum of weighted penalties)
4. Store component breakdown in `self.last_components` for logging

**Why Minimal Improvement**:

- ARM assumes conservation laws hold in training data; if data violates conservation, ARM may over-penalize good solutions
- Burgers1D is advection-dominated with no true conservation constraints
- 64x64 evaluation grid loses fine-grained spatial information from 32-token latent
- All penalties are instantaneous; doesn't capture future stability

#### 2.2 Feature Critic Reward Model

Learned scalar critic operating on physics features extracted from latents.

```python
class FeatureCriticRewardModel(RewardModel):
    def __init__(
        self,
        decoder: AnyPointDecoder,
        grid_shape: Sequence[int],
        mass_field: Optional[str] = None,
        momentum_field: Optional[Iterable[str]] = None,
        energy_field: Optional[str] = None,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
```

**Feature Vector**: [mass_sum, negativity_penalty, momentum1, momentum2, energy]

**Network**: 2-layer MLP with SiLU activation

**Status**: Not yet trained in current pipeline; critic weight = 0.0 in golden config.

#### 2.3 Composite Reward Model

Weighted ensemble of multiple reward models.

```python
class CompositeRewardModel(RewardModel):
    def __init__(self, models: Sequence[RewardModel], weights: Optional[Sequence[float]] = None) -> None:
```

Computes weighted average of sub-model scores.

---

#### 3. Reward Model Builder

Instantiates reward models from YAML configuration.

```python
def build_reward_model_from_config(
    ttc_cfg: Dict[str, Any],
    latent_dim: int,
    device: torch.device,
) -> RewardModel:
```

Returns:
- `AnalyticalRewardModel` if `analytical_weight != 0`
- `FeatureCriticRewardModel` if critic configured and loaded
- `CompositeRewardModel` if multiple models active
- Raises error if no models configured

---

## Integration Points

### Training Pipeline

TTC evaluation is **not** used during training, only at inference/evaluation.

### Evaluation Pipeline (`scripts/evaluate.py`)

1. Load operator and optional diffusion corrector
2. Load and instantiate reward model from config (if `ttc.enabled: true`)
3. Create `TTCConfig` object with all sampler/reward parameters
4. Call `evaluate_latent_operator()` with `ttc_config` and `reward_model`
5. Collect per-step logs and final metrics

**Key Parameters**:

```python
ttc_runtime_cfg = TTCConfig(
    steps=ttc_cfg.get("steps", 1),                    # Usually 1 for single-step
    dt=ttc_cfg.get("dt", ...),
    candidates=ttc_cfg.get("candidates", 4),          # 4-24 in practice
    beam_width=ttc_cfg.get("beam_width", 1),          # 1-8
    horizon=ttc_cfg.get("horizon", 1),                # 1-2
    tau_range=(tau_range[0], tau_range[1]),           # e.g., (0.15, 0.85)
    noise_std=float(sampler_cfg.get("noise_std", 0.0)), # 0.015-0.05
    noise_schedule=noise_schedule,                     # [0.08, 0.05, 0.02]
    residual_threshold=ttc_cfg.get("residual_threshold"),
    max_evaluations=ttc_cfg.get("max_evaluations"),   # e.g., 200
    early_stop_margin=ttc_cfg.get("early_stop_margin"),
    gamma=float(ttc_cfg.get("gamma", 1.0)),
    device=device,
)
```

### Evaluation Runner (`src/ups/eval/pdebench_runner.py`)

For each batch in validation set:

```python
if ttc_config is not None and reward_model is not None:
    rollout_log, step_logs = ttc_rollout(
        initial_state=state,
        operator=operator,
        reward_model=reward_model,
        config=ttc_cfg,
        corrector=diffusion,
    )
    pred = rollout_log.states[-1].z
else:
    # Standard forward pass
    predicted_state = operator(state, dt_tensor)
    pred = predicted_state.z
    if diffusion is not None:
        drift = diffusion(predicted_state, tau_tensor)
        pred = pred + drift
```

---

## Configuration

### Golden Config Example

From `configs/train_burgers_golden.yaml`:

```yaml
ttc:
  enabled: true
  steps: 1                              # Single-step rollout
  candidates: 16                        # 16 candidates per step
  beam_width: 5                         # Expand top-5 for lookahead
  horizon: 1                            # No actual lookahead (horizon=1 means no recursion)
  residual_threshold: 0.35
  gamma: 1.0
  max_evaluations: 200

  sampler:
    tau_range: [0.15, 0.85]             # Diffusion strength range
    noise_std: 0.05                     # Latent noise std dev
    noise_schedule: [0.08, 0.05, 0.02]  # Per-step schedule (unused if steps=1)

  reward:
    analytical_weight: 1.0              # Only use analytical model
    grid: [64, 64]                      # Evaluation grid
    mass_field: rho
    energy_field: e
    momentum_field: []
    
    weights:
      mass: 1.0
      energy: 1.0
      penalty_negative: 0.5
    
    critic:
      weight: 0.0                       # Critic disabled
      
  decoder:
    latent_dim: 16
    query_dim: 2
    hidden_dim: 96
    mlp_hidden_dim: 128
    num_layers: 3
    num_heads: 4
    frequencies: [1.0, 2.0, 4.0, 8.0]
    output_channels:
      rho: 1
      e: 1
```

### Key Configuration Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `candidates` | 4 | 4-24 | More candidates = slower but better search |
| `beam_width` | 1 | 1-8 | Lookahead breadth; 1 = no lookahead |
| `horizon` | 1 | 1-2 | Lookahead depth; 1 = greedy only |
| `noise_std` | 0.0 | 0.01-0.1 | Latent noise; higher = more diversity |
| `tau_range` | (0.3, 0.7) | (0.1, 0.9) | Diffusion correction strength |
| `max_evaluations` | None | 50-300 | Hard budget cap; None = unlimited |
| `mass` weight | 1.0 | 0.1-2.0 | Mass conservation penalty weight |
| `energy` weight | 0.0 | 0.0-1.0 | Energy conservation penalty weight |

---

## Candidate Generation Mechanism

### Sources of Stochasticity

At each step, TTC generates diverse candidates by:

1. **Diffusion Correction** (if corrector available):
   - Sample tau ~ Uniform[tau_range[0], tau_range[1]]
   - Compute drift = corrector(state, tau)
   - Candidate_i.z = base.z + drift

2. **Latent Noise**:
   - If noise_std > 0: Add Gaussian noise ~ N(0, noise_std)
   - If noise_schedule provided: Use schedule[step] instead

3. **Base Prediction**:
   - Base = operator(state, dt) is deterministic
   - All diversity comes from corrections + noise

### Why Limited Diversity

For Burgers1D:
- Diffusion corrector has limited capacity (small latent_dim=16)
- Noise perturbations in latent space may not translate to meaningful physical differences
- Small time step (dt=0.1) limits range of plausible solutions

---

## Beam Search and Lookahead

### Design

```python
def lookahead(candidate_state: LatentState, depth: int, step_idx: int) -> float:
    if depth <= 0:
        return 0.0
    next_base = operator(candidate_state, dt_tensor)
    candidates, rewards = sample_candidates(candidate_state, next_base, step_idx)
    if not candidates:
        return 0.0
    totals = rewards[:]
    if depth > 1:
        order = sorted(range(len(rewards)), key=lambda idx: rewards[idx], reverse=True)
        top = order[: max(config.beam_width, 1)]
        for idx in top:
            totals[idx] += config.gamma * lookahead(candidates[idx], depth - 1, step_idx + 1)
    return max(totals)
```

### Algorithm

1. Generate candidates at current state
2. Score each candidate
3. If depth > 1:
   - Select top `beam_width` candidates by immediate reward
   - Recursively lookahead from each top candidate
   - Add discounted future reward: `gamma * lookahead_value`
4. Return max total reward

### Complexity

- **Greedy (beam_width=1, horizon=1)**: 1 forward pass per candidate
- **Lookahead (beam_width=B, horizon=H)**: ~B^H forward passes (exponential)

Example: 16 candidates × beam_width=5 × horizon=2 = 80 evaluations per step

### Current Usage

- Golden config: `beam_width=5`, `horizon=1`
- **Result**: horizon=1 means no actual lookahead recursion!
- Only greedy selection happens

---

## Observed Performance

### Burgers1D Results

From `DIFFUSION_ABLATION_RESULTS.md`:

| Config | Baseline NRMSE | TTC NRMSE | Improvement |
|--------|----------------|-----------|-------------|
| Golden | 0.0782 | 0.0782 | 0% |
| Light-diffusion | 0.0651 | 0.0651 | 0.02% |

**Observed Issues**:

1. **Reward model misalignment**: ARM rewards don't correlate with test NRMSE
2. **Diffusion corrections**: tau sampler creates candidates that are not necessarily better
3. **Latent space representation**: 16-dim latent + 32 tokens may not encode enough information
4. **Physics assumptions**: Burgers assumes mass/energy conservation; actual dynamics are advection + diffusion

---

## Step Logging and Diagnostics

### TTCStepLog Structure

```python
TTCStepLog(
    rewards: [r1, r2, ...],           # Immediate reward for each candidate
    totals: [t1, t2, ...],            # Total reward (after lookahead)
    chosen_index: 3,                  # Index of selected candidate
    beam_width: 5,
    horizon: 1,
    noise_std: 0.05,
    reward_components: {              # From AnalyticalRewardModel
        'mass_gap': 0.0042,
        'mass_penalty': 0.0042,
        'energy_gap': 0.0123,
        'energy_penalty': 0.0123,
        'negativity': 0.0001,
        'negativity_penalty': 0.00005,
        'reward_mean': -0.0166,
        'reward_std': 0.0012,
    }
)
```

### Logging to WandB

From `scripts/evaluate.py`:

- Saves `_ttc_step_logs.json` with raw per-step data
- Creates `_ttc_rewards.png` plot showing best vs chosen reward trajectory
- Logs metadata: `eval_ttc_enabled=true`

### HTML Report

Tables in evaluation HTML show:
- Step index
- Chosen candidate index
- Best total reward (max across all candidates)
- Chosen total reward (selected candidate)

---

## Testing

### Unit Tests (`tests/unit/test_ttc.py`)

Key test cases:

1. **Analytical Reward Basics**:
   - Mass gap correctly computed
   - Penalties are negative rewards

2. **TTC Rollout Selection**:
   - Best candidate (highest reward) is actually chosen
   - Noise schedule properly applied

3. **Lookahead with Beam Search**:
   - Future reward influences present choice
   - Top-k selection works correctly

4. **Composite Rewards**:
   - Weighted average of sub-models computed correctly

### Integration Tests

TTC integration tested through end-to-end evaluation pipeline with synthetic configs.

---

## Why TTC Improvement is Minimal

### Fundamental Issues

1. **Physics Mismatch**:
   - ARM assumes mass/energy conservation
   - Burgers1D is inherently dissipative (advection + diffusion)
   - Penalizing conservation violations doesn't help on advection problems

2. **Reward-Metric Misalignment**:
   - ARM rewards don't correlate strongly with NRMSE
   - High ARM score ≠ low test error
   - Diffusion corrector may improve one metric but not NRMSE

3. **Limited Diversity**:
   - Diffusion tau scaling (0.15-0.85) produces similar candidates
   - Latent noise in 16-dim space may collapse to similar physical predictions
   - Operator deterministic; all diversity artificial

4. **Temporal Mismatch**:
   - TTC evaluates single next step
   - Long-horizon errors (accumulation) not captured
   - Greedy may fail on problems needing foresight

5. **Latent Bottleneck**:
   - 16-dim latent × 32 tokens = modest representational capacity
   - Decoder must map 512-dim latents → 64×64 physical grid
   - Fine-grained spatial information lost

---

## Future Improvements (Not Implemented)

### Process Reward Model (PRM)

From design doc (`docs/ttc_prm_arm_integration_plan.md`):

- Train learned critic on triplet loss (best vs median vs worst)
- Would replace ARM with learned model
- Requires pre-generation of training triplets
- Status: **Not implemented**

### Longer Lookahead

- Increase `horizon` to 2+ for true multi-step planning
- Requires more GPU memory and compute
- May improve long-horizon stability

### Multi-Field Physics Constraints

- Add momentum and boundary condition constraints
- Customize for specific PDEs (Navier-Stokes, compressible flow)

### Adaptive Beam Width

- Reduce beam if all candidates have similar rewards
- Increase if reward variance high
- Budget-aware adaptation

---

## Summary

The TTC system in UPS is a fully functional greedy candidate selection framework with:

- **Analytical Reward Model**: Physics-based scoring using conservation laws
- **Flexible Candidate Generation**: Via diffusion correction + latent noise
- **Optional Lookahead**: Beam search to limited depth
- **Comprehensive Logging**: Step-by-step reward breakdown

**Current Limitations**:
- Minimal empirical improvement on Burgers1D (~0-2%)
- Physics assumptions (conservation) don't match problem (advection)
- Reward model doesn't correlate with downstream metric (NRMSE)
- Latent representation too constrained for large diversity

**Status**:
- Code complete and tested
- Ready for deployment
- Recommendations for improvement documented but not implemented

---

## Code Locations

| Component | File |
|-----------|------|
| TTC Rollout | `src/ups/inference/rollout_ttc.py` |
| Reward Models | `src/ups/eval/reward_models.py` |
| PDEBench Integration | `src/ups/eval/pdebench_runner.py` |
| Evaluation Script | `scripts/evaluate.py` |
| Unit Tests | `tests/unit/test_ttc.py` |
| Config Example | `configs/train_burgers_golden.yaml` |
| Design Document | `docs/ttc_prm_arm_integration_plan.md` |
| Performance Notes | `DIFFUSION_ABLATION_RESULTS.md` |

