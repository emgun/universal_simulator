# Using TTC During Training: Feasibility Analysis

**Question**: Would adding TTC to training improve the model?

**Short Answer**: **Theoretically yes, but impractical with current architecture.** Would require significant changes and 16-24× training cost increase.

---

## Current Training Pipeline (Supervised Learning)

### How Training Works Now

**Data**: Pre-computed latent pairs `(state_t, state_{t+1})` from PDEBench

**Training Loop**:
```python
for state_t, state_tp1 in data_loader:
    # Single forward pass
    predicted = operator(state_t, dt)

    # Simple supervised loss
    loss = MSE(predicted.z, state_tp1.z)

    # Backward pass
    loss.backward()
    optimizer.step()
```

**Key Properties**:
- ✅ **Fast**: Single forward pass per sample
- ✅ **Simple**: Direct MSE supervision
- ✅ **Scalable**: Batch processing
- ✅ **Deterministic**: Fixed ground truth targets

**Training Time**: ~25 minutes for 25 epochs on A100

---

## What TTC During Training Would Require

### Option 1: TTC-Augmented Targets (Hard Example Mining)

**Idea**: Use TTC to select harder training examples.

**Implementation**:
```python
for state_t, state_tp1_gt in data_loader:
    # Generate multiple predictions
    candidates = []
    for _ in range(N):  # N = 16 candidates
        pred = operator_stochastic(state_t)
        candidates.append(pred)

    # Score with reward model
    rewards = [reward_model(state_t, cand) for cand in candidates]

    # Select candidate with WORST reward (hardest example)
    worst_idx = argmin(rewards)
    hard_target = candidates[worst_idx]

    # Train to match GROUND TRUTH, not hard example
    loss = MSE(predicted.z, state_tp1_gt.z)
    loss.backward()
```

**Pros**:
- Uses TTC reward signal to identify hard regions
- Still trains on ground truth (stable)

**Cons**:
- 16× more forward passes (candidate generation)
- Reward model evaluation overhead
- Requires stochastic operator (need dropout/noise)

**Training Time**: ~6-7 hours (16× slower)

---

### Option 2: TTC-Guided Loss Weighting

**Idea**: Weight training loss by reward model scores.

**Implementation**:
```python
for state_t, state_tp1_gt in data_loader:
    # Standard prediction
    predicted = operator(state_t, dt)

    # Score prediction quality
    reward = reward_model(state_t, predicted)

    # Weight loss inversely to reward (focus on bad predictions)
    weight = 1.0 / (reward + epsilon)
    loss = weight * MSE(predicted.z, state_tp1_gt.z)

    loss.backward()
```

**Pros**:
- Minimal overhead (1 reward eval per sample)
- Focuses training on physics-violating predictions
- No architecture changes needed

**Cons**:
- ARM may give misleading weights (conservation ≠ accuracy)
- Training instability if weights vary too much
- Requires well-calibrated reward model

**Training Time**: ~30 minutes (~20% slower)

**Verdict**: Worth trying! Low cost, potentially helpful.

---

### Option 3: Rollout-Based Training (Multi-Step)

**Idea**: Train on multi-step rollouts, not single steps.

**Current**: Predict 1 step ahead
**Proposed**: Predict T steps ahead with TTC selection

**Implementation**:
```python
for initial_state, target_traj in data_loader:
    # Multi-step rollout with TTC
    predicted_traj = ttc_rollout(
        initial_state,
        operator,
        reward_model,
        steps=T
    )

    # Loss on final state
    loss = MSE(predicted_traj[-1].z, target_traj[-1].z)

    # Or loss on full trajectory
    loss = MSE_trajectory(predicted_traj, target_traj)

    loss.backward()
```

**Pros**:
- Learns long-horizon stability directly
- Naturally integrates TTC into training
- Addresses accumulation of errors

**Cons**:
- **Extremely expensive**: T × N candidate evaluations
  - T=10 steps, N=16 candidates = 160× cost
- Gradient vanishing through long rollouts
- Requires backprop through TTC selection (non-differentiable!)
- Training instability

**Training Time**: ~60+ hours (100×+ slower)

**Verdict**: Not feasible with current resources.

---

### Option 4: Offline Rollout Filtering (Dataset Curation)

**Idea**: Pre-filter training data using TTC offline.

**Pipeline**:
```bash
# 1. Train baseline operator (25 min)
python scripts/train.py --stage operator

# 2. Generate TTC rollouts on training set (2-3 hours)
python scripts/generate_ttc_rollouts.py \
  --input data/train \
  --output data/train_ttc_filtered

# 3. Filter to only good trajectories (high reward)
python scripts/filter_by_reward.py \
  --threshold 0.9 \
  --keep-fraction 0.8

# 4. Re-train on filtered data (25 min)
python scripts/train.py --stage operator --data data/train_ttc_filtered
```

**Pros**:
- One-time cost (preprocessing)
- No changes to training loop
- Removes "bad" training examples
- Can iterate (train → filter → retrain)

**Cons**:
- Only helps if bad data is the problem
- May reduce dataset size significantly
- Requires well-calibrated reward model

**Training Time**: Same (25 min), but 2-3 hour preprocessing

**Verdict**: Interesting for iteration, but requires ARM to work first.

---

## Reinforcement Learning Perspective

TTC is essentially **tree search at inference time**, similar to:
- **AlphaGo**: MCTS at test time, policy network trained on self-play
- **MuZero**: Learned model + planning at test time

### RL-Style TTC Training

**Approach**: Train reward model and operator jointly.

**Algorithm**:
```python
# Phase 1: Train operator on supervised data (standard)
train_operator_supervised(operator, data)

# Phase 2: Train reward model on operator predictions
train_reward_model(reward_model, operator, data)

# Phase 3: Improve operator using reward model feedback
for epoch in range(fine_tune_epochs):
    for state_t, state_tp1_gt in data_loader:
        # Generate candidates
        candidates = generate_candidates(operator, state_t, N=16)

        # Score with reward model
        rewards = [reward_model(state_t, c) for c in candidates]

        # Select best candidate
        best_idx = argmax(rewards)
        best_candidate = candidates[best_idx]

        # Train operator to produce high-reward predictions
        # Using policy gradient or direct supervision
        loss = MSE(operator(state_t).z, best_candidate.z)
        loss.backward()
```

**Pros**:
- Iteratively improves operator and reward model
- Operator learns to produce high-reward trajectories
- Similar to successful RL approaches (AlphaGo, etc.)

**Cons**:
- Requires differentiable or policy-gradient training
- Risk of reward hacking (operator exploits ARM bugs)
- Much more complex than current pipeline
- Needs many iterations to converge

**Training Time**: 10-20× slower, many iterations

**Verdict**: Research project, not production ready.

---

## Practical Recommendation

### What You Should Try (Low Cost)

**Option 2: TTC-Guided Loss Weighting**

1. **Modify training loop** in `src/ups/training/loop_train.py`:
   ```python
   # After prediction
   predicted = self.operator(state, dt_tensor)

   # Score prediction (if reward model available)
   if self.reward_model is not None:
       reward = self.reward_model.score(state, predicted)
       # Higher reward = lower loss weight (it's already good)
       # Lower reward = higher loss weight (needs more training)
       weight = torch.exp(-reward)  # Weight inversely to reward
   else:
       weight = 1.0

   # Weighted loss
   loss = weight * mse_loss(predicted.z, target.z)
   ```

2. **Add reward model to training config**:
   ```yaml
   training:
     use_reward_weighting: true
     reward_model:
       type: analytical
       weights:
         mass: 0.0  # Disabled for Burgers
         energy: 0.0
         penalty_negative: 1.0
   ```

3. **Test on short training run** (5 epochs):
   ```bash
   python scripts/train.py \
     --config configs/train_burgers_reward_weighted.yaml \
     --stage operator \
     --epochs 5
   ```

**Expected Cost**: +20% training time
**Expected Benefit**: 2-5% NRMSE improvement (if ARM correlates with quality)
**Risk**: Low (can easily disable if it hurts)

---

### What You Should NOT Try (High Cost)

**❌ Option 3: Rollout-Based Training**
- Too expensive (100× training time)
- Not proven necessary
- Complex implementation

**❌ Option 5: Full RL Pipeline**
- Research project (weeks/months)
- Requires extensive validation
- High risk of failure

---

## Key Insights

### 1. TTC is Designed for Inference, Not Training

From the TTC paper (arXiv:2509.02846):
> "Test-time computing enables improved predictions **without retraining** by leveraging physics-based reward models during inference."

The whole point is to **avoid retraining**.

### 2. Current Training is Already Good

From ablation study results:
- **Baseline NRMSE**: 0.0651 (light-diffusion)
- **Training is not the bottleneck**; inference is

TTC aims to fix inference issues, not training issues.

### 3. The Reward Model is the Blocker

If ARM doesn't correlate with NRMSE:
- Using ARM during training will **hurt** performance
- Training will optimize for wrong objective
- Must fix ARM first (see `ARM_CRITICAL_ISSUES_ANALYSIS.md`)

### 4. Cost-Benefit is Poor

| Approach | Training Time | Implementation | Expected Gain |
|----------|---------------|----------------|---------------|
| **Current** | 25 min | ✅ Done | Baseline |
| **Loss Weighting** | 30 min | 🟡 Easy | +2-5% |
| **Hard Example Mining** | 6-7 hours | 🟠 Medium | +5-10% |
| **Rollout-Based** | 60+ hours | 🔴 Hard | +10-20% |
| **RL Fine-Tuning** | Days/weeks | 🔴 Very Hard | +20-30% |

For 5-10% gain, you'd spend 24× more training time.
**Better to spend that time fixing ARM** for 88% TTC gain at inference.

---

## Conclusion

### Direct Answer to Your Question

**Q**: Would adding TTC to training improve the model?

**A**:
- **Theoretically**: Yes, if reward model is good and multi-step training helps
- **Practically**: No, because:
  1. ARM doesn't correlate with NRMSE yet (must fix first)
  2. Cost is too high (6-100× training time)
  3. Current single-step training is already effective
  4. TTC designed for inference, not training

### Recommended Path Forward

**Phase 1** (This week):
1. ✅ Fix ARM (disable conservation penalties)
2. ✅ Test fixed ARM with evaluation
3. ✅ Measure TTC improvement (target: >5%)

**Phase 2** (If ARM works):
- **Option A**: Use TTC at inference only (original design)
- **Option B**: Try loss weighting (+20% training time, low risk)

**Phase 3** (If loss weighting helps):
- Consider hard example mining
- Requires stochastic operator (add dropout/noise)
- Incremental benefit (~5%)

**Phase 4** (Research):
- Investigate rollout-based training
- Requires significant resources
- Publish findings if successful

### Bottom Line

**Don't add TTC to training yet.** Fix ARM first, measure inference TTC gains, then decide if training integration is worth the cost.

The 88% improvement from TTC at inference (per paper) >> 5-10% from training integration.

---

## Related Work

### Papers Using Test-Time Optimization

1. **AlphaGo** (Silver et al., 2016)
   - MCTS at inference, policy network trained offline
   - Test-time search ≠ training-time search

2. **MuZero** (Schrittwieser et al., 2020)
   - Learned model + MCTS at inference
   - Training uses self-play, but not full tree search

3. **Test-Time Training** (Sun et al., 2020)
   - Adapts model at test time using self-supervised objectives
   - Different from TTC (which uses reward-guided search)

4. **Hindsight Experience Replay** (Andrychowicz et al., 2017)
   - Relabels failed RL trajectories as successes for other goals
   - Similar to using TTC rollouts for training data augmentation

### Key Difference

These methods use **test-time computation for better predictions**, but:
- Training is still efficient (no tree search in training loop)
- Test-time search guides exploration, not gradient updates
- Our case: TTC for PDE simulation, not RL policy learning

---

## Appendix: Implementation Sketch for Loss Weighting

If you want to try the low-cost option:

```python
# src/ups/training/reward_weighted_loss.py

import torch
from typing import Optional

from ups.eval.reward_models import RewardModel

class RewardWeightedLoss:
    def __init__(
        self,
        base_loss_fn,
        reward_model: Optional[RewardModel] = None,
        temperature: float = 1.0,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        self.base_loss_fn = base_loss_fn
        self.reward_model = reward_model
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight

    def __call__(self, predicted, target, state):
        # Compute base loss
        loss = self.base_loss_fn(predicted.z, target.z)

        # If no reward model, return unweighted loss
        if self.reward_model is None:
            return loss

        # Score prediction quality
        with torch.no_grad():
            reward = self.reward_model.score(state, predicted)

        # Convert reward to weight
        # High reward = low weight (already good, less training needed)
        # Low reward = high weight (bad prediction, more training needed)
        weight = torch.exp(-reward / self.temperature)
        weight = torch.clamp(weight, self.min_weight, self.max_weight)

        # Weighted loss
        return weight * loss
```

**Usage**:
```python
# In training loop
loss_fn = RewardWeightedLoss(
    base_loss_fn=nn.MSELoss(),
    reward_model=reward_model,
    temperature=1.0,
)

loss = loss_fn(predicted, target, state)
loss.backward()
```

**Test in 5 min**: Just add to one epoch and check if loss decreases faster.
