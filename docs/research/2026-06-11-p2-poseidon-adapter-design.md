# P2.1 Backbone Transplant Adapter Design: Poseidon ScOT (primary), DPOT (probe)

Date: 2026-06-11

Status: design document for north-star roadmap Phase 2 (P2.1). No measurements, no GPU use, no claim evidence. Implements the roadmap requirement: adapter design with parameter counts, frozen/trainable split, and provenance/hash plan, explicitly avoiding the failed scalar-layer-replacement path.

## Why the previous attempt failed (grounded in committed evidence)

`docs/claim_evidence/artifacts/poseidon_scot_val_light_v1.tar.gz` records the exact contract of the stopped attempt (`poseidon_scot_scalar_ft_val_light_v1`, validation `0.5453508470039229`, above the 0.5 stop threshold):

- Poseidon-T original config: `num_channels=4`, `num_out_channels=4`, `image_size=128`, `embed_dim=48`, `window_size=16`, `use_conditioning=True` (lead-time conditioned layer norms), `learn_residual=False`.
- The attempt set `num_channels=1`/`num_out_channels=1` and **replaced** `embeddings.patch_embeddings.projection`, `patch_recovery.projection`, and `patch_recovery.mixup` with freshly initialized scalar layers (`embedding_recovery_replaced=True`), then trained only those.

That throws away the pretrained input/output interface: the backbone's features were learned through the original 4-channel patch embedding, and a from-scratch projection must rediscover that mapping from 32-sample light-v1 data. The fix is to **keep the pretrained embedding and recovery frozen and intact**, and adapt on the outside of them.

## Design principle

Adapt in *channel space at native resolution*, before the frozen patch embedding and after the frozen patch recovery:

```
scalar field u (H x W)
  -> lift: 1 -> 4 channels        (trainable, initialized to replicate)
  -> frozen pretrained patch embedding -> frozen ScOT backbone -> frozen patch recovery
  -> readout: 4 -> 1 channels     (trainable, initialized to mean)
  -> scalar prediction
```

The pretrained model sees inputs with the channel structure it was trained on; the adapter only learns how to express a scalar field in that structure.

## Option A (primary): linear channel lift/readout

- Lift: 1x1 conv `1 -> 4`, weights initialized to 1 (broadcast replicate), bias 0. 8 params.
- Readout: 1x1 conv `4 -> 1`, initialized to channel mean. 5 params.
- Optional task modulation: per-task channel gains/biases from the 3-task vocabulary, `3 x (4+4) = 24` params, initialized to identity, so tasks can occupy different channel mixes.
- Trainable total: **~13-37 params** (plus nothing else). Backbone (Poseidon-T, ~tens of M params) fully frozen, hash-verified (`model.safetensors` SHA256 `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`, already recorded in claim evidence).
- At initialization the model computes exactly the pretrained operator on a replicated scalar field — training can only improve from a meaningful starting point, unlike the random-projection start of the failed attempt.

## Option B (if A plateaus): shallow nonlinear lift/readout

- Lift: 3x3 conv `1 -> 16`, GELU, 1x1 `16 -> 4`, residual-added to the replicated broadcast so initialization still reduces to Option A. ~230 params.
- Readout: mirror structure, residual around the channel mean. ~230 params.
- Trainable total: **~500 params**. Same frozen backbone.

## Option C (roadmap Gate 2, only after A/B clears Gate 1 continue-zone): LoRA unfreeze

- Keep A or B adapter; add rank-4 LoRA on attention qkv/proj across the hierarchical stages (embed_dim 48 doubling per stage). Estimated **30-80K trainable params**.
- Enters only under the existing Gate 2 contract: same train/validation discipline, compared against the Gate 1 validation artifact.

## Shared contract (all options)

- Data and metric: light-v1 protocol exactly as the external-baseline scripts already implement (`scripts/run_external_poseidon_scot_finetune.py` machinery for pairs/pixels), train split only for fitting, validation only for selection. 1D tasks map to 128x128 via the existing square-pixel broadcast (recorded roundtrip distortion `0.0023447850529950184`).
- Lead-time conditioning: pass the protocol's actual normalized lead time per rollout step (the prior scripts expose `time_value`); do not leave it at a constant default — ScOT's conditioned layer norms are a pretrained capability we should use, and 16-step decoded rollout evaluation must condition each step correctly.
- Rollout evaluation: autoregressive in scalar space (readout -> lift between steps), 16 steps, `decoded_rollout_nrmse`, per-task/horizon metrics, no estimators.
- Training objective: one-step MSE plus a cheap 4-step rollout term (the P1 sweeps showed one-step-only training is the failure mode; the term is cheap at these adapter sizes).
- Optimizer: AdamW, lr 1e-2 for Option A (tiny parameter set), 1e-3 for B, cosine decay, <= 30 epochs, patience 5. These are starting points, tunable on validation only.
- Provenance per run: HF model id + safetensors SHA256, adapter parameter names and counts, resolved config, command, splits, artifact hashes — the existing external-baseline evidence schema already covers this.

## Gates (unchanged from the roadmap)

- **G2a (Gate 1)**: validation `decoded_rollout_nrmse <= 0.363424243629033` (best external validation baseline, UNO) with no task collapsing near 1.0 -> eligible for a held-out pre-test contract.
- Continue-zone `0.3634-0.5`: justifies Option B and then Option C (Gate 2).
- Stop: above 0.5 after a clean Option A+B run -> the Poseidon path stops again, and the DPOT probe becomes primary.

## DPOT probe (P2.4, parallel and cheap)

DPOT checkpoints (Hugging Face) are autoregressive-denoising pretrained across PDE families with channel-flexible input handling. The probe re-uses the same lift/readout principle where channel counts differ. One fine-tune run under the identical light-v1 contract; compare on validation; pick the primary backbone. Budgeted at <= 2 GPU-hours.

## Execution checklist (P2.2 next)

- [x] Extend `scripts/run_external_poseidon_scot_finetune.py` with `--adapter-mode channel_lift` implementing Option A (keep `scalar_layers` for comparison; default to the new mode), including correct per-step lead-time conditioning and the 4-step rollout loss term.
- [x] Unit tests: replicate-init exactness (lift+readout at init == channel-mean of pretrained 4-channel output), parameter-count assertions, frozen-backbone assertion (no grads outside adapter names). Verified 2026-06-22 with `python -m pytest tests/unit/test_external_poseidon_scot_finetune.py -q`.
- [x] One CPU smoke run at 2 samples to validate the path before GPU spend.
  Verified 2026-06-22 with
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_smoke_val_light_v1/summary.json`:
  validation-only `decoded_rollout_nrmse = 0.31116372295004086`,
  `adapter_mode = channel_lift`, `held_out_test_used = false`,
  `embedding_recovery_replaced = false`, and 13 trainable parameters.
- [x] GPU run on light-v1 train/val (Gate 1 measurement), evidence JSON + validator per the external-baseline schema.
  Verified 2026-06-23 with
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`:
  validation-only aggregate `decoded_rollout_nrmse = 0.35782889238675264`
  clears G2a `<= 0.363424243629033`, `held_out_test_used = false`,
  `adapter_mode = channel_lift`, `embedding_recovery_replaced = false`, source
  commit `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, checkpoint SHA256
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`, and 13
  trainable parameters. Per-task validation NRMSE:
  advection1d `0.4937043430599529`, burgers1d `0.15674926288225416`, darcy2d
  `0.2071060212271272`.
- Next: draft held-out pre-test contract and evidence manifest. Do not run
  held-out test until that contract is explicit.
