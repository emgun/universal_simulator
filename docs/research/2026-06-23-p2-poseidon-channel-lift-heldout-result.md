# P2 Poseidon Channel-Lift Held-Out Result

Date: 2026-06-23

Status: completed, negative/mixed transfer. This is research evidence for an
adapted external challenger, not claim evidence and not public-language support.

## Contract

Pre-test contract:
`docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-pretest-contract.json`.

Evidence manifest:
`docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-evidence-manifest.json`.

Registered measurement key:
`b487e8841f7631554248fcaeedf9dd3a1fba1faa7f003f0e6304a2b96375516a`.

The run used the exact bounded path: train split `train`, evaluation split
`test`, `channel_lift`, three `light-v1` tasks, 32 train/eval samples, 16
decoded rollout steps, 30 epochs, seed `17`, no repeat flag, and a held-out
ledger. Vast instance `42254247` was destroyed after completion.

## Evidence

- Local summary:
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4/summary.json`.
- Local ledger:
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4/test_ledger.json`.
- B2 artifact:
  `b2://pdebench/remote-runs/poseidon-channel-lift/poseidon_channel_lift_light-v1_20260623T191655Z.tar.gz`.
- Summary SHA256:
  `9aeb5cf9d5923ca5fb75c6d1b6b725213566df5c74b5e4a8b36889d5dbc8d895`.
- Ledger SHA256:
  `4a4385201387d9a07fbe8d4ea9e654ba584401b7a2ae03dd2144afe8f7450d65`.

## Result

Held-out aggregate decoded rollout NRMSE was `0.5551415687535287`, worse than
the positive-transfer threshold `0.4165820594268877`.

Task metrics:

- `task_advection1d_decoded_rollout_nrmse = 0.7840223655431167`
- `task_burgers1d_decoded_rollout_nrmse = 0.18316455707528173`
- `task_darcy2d_decoded_rollout_nrmse = 0.21459086990463278`

Integrity checks:

- `held_out_test_used = true`
- `held_out_test_data_read = true`
- `claim_comparable = false`
- `published_numbers_directly_comparable = false`
- `adapter_mode = channel_lift`
- `trainable_parameter_count = 13`
- Poseidon source commit:
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`
- Checkpoint SHA256:
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`
- `embedding_recovery_replaced = false`
- `pretrained_embedding_recovery_intact = true`
- Ledger recorded exactly one measurement for the registered key.

## Decision

Do not promote Poseidon `channel_lift` Option A. Do not rerun this held-out key.
The validation win did not transfer to held-out, and the miss is dominated by
transport/advection. Burgers and Darcy remained reasonable, so the failure is
not a generic runtime or ledger issue; it is the known transport phase/general-
ization gap returning under the external backbone.

Next best path: run a no-held-out branch check between DPOT, a transport-aware
Poseidon adapter/backbone modification, and local transport-sidecar lessons.
Any future held-out pretest should require a stricter validation gate that
explicitly covers advection/transport, not just aggregate validation NRMSE.
