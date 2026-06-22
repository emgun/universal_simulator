# External Baseline Mapping

This note summarizes the baseline records in
`docs/claim_evidence/external_baseline_mapping.json`.

## Protocol

The current matched comparison uses the `light-v1` PDEBench-shaped task set,
train/validation/test split boundaries, 32-sample caps, a 16-step decoded
rollout horizon, `decoded_rollout_nrmse`, a held-out ledger, and recorded
artifact hashes.

The comparison is useful because it asks a narrow question: if another model is
trained with the same data budget and evaluated with the same metric, how does
UPS compare?

## UPS Variants

The primary UPS row is `ups_light_shared_context_transport_guarded` with held-out
`decoded_rollout_nrmse = 0.4165820594268877`.

Two additional held-out UPS variants use the same split, task set, metric, and
ledger discipline, but change the inference contract:

| Surface | Run | Held-out decoded rollout NRMSE | Scope |
| --- | --- | ---: | --- |
| Primary UPS | `ups_light_shared_context_transport_guarded` | `0.4165820594268877` | Frozen CT8 shared-context protocol |
| CT1 online context | `ups_light_advection_context_transport_only_ct1_guarded` | `0.20177292896682064` | Separate online transport-context variant |
| Data-conditioned context phase | `ups_light_advection_data_conditioned_context_phase_guarded` | `0.1808155304023394` | Separate data-conditioned context-phase variant |

The scoped variants should be shown beside the primary result, not merged into a
single aggregate table.

## Measured External Baselines

The current external rows are reruns or official-source adapters measured under
the same `light-v1` setup:

| Surface | Held-out decoded rollout NRMSE | Note |
| --- | ---: | --- |
| NeuralOperator FNO | `0.6391747076887233` | Canonical FNO family via NeuralOperator |
| NeuralOperator UNO | `0.5560551396226746` | Second NeuralOperator architecture family |
| PDEBench U-Net | `0.6095843876848097` | Official PDEBench U-Net architecture adapter |
| CNO1d | `0.5918753212407414` | Official simplified CNO1d adapter for height-1 light-v1 grids |

Published paper tables are not mixed into these numbers unless the protocol
differences are mapped or rerun.

## Transfer And Ecosystem Rows

Poseidon is tracked as a validation-only foundation-transfer path. The scalar
adapter finetune improved over zero-shot but did not clear the validation gate,
so there is no held-out Poseidon transfer row.

PhysicsNeMo has a validation-only FNO recipe-adapter metric repeated under Torch
2.10. PDEArena and RealPDEBench remain future protocol surfaces.

The generated tables and cards in `docs/results/generated/` are derived from
this mapping so the public figures stay tied to source records.
