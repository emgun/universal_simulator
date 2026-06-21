# External Baseline Mapping

This note explains the distinction enforced by
`docs/claim_evidence/external_baseline_mapping.json`.

## Claim Protocol

The claim protocol is the exact measurement contract for the current UPS claim:
the `light-v1` PDEBench-shaped task set, train/validation/test split boundaries,
sample caps, 16-step decoded rollout horizon, `decoded_rollout_nrmse`, command,
promotion rule, held-out ledger, and artifact bundle.

A fair baseline on the claim protocol is valuable because it answers: if another
model is given the same data budget and judged by the same metric, does UPS win?
The committed `physical_fourier_light_test_strong_baseline` answers that for a
repo-local Fourier neural baseline.

## Primary Claim vs Scoped Variants

The current primary UPS claim remains
`ups_light_shared_context_transport_guarded`: the CT8 shared-context decoded UPS
candidate on held-out `light-v1` test with
`decoded_rollout_nrmse = 0.4165820594268877`.

The CT1 result is recorded separately as
`light_v1_ct1_online_transport_context`, run
`ups_light_advection_context_transport_only_ct1_guarded`. It uses the same
held-out `light-v1` test split, task set, metric, 16-step decoded rollout
horizon, and ledger discipline, and records
`decoded_rollout_nrmse = 0.20177292896682064`. It is not the same exact
inference contract as the CT8 primary claim: CT1 changes the online
roll-persistence correction to `context_transitions = 1`,
`families = [transport]`, and `slope = 1.0`, while CT8 used
`context_transitions = 8`,
`families: [transport, conservation]`, and the shared calibrated slope.

The data-conditioned result is also recorded separately as
`light_v1_data_conditioned_context_phase`, run
`ups_light_advection_data_conditioned_context_phase_guarded`. It uses the same
held-out `light-v1` test split, task set, metric, 16-step decoded rollout
horizon, and ledger discipline, and records
`decoded_rollout_nrmse = 0.1808155304023394`. It is not the same exact
inference contract as CT8 or CT1: it fits a train-only `context_shift`
coefficient and uses one observed context transition at evaluation time before
roll-persistence from horizon 2.

| Surface | CT8 primary claim | CT1 scoped variant | Data-conditioned scoped variant |
| --- | --- | --- | --- |
| Run | `ups_light_shared_context_transport_guarded` | `ups_light_advection_context_transport_only_ct1_guarded` | `ups_light_advection_data_conditioned_context_phase_guarded` |
| Held-out test `decoded_rollout_nrmse` | `0.4165820594268877` | `0.20177292896682064` | `0.1808155304023394` |
| Advection test NRMSE | `0.5765863333379032` | `0.22508631227914033` | `0.18345021264323003` |
| Contract status | Primary frozen `light-v1` claim contract | Separate online transport-context variant | Separate data-conditioned context-phase variant |
| Claimable wording | UPS beats the measured fair/external baselines under the frozen CT8 light-v1 protocol | CT1 improves the held-out online transport-context variant under a changed inference contract | Data-conditioned context-phase improves the held-out scoped variant under an explicitly changed inference contract |
| Not claimable | Published-paper SOTA or published-table equivalence | Replacement for CT8, autonomous rollout claim, external-paper reproduction, or published-table equivalence | Replacement for CT8, no-context/autonomous rollout claim, external-paper reproduction, or published-table equivalence |

The machine-readable copy of this distinction lives in
`scoped_claim_variants` in both
`docs/claim_evidence/universal_sota_claim_evidence.json` and
`docs/claim_evidence/external_baseline_mapping.json`.

## External-Paper Reproduction

An external-paper reproduction is stricter. It means running an outside paper's
official implementation, or a faithful implementation with declared architecture
and hyperparameters, and then mapping the result to this claim protocol. A
published table value is not enough when the split, sample budget, rollout
horizon, metric, or task bundle differs.

The highest-signal next reproduction path is FNO through PDEBench and/or
NeuralOperator because FNO is canonical, PDEBench names it as an official
baseline family, and the current repo already has PDEBench-shaped data plus a
local Fourier sanity baseline.

The executable adapter is
`scripts/run_external_neuraloperator_fno_baseline.py`. It keeps NeuralOperator as
an optional dependency: `--dry-run` validates and writes the contract without
loading data or requiring `neuralop`, while a live run imports
`neuralop.models.FNO`. A live `split=test` run fails closed unless
`--allow-held-out-test-eval` is set, so adapter debugging stays on validation
unless the held-out measurement is deliberate.

The first held-out external FNO measurement is
`external_neuraloperator_fno_test_light_v1_e3`. It uses NeuralOperator `2.0.0`
on the same light-v1 task set, `32/32` sample caps, and 16-step
`decoded_rollout_nrmse` protocol. It records
`decoded_rollout_nrmse = 0.6391747076887233`; the UPS claim candidate is
`0.4165820594268877`, a `34.825%` improvement over this measured external FNO
baseline. This comparison is protocol-comparable, but it is still not a claim
about published FNO paper-table values.

The second held-out external-family measurement is
`external_neuraloperator_uno_test_light_v1_e3`. It uses NeuralOperator `2.0.0`
UNO with the same task set, sample caps, train stride, and 16-step
`decoded_rollout_nrmse` protocol. It records
`decoded_rollout_nrmse = 0.5560551396226746`; the UPS claim candidate is
`0.4165820594268877`, a `25.083%` improvement over this measured external UNO
baseline. This expands the claim from one measured external family to two, but
it still does not make a published-table SOTA claim.

The third held-out external-family measurement is
`external_pdebench_unet_test_light_v1_e3`. It uses the official PDEBench U-Net
architecture adapted from PDEBench commit
`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`, with the same task set, sample caps,
train stride, and 16-step `decoded_rollout_nrmse` protocol. It records
`decoded_rollout_nrmse = 0.6095843876848097`; the UPS claim candidate is
`0.4165820594268877`, a `31.661%` improvement over this measured PDEBench U-Net
baseline. This reduces single-library bias because the baseline no longer comes
from NeuralOperator, but it is still not a published PDEBench checkpoint or
paper-table reproduction.

The fourth held-out external-family measurement is
`external_cno1d_test_light_v1_e3`. It uses the official simplified CNO1d
architecture adapted from ConvolutionalNeuralOperator commit
`6e765198aa02b56352e0a3437104b9d9e337176e`, with the same task set, sample
caps, train stride, and 16-step `decoded_rollout_nrmse` protocol. It records
`decoded_rollout_nrmse = 0.5918753212407414`; the UPS claim candidate is
`0.4165820594268877`, a `29.617%` improvement over this measured CNO1d
baseline. This adds a distinct official neural-operator family, but the scope is
the current height-1 light-v1 adapter, not a published CNO representative-PDE
benchmark table, CNO2d square-grid run, or Poseidon foundation-model transfer.

## Foundation-Transfer Contract

`foundation_transfer_readiness_light_v1` records the next transfer gate without
spending held-out test budget. It inspected only train/validation metadata for
the light-v1 tasks plus official source snapshots: Poseidon commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61` and ConvolutionalNeuralOperator
commit `6e765198aa02b56352e0a3437104b9d9e337176e`.

The contract is intentionally not a claim-comparable measurement. It records
three blockers before any Poseidon/CNO-FM result can be compared: light-v1 has no
direct official Poseidon dataset identifiers, the repo-inferred light-v1 grids
are height-1 rather than uniformly square image tensors, and CNO-FM declares a
5-channel foundation input/4-channel output while the current light-v1 protocol
is scalar. The next best path is a validation-only Poseidon ScOT adapter with
pretrained checkpoint provenance and train/val-only dataset adapter evidence;
CNO-FM should remain a separate 2D/channel-rich transfer track.

`poseidon_transfer_adapter_manifest_light_v1` implements that first Poseidon
adapter gate. It converts repo field steps to Poseidon-style square
`pixel_values` with bilinear resizing and then round-trips back to the repo
flattened metric shape. It inspected only `train` and `val`, used Poseidon
commit `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, and records
`adapter_roundtrip_nrmse = 0.0023447850529950184`. This is adapter distortion,
not `decoded_rollout_nrmse` and not a Poseidon model score. The checkpoint
handle is `camlab-ethz/Poseidon-T`, but the checkpoint hash remains pending, so
the next gate is loading `ScOT.from_pretrained` with hashed weights and recording
validation-only `decoded_rollout_nrmse` before any held-out transfer test spend.

`poseidon_scot_val_light_v1` executes that validation-only checkpoint gate. It
loads `camlab-ethz/Poseidon-T` from Hugging Face snapshot
`ec976ed5d25883ec9db4e486ebbeeefa9e08303b`, verifies
`model.safetensors` SHA256
`e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`, and runs
the full light-v1 validation cap: 32 validation samples, 16 teacher-forced steps,
and all three tasks. It records
`decoded_rollout_nrmse = 0.9999999950370435` with per-task validation NRMSE near
`1.0`. This is a real validation measurement, but not claim-comparable: adapting
the official 4-channel checkpoint to scalar light-v1 newly initializes the input
embedding and recovery layers, so the result mainly proves that direct zero-shot
transfer is not useful under this adapter. The next gate is finetuning those
scalar adapter layers on train and selecting on validation before any held-out
Poseidon transfer test.

`poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1` executes that scalar-layer
finetune gate without using held-out test data. It keeps the Poseidon backbone
frozen, trains only the newly initialized scalar embedding/recovery layers on
the train split, and evaluates the full light-v1 validation cap: 32 validation
samples, 16 teacher-forced steps, and all three tasks. The run records
`decoded_rollout_nrmse = 0.5453508470039229`, with task validation NRMSE
advection `0.6030753349043854`, Burgers `0.49033314173084885`, and Darcy
`0.47892385326272763`. This improves over zero-shot Poseidon but does not clear
the validation gate: it is above the `0.5` scalar-path stop threshold and above
the current best external validation baseline, UNO at `0.363424243629033`.
Therefore the scalar-only Poseidon transfer path should not spend held-out test
budget. If Poseidon remains worth pursuing, the next path is controlled unfreeze
or low-rank adaptation on train/validation only; otherwise the higher-signal
work is UPS-side advection robustness.

## Official/Ecosystem Compatibility

`ecosystem_compatibility` records the public-facing compatibility layer behind
the showcase readiness card. It is separate from the numeric claim table:

- NeuralOperator FNO and UNO, PDEBench U-Net, and CNO1d are measured official or
  official-source architecture adapters under the matched `light-v1` protocol.
- Poseidon ScOT is validation-only transfer evidence and is explicitly stopped
  before held-out test because the scalar-layer finetune did not clear the
  validation gate.
- PDEArena is tracked as a future official external protocol, not as a direct
  `light-v1` claim comparison.
- NVIDIA PhysicsNeMo has a dry recipe-compatibility smoke manifest at
  `docs/claim_evidence/physicsnemo_compatibility_smoke_light_v1.json`. It
  records the package/docs/source links, light-v1 train/validation task
  interface, and no-test/no-metric boundary without requiring a local
  PhysicsNeMo install.

The generated `docs/showcase/generated/ecosystem_compatibility_summary.tsv` and
`ecosystem_compatibility.png` are derived from this evidence section. This keeps
official protocol and ecosystem rows source-of-truth driven instead of
hard-coded in the showcase generator.

## Tradeoff

The local strong baseline is fast and already comparable to the current claim,
but it cannot support an external SOTA claim. Measured NeuralOperator FNO and
UNO reproductions, a measured PDEBench U-Net architecture reproduction, and a
measured official simplified CNO1d reproduction are stronger evidence because
they use external architectures under the claim protocol. The remaining gap is
publication and transfer comparability: published table values, CNO2d square-grid
settings, pretrained checkpoints, and stronger foundation-model transfer
protocols remain unmapped. The scalar-only Poseidon finetune gate is now
measured and stopped on validation, so broader public-baseline claims need either
a stronger train/validation-only foundation adapter or UPS-side validation
improvement before any new held-out transfer measurement.
For PhysicsNeMo, the next gate is deliberately narrower: run a live recipe
adapter on train/validation with `python scripts/run_physicsnemo_compatibility_smoke.py --live-recipe --eval-split val`
in a Python 3.11+ or PhysicsNeMo container environment and record provenance
before reporting any UPS metric.
