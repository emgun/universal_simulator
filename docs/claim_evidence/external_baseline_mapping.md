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

## Tradeoff

The local strong baseline is fast and already comparable to the current claim,
but it cannot support an external SOTA claim. Measured NeuralOperator FNO and
UNO reproductions are stronger evidence because they use external
implementations under the claim protocol. The remaining gap is breadth and
publication comparability: CNO, PDEBench U-Net, or a foundation-model transfer
contract would reduce single-library bias before making broader public-baseline
claims.
