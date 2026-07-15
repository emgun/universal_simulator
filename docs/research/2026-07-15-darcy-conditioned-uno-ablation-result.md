# Darcy conditioned-UNO D4 result

The validation-only D4 architecture comparison is complete. A parameter-conditioned
NeuralOperator UNO did not beat the frozen D3 direct conditioned-FNO control and did
not clear the specialist gate. No held-out object was staged or read.

## Frozen comparison

| Metric | D3 conditioned FNO | D4 conditioned UNO | Gate |
| --- | ---: | ---: | --- |
| Primary validation NRMSE | `0.1169416555` | `0.1419979831` | UNO strictly lower: fail |
| Beta-100 global-scale NRMSE | `0.2576214797` | `0.3151537442` | UNO strictly lower: fail |
| Maximum corrected spread | `2.2029915564` | `2.2194240893` | `<=1.5`: fail |
| Plateau epoch | `384` | none by `384` | by epoch 384: fail |

UNO used beta causally: shuffling beta degraded its primary metric by `8.4713x`, and
the counterfactual relative prediction RMS was `0.998869`. Those diagnostics rule out
an ignored-conditioning explanation, but they do not offset worse accuracy and regime
balance. The architecture comparison therefore closes negative.

## Protocol and provenance

- Plan SHA-256: `6d988f77706c1f7ab342c398dde7a17509dd19090d8e505d5af01f02f3ccf27c`
- Materialized result SHA-256: `353cdcff26ce14310d9d59ce28e32bbf3b2659e824c6d266a26f6c682e098622`
- Source summary artifact SHA-256: `593aa78d90a1ad658386fc9544387cb9eee32bc51f0e4013858f8cda12866152`
- Immutable archive SHA-256: `6b984e683f533c6b4060a29ac18746ef31835fc74232d960383d5a991464ec41`
- Archive: `b2://pdebench/remote-runs/darcy-conditioned-uno-ablation/immutable/sha256/6b984e683f533c6b4060a29ac18746ef31835fc74232d960383d5a991464ec41/darcy_conditioned_uno_ablation_20260715T200550Z.tar.gz`
- Successful Vast contract: `45016552`, automatically destroyed after publication
- Held-out reads: `0`

The first launch (`45016045`) stopped before epoch 1 because Torch 2.2 has no strict
deterministic CUDA backward implementation for UNO's bicubic interpolation. The
instance auto-terminated. The replacement plan retained fixed seeds, deterministic
cuDNN, and deterministic alternatives while making only unsupported algorithms
warning-only; a regression test and a new immutable plan were committed before the
successful run.

## Decision

Do not promote this UNO recipe, run additional seeds, or access held-out data. D1-D4
show that explicit parameter conditioning is necessary and causal, but neither FNO
head/loss adjustments nor this materially different UNO backbone solves the Darcy
high-beta concentration. Close the narrow specialist search and move up one level:
repair the shared-model conditioning/evaluation contract, then measure whether shared
cross-task structure improves regime handling under the same validation-only gate.
