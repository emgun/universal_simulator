# Research Diagnostics

These figures are useful public context, but they are not held-out benchmark
results. They help explain the current technical path and why the main public
result stays narrow.

## Transport Context Ablation

`generated/transport_ablation.png` is generated from
`docs/claim_evidence/artifacts/ups_advection_data_conditioned_ablation_matrix.json`.

It compares validation-only advection variants:

| Variant | Validation NRMSE | Interpretation |
| --- | ---: | --- |
| Full context shift | 0.0005 | Strongest validation result with full context information. |
| Bounded context shift | 0.4218 | Degrades when the shift search is bounded. |
| No data conditioning | 0.5028 | Degrades without the context-derived shift signal. |

The result supports a technical conclusion, not a broad public result: the
transport win is context-dependent and should remain scoped until a learned or
reduced-context sidecar carries the same behavior under a stricter contract.

## Inferred Transport Transfer

`generated/transfer_validation.png` is generated from
`docs/claim_evidence/artifacts/inferred_transport_transfer_scorecard.json`.

It records train/validation transfer results:

| Task | Status | Validation NRMSE |
| --- | --- | ---: |
| advection1d | validated | 0.0002 |
| burgers1d | validated | 0.0058 |
| darcy2d | skipped | unavailable |

The source scorecard explicitly records train/validation scope and no held-out
test touch. This is useful for the roadmap because it shows cross-task transfer
signal on two tasks, but it is not a replacement for held-out `light-v1`
results.

## Repeatability

Both diagnostics are regenerated with the generated asset command:

```bash
python scripts/build_public_assets.py
python scripts/build_public_assets.py --check
```

Future diagnostic figures should follow the same rule: tracked record in,
generated artifact out, explicit scope note in the docs.
