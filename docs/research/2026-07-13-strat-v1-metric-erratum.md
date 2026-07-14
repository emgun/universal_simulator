# `strat-v1` Regime-Metric Erratum

Status: repaired by frozen metric-only `strat-v1.1` addendum and validation-only
diagnostic reprojection; candidate promotion may use the corrected gate.

## Finding

The frozen `strat-v1` contract requires every per-regime NRMSE to remain below
`1.5x` the task mean. That comparison is not scale coherent for Darcy because
each regime NRMSE uses its own target slice as the denominator.

The frozen A4 validation summaries demonstrate the defect:

| Model | Darcy task NRMSE | Selected slice-normalized regime NRMSE values |
| --- | ---: | --- |
| UNO | `0.896466` | beta `0.01`: `427.210569`; beta `0.1`: `145.816308`; beta `1`: `16.721679` |
| Persistence | `0.972114` | beta `0.01`: `113.968990`; beta `0.1`: `38.621463`; beta `1`: `4.155866` |

The same pathology in persistence rules out a model-specific explanation.
These raw metrics remain useful diagnostics of relative error inside each
physical regime, but they cannot be compared directly with a pooled task NRMSE
or used for the current `1.5x` promotion rule.

## Non-mutating correction

Do not edit the frozen data release, A3 contract, A4 scorecard, summaries, or
their hashes. Create `strat-v1.1` as a metric-only protocol addendum over the
same immutable data objects and split membership.

For each task, compute the validation target scale once from all validation
targets. For each regime, report:

1. `global_scale_regime_nrmse = regime_rmse / task_validation_target_rms`;
2. `regime_error_ratio_to_persistence = candidate_global_scale_regime_nrmse /
   persistence_global_scale_regime_nrmse`; and
3. the original slice-normalized regime NRMSE for continuity.

The addendum must freeze the epsilon policy, pooling dimensions, aggregation,
and promotion threshold before candidate selection. It may tighten eligibility
but may not change task membership, primary task metrics, training data, or
held-out access rules. The correction is derived and tested on validation only.

## Required implementation evidence

- shared metric helper used by persistence and learned runners;
- unit tests with differently scaled regimes showing the old metric can diverge
  while the globally normalized metric remains comparable;
- regenerated validation-only diagnostic values for persistence and the frozen
  A4 summaries where predictions are available;
- a frozen `strat-v1.1` addendum with a content hash; and
- no measurement-lock staging or test-data read.

## Closure

The frozen addendum is
`docs/data/protocols/strat_v1_1_metric_addendum.yaml` with self-hash
`2fedaaf445d093a40571a475d5793567842582b5a457d7039ab21db525f50ad0`.
The derived artifact is
`docs/research/artifacts/strat_v1_1_validation_regime_diagnostics.json` with
artifact hash
`83f7e9579f641f3c2bc302723d7543137906d1d5bef6663b959dcc06a29254d4`.
It verifies the frozen A4 source-summary hashes, reads only the three locked
validation objects, retains the raw metrics, and reports zero held-out
measurements. The weighted corrected regime metrics reconstruct every frozen
task primary within `3e-9`.
