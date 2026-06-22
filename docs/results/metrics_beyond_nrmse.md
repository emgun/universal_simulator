# Metrics Beyond NRMSE

Decoded rollout NRMSE remains the primary public metric because it is the metric
shared by the current `light-v1` result, matched persistence/Fourier
baselines, and measured third-party reruns.

Secondary metrics are still important. They show what kind of improvement the
primary metric is capturing and where it is not yet broad.

## Current Secondary Metric Read

The committed evidence supports this primary UPS versus persistence comparison:

| Metric | UPS | Persistence | Relative Change |
| --- | ---: | ---: | ---: |
| Rollout NRMSE | 0.4166 | 0.5702 | 26.94% lower |
| Rollout MAE | 0.1789 | 0.3007 | 40.51% lower |
| Rollout MSE | 0.1734 | 0.3249 | 46.62% lower |
| Spectral energy error | 0.0672 | 0.0672 | approximately neutral |
| Step-1 NRMSE | 0.7177 | 0.7177 | approximately neutral |
| H4 NRMSE | 0.5580 | 0.5580 | approximately neutral |
| H16 NRMSE | 0.0877 | 0.5582 | 84.29% lower |

The public interpretation is narrow and useful: UPS currently wins by improving
aggregate rollout error and long-horizon H16 behavior. It does not yet show a
broad win on one-step accuracy, H4 error, or spectral energy preservation.

## Why This Matters

NRMSE alone can hide the shape of the result. MAE/MSE say whether the aggregate
error win is visible under common loss scales. Step and horizon metrics show
where in the rollout the improvement appears. Spectral energy error checks
whether the prediction preserves the broad frequency-energy profile.

That mix makes the result harder to overstate: the current result is a bounded
longer-horizon rollout improvement, not a blanket accuracy or physics-fidelity
statement.

## Metrics To Add Later

These should become future result gates once the evaluator records them for
UPS and comparable baselines:

- conservation or invariant gaps for mass/energy-like quantities;
- PDE residual or physics-equation violation where the task exposes operators;
- boundary-condition violation for tasks with explicit boundary contracts;
- runtime, memory, and cost per rollout;
- robustness or out-of-distribution performance across coefficient and initial
  condition shifts;
- uncertainty calibration when models emit predictive uncertainty.

Those future metrics should not be hand-entered into public figures. They
should be emitted by the evaluator, recorded in source records, and regenerated
through `scripts/build_public_assets.py`.
