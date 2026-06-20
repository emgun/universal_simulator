# External Benchmark Matrix

This matrix separates measured third-party baselines from future external
benchmark surfaces. Rows marked `measured` are already rerun under the UPS
`light-v1` protocol. Rows marked `future_or_partial` are useful public
credibility targets, but they are not current benchmark claims.

| Surface | Status | What It Proves | Next Step | Claim Boundary |
| --- | --- | --- | --- | --- |
| PDEBench FNO via NeuralOperator | measured | Canonical Fourier Neural Operator family measured on the same `light-v1` split, horizon, sample cap, and decoded rollout NRMSE. | Keep in the matched-protocol benchmark table and refresh if the claim protocol changes. | Fair repo-protocol baseline; not a published PDEBench table comparison. |
| NeuralOperator UNO | measured | A second NeuralOperator family measured under the same claim protocol. | Keep as a measured third-party baseline. | Fair repo-protocol baseline; not a published-paper comparison. |
| PDEBench U-Net | measured | An official PDEBench architecture family adapted and measured under the same claim protocol. | Keep as a measured external architecture baseline. | Fair repo-protocol baseline; not a published PDEBench leaderboard row. |
| CNO1d | measured | Official simplified CNO family measured under the same claim protocol. | Keep as a measured third-party baseline; avoid implying CNO2d or CNO-FM parity. | Fair repo-protocol baseline; not a full Representative PDE Benchmark claim. |
| Poseidon | future_or_partial | Foundation-transfer path has adapter/readiness and validation-only evidence, but scalar finetuning did not clear the held-out gate. | Continue only with controlled unfreeze or LoRA on train/validation before any held-out test. | Foundation-model transfer track, not currently a held-out claim-comparable benchmark. |
| PDEArena | future_or_partial | Would test UPS against an independent multi-scale PDE surrogate benchmark protocol. | Add only after current `light-v1` showcase is stable. | External protocol; not directly comparable to `light-v1`. |
| PhysicsNeMo | future_or_partial | Would demonstrate compatibility with a broader SciML benchmarking/validation ecosystem. | Treat first as compatibility and recipe coverage, not a win/loss benchmark. | Ecosystem compatibility surface; no current UPS metric. |
| RealPDEBench | future_or_partial | Would test whether simulated-PDE evidence transfers toward real-world paired data. | Use later as a guard against overbroad physics-foundation claims. | Separate sim-to-real benchmark, not a current UPS claim. |

The durable machine-readable version of the current matrix is
`docs/showcase/generated/external_benchmark_matrix.tsv`.
