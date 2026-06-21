# Credibility Cards

The credibility cards make the public evidence easier to audit without changing
the benchmark claim.

## Cost And Reproducibility

`generated/reproducibility_card.png` and
`generated/reproducibility_card.tsv` are generated from committed evidence.

Current facts:

- Showcase regeneration is zero-GPU.
- Showcase regeneration does not hydrate datasets or rerun benchmarks.
- The repeatability command is `python scripts/build_showcase_assets.py --check`.
- Seven evidence input files and the generated asset set are tracked in
  `generated/showcase_manifest.json`; the reproducibility card reports
  twenty-six generated files, including the manifest itself.
- The primary artifact hash is recorded in claim evidence.
- Benchmark dollar cost is not displayed because the committed scorecards do not
  record dollar cost.

That last point is intentional. A public cost claim should only appear after the
run evidence records provider, hardware, wall-clock, GPU-hours, and dollar cost.

## Benchmark Readiness

`generated/benchmark_readiness.png` and
`generated/benchmark_readiness_summary.tsv` split benchmark surfaces into lanes:

- matched third-party baselines: FNO, UNO, PDEBench U-Net, and CNO1d are already
  measured under the repo `light-v1` protocol;
- official external protocols: PDEArena and RealPDEBench are planned external
  protocol checks, not directly comparable to `light-v1`;
- ecosystem compatibility: PhysicsNeMo is a planned compatibility surface, not
  a win/loss benchmark;
- future model or recipe surfaces: CFO, PDEformer-2, and Poseidon remain useful
  research tracks but are not current held-out claim-comparable benchmarks.

The card exists to prevent a common overclaim: measured repo-protocol baselines
are not the same thing as published-paper leaderboard results.

`generated/ecosystem_compatibility.png` and
`generated/ecosystem_compatibility_summary.tsv` expand those lanes into the
concrete official-source adapters, validation-only transfer gate, and planned
PDEArena/PhysicsNeMo compatibility surfaces.

## Repeatability

Regenerate and check all credibility cards with:

```bash
python scripts/build_showcase_assets.py
python scripts/build_showcase_assets.py --check
```
