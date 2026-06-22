# Credibility Cards

The credibility cards make the public results easier to review without changing
the benchmark metrics.

## Cost And Reproducibility

`generated/reproducibility_card.png` and
`generated/reproducibility_card.tsv` are generated from committed records.

Current facts:

- Asset regeneration is zero-GPU.
- Asset regeneration does not hydrate datasets or rerun benchmarks.
- The repeatability command is `python scripts/build_public_assets.py --check`.
- Seven input files and the generated asset set are tracked in
  `generated/asset_manifest.json`; the reproducibility card reports
  twenty-six generated files, including the manifest itself.
- The primary artifact hash is recorded in the source record.
- Benchmark dollar cost is not displayed because the committed scorecards do not
  record dollar cost.

That last point is intentional. A public cost number should only appear after
the run record includes provider, hardware, wall-clock, GPU-hours, and dollar
cost.

## Benchmark Readiness

`generated/benchmark_readiness.png` and
`generated/benchmark_readiness_summary.tsv` split benchmark surfaces into lanes:

- matched third-party baselines: FNO, UNO, PDEBench U-Net, and CNO1d are already
  measured under the repo `light-v1` protocol;
- official external protocols: PDEArena and RealPDEBench are planned external
  protocol checks, not directly comparable to `light-v1`;
- ecosystem compatibility: PhysicsNeMo has a dry compatibility smoke manifest
  and a live validation-only FNO recipe-adapter repeat under Torch 2.10;
- future model or recipe surfaces: CFO, PDEformer-2, and Poseidon remain useful
  research tracks but are not current held-out comparable benchmarks.

The card keeps measured repo-protocol baselines separate from published-paper
leaderboard results.

`generated/ecosystem_compatibility.png` and
`generated/ecosystem_compatibility_summary.tsv` expand those lanes into the
concrete official-source adapters, validation-only transfer gate, and planned
PDEArena plus validation-only PhysicsNeMo compatibility surfaces.

## Repeatability

Regenerate and check all credibility cards with:

```bash
python scripts/build_public_assets.py
python scripts/build_public_assets.py --check
```
