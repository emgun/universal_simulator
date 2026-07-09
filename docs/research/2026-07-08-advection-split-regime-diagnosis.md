# Advection Split Regime Diagnosis: light-v1 Train/Val/Test Are Disjoint Physics Regimes

Date: 2026-07-08

Status: protocol-level diagnostic. Deterministic, model-free analysis of split data distributions; nothing was trained, tuned, selected, or scored, and no ledger was written. Reading the held-out advection test shard's field values was required to characterize its distribution and is recorded honestly here and in the diagnostic JSON (`held_out_test_data_read = true`, `candidate_scored = false`). Existing claim evidence and claim language are unchanged.

## Motivation

Two independent held-out attempts failed with the same signature — validation-competitive advection that collapses on the held-out test split while Burgers and Darcy transfer cleanly:

- The no-context model-side candidate (2026-06): validation `0.3508` overall, held-out advection `0.7374` (h16 collapse).
- Poseidon `channel_lift` Option A (2026-06-23): validation `0.3578` overall (G2a passed), held-out `0.5551` with advection `0.4937 -> 0.7840`.

A third data point pointed the same way: train-fitted fixed shifts never transferred to validation ("train/validation shift mismatch"), while online context estimators (CT1/CT8) — which measure the shift per sample at inference — transferred fine.

## Method

`scripts/diagnose_advection_split_shift_distributions.py` estimates each sample's transport shift per raw timestep via circular cross-correlation of consecutive frames with parabolic subpixel refinement, then maps it to the nearest official PDEBench Advection beta (shift/step ~= 10.24 * beta pixels at dt=0.01, W=1024).

Calibration on the official beta-provenance shard (`data/pdebench_official_advection_light`, val split, 64 samples): implied betas matched `source_file_index` provenance 63/64 (98.4%). Artifact: `docs/research/artifacts/advection_split_shift_diagnostic_official_val_calibration.json`.

## Result: the light-v1 advection splits are single-regime and disjoint

`data/pdebench` (the frozen light-v1 claim root), all samples analyzed. Artifact: `docs/research/artifacts/advection_split_shift_diagnostic_light_v1.json`.

| split | samples | implied beta (all samples) | mean shift/step (px) |
|---|---|---|---|
| train | 128 | 0.1 | 1.02 |
| val | 32 | 4.0 | 40.97 |
| test | 32 | 7.0 | 71.70 |

There is no overlap. Training data teaches transport at 1 px/step; validation selects candidates at 41 px/step; the held-out test demands 72 px/step. The splits were evidently sliced sequentially from beta-ordered source files rather than stratified.

## What this explains

1. **Both held-out advection collapses.** Any candidate selected on validation is selected for beta-4 dynamics and then evaluated on beta-7 — pure physics extrapolation, roughly 75% faster transport than anything validation rewarded. The val->test gap is a property of the protocol, not of the candidates.
2. **Why the in-house operator never learned advection.** It trained exclusively on near-static beta-0.1 transport and was evaluated on 40x faster dynamics. No amount of capacity (P1.2) or training recipe (P1.3) could bridge a regime that is absent from training data — consistent with both sweeps being flat on advection.
3. **Why online context estimators transfer.** CT1/CT8 and the observed-shift estimator measure each sample's shift at inference time, so regime disjointness is invisible to them. Their held-out robustness is now explained mechanically.
4. **Why beta-conditioned mechanisms transfer on official shards.** The official-provenance roots contain all eight betas in every split, so a train-fitted `shift = f(beta)` rule interpolates rather than extrapolates.
5. **Why Burgers/Darcy transfer.** Only advection has a strong ordered regime parameter driving the split slicing into disjoint dynamics.

## Implications and recommendations

- **The frozen light-v1 protocol stands as-is.** Existing claims remain scoped to it; this diagnosis changes interpretation, not evidence: light-v1 advection is, by construction, a zero-shot transport-speed extrapolation test. Candidates that pass held-out advection must either infer speed online (context estimators), condition on the physical parameter (beta head), or genuinely extrapolate physics.
- **Stop expecting val-selected static candidates to pass held-out advection.** Selection on the val split cannot reward beta-7 competence. Further Phase 2 spend on Option B/LoRA under the current selection protocol has a structurally capped held-out ceiling; this was the right call to make before, not after, that spend.
- **The pending model-side beta-head pretest is the correctly-shaped candidate** for this protocol: it conditions on beta explicitly, and its pretest root carries beta provenance on both val and test. Its held-out result is the next informative measurement.
- **Future protocols (medium-v1 revision, universal-v1) must stratify splits by regime parameter.** The universal-v1 contract (roadmap P3.1) should require per-parameter stratified train/val/test composition, recorded in the shard manifests, so in-distribution generalization and parameter extrapolation become separately measurable claims instead of an accidental mixture.
- **Confirmed: medium-v1 has the identical construction.** All 512 train samples are beta 0.1, all 128 val samples beta 4.0, all 128 test samples beta 7.0 (artifact: `docs/research/artifacts/advection_split_shift_diagnostic_medium_v1.json`). The shared shard-prep pipeline propagated the slicing to every protocol tier, so the medium confirmation result also reflects a context-estimator candidate evaluated on a single-regime beta-7 test split. Any medium-v1 revision or universal-v1 shard prep must stratify before slicing.

## Provenance

- Diagnostic script: `scripts/diagnose_advection_split_shift_distributions.py` (requires `--include-test` to read the test shard; writes `held_out_test_data_read` truthfully; no ledger writes).
- Light-v1 artifact: `docs/research/artifacts/advection_split_shift_diagnostic_light_v1.json`.
- Calibration artifact: `docs/research/artifacts/advection_split_shift_diagnostic_official_val_calibration.json`.
