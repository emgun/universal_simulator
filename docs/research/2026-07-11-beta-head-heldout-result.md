# Beta Transport Head Held-Out Pretest: Scoped Result 0.1298, Tight Val->Test Transfer

Date: 2026-07-11

Status: executed scoped held-out measurement under the pre-registered contract `docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json`, measurement key `9c028afbfb85328fd21fc7de4cffb277fbde274aa042ad63e6499abc562addc3`, recorded exactly once in the ledger. This is a scoped `light-v1 model-side beta-parameter transport-head UPS variant` per the protocol mapping — not a primary claim replacement, not comparable to published tables, and its root swaps Advection to reserved official beta-provenance shards (canonical-root caveat).

## Result

| metric | validation | held-out test |
|---|---|---|
| overall decoded_rollout_nrmse | 0.11122069865007121 | **0.12976493407013082** |
| advection1d | 0.0017868130908052495 | 0.0011108774108008665 |
| burgers1d | 0.14738121412908425 | 0.17446879896821743 |
| darcy2d | 0.188979512124482 | 0.20909553062258152 |

All pre-registered promotion gates passed (`promotion_passed = true`); the summary validator passed with no errors.

## Interpretation

1. **First tight val->test transfer in project history.** Every previous held-out attempt collapsed on advection (0.494->0.784 for Poseidon Option A; 0.35->0.74 for the no-context candidate). The beta head *improved* on test (0.0018 -> 0.0011) because its pretest root carries beta provenance on both splits: the train-fitted `shift = 10.2369*beta - 0.0810` rule interpolates. This is the 2026-07-08 regime diagnosis confirmed by a guarded measurement.
2. **Strongest scoped held-out number recorded**: 0.1298 vs CT1 online transport-context 0.2018 and data-conditioned context-phase 0.1808 — and unlike those, it requires no online context frames, only the physical parameter metadata.
3. Burgers/Darcy match their historical test values exactly (0.17447/0.20910), as expected: the head only acts on advection; the other tasks pass through the frozen checkpoint. Their caveats from the split-integrity audit apply (Burgers test is train-contaminated; Darcy is the honest task).
4. This validates roadmap Track C's premise: parameter conditioning is a working regime-handling capability, ready to be promoted into the strat-v1 candidate contract (C2) and extended with the inferred-parameter mode (C3).

## Execution notes

- Both remote GPU attempts died to credit exhaustion during official-source re-hydration (~$8 total) without reaching the measurement. The pre-registered command is `--device cpu`, and all required data existed locally, so the guarded route was executed locally via `scripts/run_remote_model_side_beta_head_pretest.sh` (contract validation, guarded root build, single ledgered measurement, B2 publish) at zero GPU cost.
- A latent defect was found and fixed in the wrapper: the pre-registered command string contains bare `metric<=value` promotion-rule tokens, and `bash -lc` treated `<` as input redirection — every prior execution path (including both remote attempts, had they survived hydration) would have failed at this line. The fix quotes the promotion-rule expressions at runtime only; the contract text and measurement key are untouched.

## Provenance

- Published bundle: `b2://pdebench/remote-runs/model-side-beta-head-pretest/model_side_beta_head_pretest_20260711T074154Z.tar.gz`.
- Committed copies: `docs/research/artifacts/beta_head_pretest_val_summary.json` (SHA256 `de105d4b09e6656c1303157c3db850d38ea07befc3b9985360b0ad025e9b2941`), `docs/research/artifacts/beta_head_pretest_test_summary.json` (`154676ac3de72ec9d89ef3aa3f9c298f87d201629041c29e3e4252438769ba55`), `docs/research/artifacts/beta_head_pretest_test_ledger.json` (`a6906716c34d0f5fd3bd0a466bced3125cc925509046a18765e57a464835542c`).
- Do not rerun this measurement key.
