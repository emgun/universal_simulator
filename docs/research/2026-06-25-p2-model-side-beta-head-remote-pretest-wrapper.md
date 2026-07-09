# P2 Model-Side Beta Head Remote Pretest Wrapper

Date: 2026-06-25
Status: scaffolded, not executed

## Context

The scoped `light-v1 model-side beta-parameter transport-head UPS variant` has
accepted validation-only evidence, a protocol mapping, a pre-registered
held-out pretest contract, and a guarded pretest-root builder. The remaining
gap was remote orchestration: the validation-only remote wrapper must not be
reused by flipping `split=test`, because the held-out path needs explicit
contract validation, val/test root materialization, ledger discipline, and
artifact traceability.

## Implemented Scaffold

Added:

- `scripts/run_remote_model_side_beta_head_pretest.sh`
- `scripts/launch_remote_model_side_beta_head_pretest_vast.sh`
- `tests/unit/test_run_remote_model_side_beta_head_pretest.py`

The wrapper is dry-run-first. It refuses execution unless both `DRY_RUN=0` and
`ALLOW_HELDOUT_PRETEST=1` are set. By default, it only previews the route.

The planned route is:

1. Validate
   `docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json`.
2. Fetch standard `light-v1` `val`/`test` Burgers and Darcy shards from B2.
3. Restore the small UPS checkpoint source archive from B2.
4. Generate or validate the official Advection hydration plan.
5. Sequentially hydrate official Advection train files into a beta-provenance
   source root using remote scratch.
6. Build reserved Advection `val`/`test` beta-provenance shards from the
   train-file rows specified by the plan.
7. Invoke `scripts/build_p2_model_side_beta_head_pretest_root.py` with
   `--allow-heldout-pretest-root` and the pre-registered measurement key
   `9c028afbfb85328fd21fc7de4cffb277fbde274aa042ad63e6499abc562addc3`.
8. Extract and run the exact pre-registered held-out command from the contract.
9. Validate the validation summary with
   `scripts/validate_model_side_transport_head_summary.py`.
10. Publish only small result artifacts under
    `remote-runs/model-side-beta-head-pretest/`.

The launcher is also dry-run-first. It points at the new pretest wrapper and
passes `ALLOW_HELDOUT_PRETEST=${ALLOW_HELDOUT_PRETEST}` through to the remote
script, so a paid launch without that explicit flag still fails closed before
held-out execution.

## Verification

Passed:

```bash
python -m pytest tests/unit/test_run_remote_model_side_beta_head_pretest.py tests/unit/test_build_p2_model_side_beta_head_pretest_root.py tests/unit/test_validate_p2_model_side_beta_head_pretest_contract.py -q
bash -n scripts/run_remote_model_side_beta_head_pretest.sh scripts/launch_remote_model_side_beta_head_pretest_vast.sh
python -m black --check tests/unit/test_run_remote_model_side_beta_head_pretest.py
python -m ruff check tests/unit/test_run_remote_model_side_beta_head_pretest.py
python -m py_compile tests/unit/test_run_remote_model_side_beta_head_pretest.py
git diff --check
```

The pytest set includes a dry-run preview with temporary output paths. It does
not hydrate local or remote data, launch Vast, execute held-out, or write claim
evidence.

## Decision

The remote wrapper scaffold is ready for review. No held-out pretest has been
executed. No provider work was launched. No claim evidence or public language
was changed.

The next action is a strategic gate, not another local scaffold tick:

- If the user explicitly directs held-out/provider execution for this scoped
  variant, run one bounded Vast offer search/launch using the dry-run-first
  launcher and require `ALLOW_HELDOUT_PRETEST=1`.
- If the user does not explicitly direct held-out/provider execution, stay
  quiet or perform only no-provider review.

