# P2 Model-Side Beta Head Held-Out Launch Credit Blocker

Date: 2026-06-25
Status: blocked before instance creation

## Scope

The user gave continuation direction for one bounded scoped beta-head held-out
pretest route. This note records the provider preflight and launch stop
condition. No held-out data was hydrated or read, no remote wrapper ran, no B2
result artifact was published, and no claim/public artifact was changed.

## Preflight

Repository state:

- Branch: `codex/poseidon-channel-lift-vast`
- Head: `6310ca3` (`Scaffold beta head remote pretest wrapper`)
- Worktree: clean

Contract validation:

```bash
python scripts/validate_p2_model_side_beta_head_pretest_contract.py \
  --contract-json docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json
```

Result: `status = valid`, `errors = []`.

Active instance check:

- No active scoped pretest route instance was found.
- `vastai show instances --raw` exposed one older stopped/exited RTX 4090
  instance (`40586170`), not a running route owner.

Offer search:

```bash
python scripts/search_vast_smoke_offers.py \
  gpu_name=RTX_4090 num_gpus=1 'disk_space>=48' 'dph_total<=0.45' \
  verified=true rentable=true \
  --limit 10 \
  --output-json reports/research/sota_loop/model_side_transport_head_heldout_pretest/vast_pretest_offers.json \
  --output-tsv reports/research/sota_loop/model_side_transport_head_heldout_pretest/vast_pretest_offers.tsv
```

Selected candidate for the single bounded launch attempt:

- Offer: `35374367`
- GPU: RTX 4090
- Price: `$0.34805555555555556/hr`
- Disk available: `761.375 GB`
- Reliability: `0.9935514`
- Location: Netherlands

Launcher dry-run:

```bash
DRY_RUN=1 ALLOW_HELDOUT_PRETEST=1 OFFER_ID=35374367 \
  GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_model_side_beta_head_pretest_vast.sh
```

The dry-run confirmed the intended route:

- remote script: `scripts/run_remote_model_side_beta_head_pretest.sh`
- script args include `DRY_RUN=0 ALLOW_HELDOUT_PRETEST=1 ENV_FILE=.env`
- git ref: `codex/poseidon-channel-lift-vast`
- disk request: `48 GB`
- B2 values were redacted in displayed output
- auto-shutdown path was present

## Launch Attempt

Command:

```bash
DRY_RUN=0 ALLOW_HELDOUT_PRETEST=1 OFFER_ID=35374367 \
  GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_model_side_beta_head_pretest_vast.sh
```

Vast returned before instance creation:

```text
failed with error 400: Your account lacks credit; see the billing page.
```

No contract id was returned. No new instance needed teardown.

## Decision

This is a provider billing/credit blocker, not an experiment result. Do not
retry launches until the user confirms Vast credit/top-up or provides a new
bounded provider route.

The scoped pretest contract, guarded root builder, and dry-run-first wrapper are
still ready. The next valid action after billing is resolved is to re-run the
live preflight, search offers again under the configured price cap, dry-run the
launcher, then make at most one bounded launch attempt.

