# D6 v5 Infrastructure-Recovery Validation Plan

Date: 2026-07-16

Status: frozen executable recovery protocol; plan self-hash
`e05db0d5bbf04bd9ca603225a1aa8a2529390b579a642bcd4a7d84f519cb1709`

## Why v5 exists

The one v4 provider allocation, Vast contract `45126713`, never reached the
tracked bootstrap, data staging, or model execution. It produced no stage
report, result artifact, metric, checkpoint, or held-out read. The sanitized
recovery evidence is bound at SHA-256
`52cc5badf1c16da7fd4bd2fb0750f5c1f687f4469594a5cb73589b9dbe7a717a`.
Therefore no scientific attempt was consumed.

V5 authorizes exactly one infrastructure-recovery provider allocation. It is
not an extra seed or a replacement after observing model evidence. V4 is
retired and rejected by the launcher before provider access.

## Unchanged scientific contract

V5 retains v4's exact architecture, seed 17, four arms, training schedule,
six training/validation objects, metrics, U1/U2 gates, cost ceiling, and
fail-closed behavior. D5 is not retrained. Test-role objects and the
measurement lock remain forbidden.

Executable plan:
`docs/research/artifacts/strat_v1_modular_shared_trunk_plan_v5.json`

- plan SHA-256:
  `e05db0d5bbf04bd9ca603225a1aa8a2529390b579a642bcd4a7d84f519cb1709`
- command SHA-256:
  `21bdc324fbe3f6d450e61eb24c44fd6687874ad754c1590266179bdf21d4009d`
- implementation commit: `ee80e89ecde584eaf6c2182d8b764d45b8372bf4`
- bound source/runtime files: 85
- authorized recovery provider allocations: 1

## Operational repair

The tracked bootstrap emits `REMOTE_BOOTSTRAP_STARTED=1` before setup. The
managed receipt persists whether that marker was observed. If it is absent 15
minutes after provider creation, the watchdog records `startup_failed`, saves
the last provider status, and destroys the instance. The 600-minute overall
runtime and `$0.45/hour` / `$4.50` caps remain unchanged.

The D6 wrapper uses one fixed v5 recovery receipt path and refuses provider
access if that receipt already exists. A failed create request that produced no
instance and no receipt may be retried; one successful provider allocation
consumes the recovery allowance.

Offer selection projects the instantiated GPU plus requested 96 GB disk price,
rather than trusting the search row's lower default-disk total. Both the
projected hourly price and its 600-minute maximum must fit the frozen
`$0.45/hour` and `$4.50` caps before provider creation.

## Interpretation

Only a verified stage report plus the sealed four-arm result can score U1/U2.
A second pre-bootstrap infrastructure failure leaves D6 unresolved and closes
this recovery authorization. A completed model run is interpreted exactly
under the frozen v4 scientific gates; it cannot be retried, extended, or
replaced.
