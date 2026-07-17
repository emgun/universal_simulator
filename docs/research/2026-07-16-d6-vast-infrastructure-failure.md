# D6 Vast Infrastructure Failure

Date: 2026-07-16

## Outcome

The single authorized D6 provider launch did not reach repository bootstrap,
data staging, or model execution. Vast contract `45126713` remained in
`loading` for about 15 minutes with zero CPU/GPU activity and a provider-side
DNS status ending in `cloud.vast.ai`. It was destroyed manually to stop
billing. The managed receipt then reconciled the instance as absent with
`destroyed: true`.

This is an infrastructure failure, not a D6 model result. U1 and U2 remain
unmeasured. No replacement launch is authorized by the frozen D6 contract.

## Frozen identity

- merged Git commit: `e3484810fe2ef4eba728acfc2a83b91fd70732b7`
- executable plan: `strat-v1-modular-shared-trunk-d6-v4`
- plan SHA-256:
  `88bcb9c70eefa1f7bda97577ff65dcd82e080022594cb9a3b5181b9418b06487`
- bound implementation commit: `4a003fa1952a0995574052c5bc5e1e5d8e119815`
- Vast offer: `41890581`
- Vast contract: `45126713`
- local managed receipt:
  `.vast/receipts/d6-e3484810fe2e-99639.json`

## Timeline and cost

- managed receipt started: `2026-07-16T23:56:24.845500+00:00`
- instance absence recorded: `2026-07-17T00:12:21.752098+00:00`
- preflight offer price: `$0.2593518519/hour`
- instantiated total price including allocated disk: `$0.3688888889/hour`
- Vast credit before launch: `$3.5535583706`
- Vast credit after teardown: `$3.5299735299`
- observed credit decrease: `$0.0235848407`

The instantiated price remained below the frozen `$0.45/hour` limit, and the
observed cost remained below the `$4.50` total cap.

## Data boundary

The launch issued six time-limited presigned capabilities for the exact frozen
three training and three validation objects. The remote bootstrap never ran,
so no stage report or D6 result artifact exists. No test object was included,
the measurement lock was not accessed, and held-out reads remain zero.

Private transfer receipts were retained locally for recovery/audit because
finalization correctly refused to publish without a succeeded, destroyed Vast
receipt. They must not be committed.

## Decision

- Do not score D6, U1, or U2 from this attempt.
- Do not treat the provider failure as evidence for or against modular sharing.
- Do not launch a replacement under the frozen one-run contract.
- Any future D6 measurement requires an explicit new authorization and a
  separately reviewed recovery contract, including a bounded startup timeout
  so an inert provider allocation cannot consume the full runtime ceiling.

