# D6 v4 Modular Shared-Trunk Validation Plan

Date: 2026-07-16

Status: retired after a pre-bootstrap provider failure; plan self-hash
`88bcb9c70eefa1f7bda97577ff65dcd82e080022594cb9a3b5181b9418b06487`;
no D6 model run occurred; superseded only by the v5 infrastructure-recovery
contract

D6 v4 inherits the exact scientific question, adapter architecture, seed 17,
training schedule, arms, metrics, U1/U2 thresholds, cost cap, stop rules, and
evidence chain from D6 v3. V1, v2, and v3 were each abandoned before execution
after pre-merge independent review and must never be launched.

The final additional invariant is literal and independent of the supplied
plan: both runner and materializer require the canonical six object-ID/digest
map, exactly six report entries, exactly six unique IDs, exact train/valid
roles, SHA-256 algorithms and values, and agreement between declared count and
list length. A validly self-hashed reduced plan/report is rejected.

The launcher verifies the v4 plan self-hash and validation-only boundary,
rejects all three retired hashes, and requires the selected Git ref to descend
from the plan-bound implementation before network or provider access.

All other boundaries remain unchanged: held-out and measurement-lock access
are forbidden; D5 is not retrained; no extra seed, replacement run, epoch
extension, threshold relaxation, dataset, optimizer, architecture, or
dependency change is permitted. U1 and U2 remain fail-closed.
