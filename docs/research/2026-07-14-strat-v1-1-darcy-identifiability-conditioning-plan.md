# `strat-v1.1` Darcy identifiability and conditioning plan

Date: 2026-07-14

Status: D0 complete; D1 pre-registered and validation-only

## Decision

Do not spend on Poseidon A/B, the `tier_b` retrial, or a broader baseline
bakeoff yet. First isolate whether the failed Darcy reference result is caused
by an omitted physical input.

The frozen Darcy operator is `(coefficient field, beta) -> solution`, but the
R0 FNO/UNO interface supplied only the coefficient field. The same coefficient
realization occurs at all five beta values, so a coefficient-only model cannot
identify the requested solution.

## D0: beta-blind identifiability diagnostic

The exact validation-only diagnostic groups rows by `source_sample_index`,
requires complete one-row coverage at betas `0.01, 0.1, 1, 10, 100`, and
verifies that coefficient inputs are byte-identical within every group. The
mean target across beta is the minimum-MSE prediction available to any
beta-blind function for that coefficient realization.

The 13 complete groups pass the identity checks. The optimistic beta-blind
oracle has pooled global-scale NRMSE `0.8690245710731431` and maximum corrected
regime spread `1.99076428424662` at beta `100`. Its self-hashed artifact is
`docs/research/artifacts/strat_v1_1_darcy_beta_blind_identifiability.json`.
No test object or measurement lock was read.

This closely matches the selected R0 FNO/UNO Darcy spread (`1.9108`/`1.9176`).
R0 is therefore an unconditioned-interface negative. It does not establish
that the FNO or UNO architecture family is inadequate for the declared Darcy
operator.

## D1: matched FNO conditioning ablation

Run one Darcy-only, seed-17 FNO trajectory per arm with validation rungs
`3/6/12/24`:

- `U`: coefficient field only;
- `K`: coefficient field plus a spatially constant, train-normalized
  `log10(beta)` channel and a presence channel.

Both arms use the same frozen train/validation objects, architecture width,
Fourier modes, layers, optimizer, sample ordering, updates, and validation
selection rule. `K` necessarily has a slightly larger lifting layer because it
accepts two additional inputs; that is the practical cost of making the
operator identifiable.

The conditioned mechanism passes only if all of the following hold:

1. `K` improves selected validation NRMSE over `U` by at least 10%;
2. `K` maximum corrected beta-regime spread is at most `1.5`;
3. `K` predictions change under counterfactual beta values;
4. deterministically shuffled beta worsens NRMSE by at least 5%;
5. all primary and regime metrics are finite; and
6. both learning curves satisfy the two-transition, less-than-1% plateau rule
   by epoch 24.

The shuffled-beta and counterfactual checks prevent attributing a gain to beta
when it actually comes from extra channels or optimization noise. Failure to
plateau makes the run budget-inconclusive rather than negative.

The canonical self-hashed plan is
`docs/research/artifacts/strat_v1_1_darcy_fno_conditioning_ablation_plan.json`.
It binds the universal training lock, exact Darcy train/validation object
hashes, D0 artifact, runner bytes, command, gates, and zero-heldout policy.

## Forward branches

- If all gates pass, make parameter value/presence universal in specialist and
  shared candidate interfaces, then repair the lock-bound Poseidon and
  `tier_b` plans before comparing them.
- If performance improves but the curve has not plateaued, pre-register one
  checkpoint-resumable budget extension; do not reinterpret it as failure.
- If beta use is real but spread remains above `1.5`, inspect representation
  and loss weighting before increasing architecture breadth.
- If shuffled beta does not hurt, reject the mechanism claim even if headline
  NRMSE improves.

No D1 outcome authorizes held-out access or public claim promotion.
