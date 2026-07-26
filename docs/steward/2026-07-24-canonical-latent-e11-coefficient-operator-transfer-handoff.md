# Canonical Latent E11 Coefficient-Operator Transfer Handoff

Date: 2026-07-24

## Decision

E11 is `coefficient_dynamics_not_qualified`.

E10 representation closure, exact truth, and cross-observation controls pass
near numerical precision. The frozen dense residual coefficient MLP fails
downstream. Do not reopen the encoder or add routing.

## Decisive evidence

- candidate one-step/eight-step decoded NRMSE: `0.28894` / `2.45417`;
- scratch/full-data/persistence rollout: `1.51263` / `0.93814` / `1.55845`;
- candidate/scratch and candidate/full ratios: `1.62245x` / `2.61598x`;
- zero-shot/persistence ratio: `1.57620x`;
- composite temporal extrapolation: `3.64030`;
- composite semigroup mismatch: `0.17979`;
- maximum advection `L2` drift: `1.01032`;
- diffusion monotonicity: `0.376953`;
- cross-observation coefficient/decoded mismatch:
  `1.5049e-14` / `7.0218e-15`;
- exact truth rollout: `0`, semigroup mismatch below `8.5e-16`;
- complete result SHA-256: `0cbb025e...`, identical in both replicates;
- clean execution HEAD: `52318903...`;
- held-out, provider, routing, and source-bypass counts: zero.

Every arm retains composite/x-advection/y-advection/diffusion temporal and
semigroup metrics plus applicable per-regime physics. The detached manifest
binds the complete decision bytes.

## Interpretation and next arc

Low training loss with poor disjoint-state rollout indicates model-form
generalization failure. A generic dense MLP must rediscover modal phase
rotation, dissipative contraction, time composition, and parameter-state
multiplication.

Freeze E12 before state access:

1. reuse E11 data, controls, and gates;
2. parameterize `v_x A_x + v_y A_y + nu D`;
3. compare combined matrix exponentiation with Strang splitting;
4. include exact oracle generators and the E11 dense control;
5. impose constant-mode, skew-advection, dissipative-diffusion, semigroup, and
   representation-blind boundaries;
6. require positive elementary-to-composite transfer.

The next move is structured composition inside the shared coefficient
operator—not a task router, family expert, or new encoder.
