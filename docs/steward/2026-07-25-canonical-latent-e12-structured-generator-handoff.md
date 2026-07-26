# Canonical Latent E12 Structured-Generator Handoff

Date: 2026-07-25

## Decision

E12 is `structured_generator_not_qualified`.

Keep the E10 exact universal projection and its ordered coefficient space
frozen. Keep routing closed. E12 demonstrates strong structured-operator
accuracy and positive elementary-to-composite transfer, but it does not satisfy
the frozen worst-basis-action or high-frequency gates.

## Decisive evidence

- candidate one-step/eight-step decoded NRMSE:
  `0.003167` / `0.005735`;
- zero-shot decoded rollout NRMSE: `0.005364`;
- candidate/scratch and candidate/full ratios:
  `0.004689x` / `0.459731x`;
- candidate/E11 dense and zero-shot/E11 dense ratios:
  `0.002337x` / `0.002184x`;
- worst temporal NRMSE: `0.050053`;
- combined semigroup mismatch: `1.01e-15`;
- cross-observation coefficient/decoded mismatch:
  `1.54e-14` / `6.25e-15`;
- advection norm drift near `4.20e-15`;
- diffusion monotonicity: `1.0`;
- final composite high-frequency NRMSE:
  `0.295429` versus `<=0.15`;
- elementary/pretrained maximum decoded basis-action error:
  `0.0999215` / `0.1084054` versus `<=0.05`;
- all other gated generator-identification diagnostics pass;
- oracle one/eight-step errors:
  `9.87e-16` / `5.24e-15`;
- splitting high-frequency NRMSE: `0.294490`, so it does not repair the
  failure;
- complete replicated SHA-256:
  `7b214639d6287ede84352bbcfe7a31a1c9e891f4080fc9c623aee5df5e9c5ccc`;
- clean accepted execution HEAD: `9b9ac597...`;
- held-out, provider, routing, label, and source-bypass counts: zero.

The accepted result retains every required arm and regime: base `48`, temporal
`48`, semigroup `48`, physics `36`, composition gaps `16`, and both
cross-observation rules with exact E10 realization counts.

## Interpretation

E11's generic dense-map failure was principally operator inductive bias: the
structured generator reduces rollout error by more than two orders of
magnitude and establishes genuine positive transfer.

E12 also shows that excellent aggregate rollout is not sufficient. Roughly
`6%` generator Frobenius error and small off-support leakage can preserve
global energy and average trajectories while producing `10%` worst-mode action
error and nearly `0.30` high-frequency NRMSE. Amplitudes remain near one, so
the residual is modal/phase fidelity rather than collapse.

The analytic generator class and E10 representation are not refuted. The
remaining question is whether E12's full 2,304-parameter skew matrices are
overparameterized for the excitation/training protocol or whether AdamW fails
to recover the identifiable modal law.

## Next arc

Freeze an E13 mode-resolved identifiability audit before state access:

1. reproduce the E12 data and accepted oracle without changing thresholds;
2. persist per-basis/per-parameter phase and amplitude diagnostics and their
   argmax identities;
3. add a closed-form or oracle-support-sparse generator-recovery control;
4. compare direct recovery with the full skew parameterization;
5. use the unchanged basis-action and high-frequency gates;
6. keep the encoder frozen and prohibit routing, experts, extra seeds,
   post-hoc E12 updates, and nonlinear expansion.

If sparse/direct recovery passes, move to a preregistered parameter-tying or
optimization challenger. If it fails, strengthen elementary excitation before
expanding the operator family.
