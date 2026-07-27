# Canonical Latent E15 Training-Package Handoff

Date: 2026-07-26

## Decision

E15 is
`deterministic_objective_adamw_restart_repairs_e12_checkpoint_only`.

The complete recovery vector is:

- deterministic schedule-weighted AdamW from neutral: fail;
- fresh-moment deterministic AdamW from the E12 checkpoint: pass;
- componentwise strong-Wolfe L-BFGS from neutral: pass; and
- componentwise strong-Wolfe L-BFGS from the E12 checkpoint: pass.

The E10 universal projection and the shared 52-coefficient latent remain frozen.
No router, modality expert, encoder update, Fourier hard-coding, or threshold
change is justified. Within the smooth periodic linear family, the remaining
problem is training robustness, not representation insufficiency.

## Decisive evidence

| Arm | Basis action | Composite rollout | Final HF | Recovery |
| --- | ---: | ---: | ---: | --- |
| E12 replay | `0.0999215` | `0.00536402` | `0.261281` | fail |
| AdamW neutral | `0.0760865` | `0.00399506` | `0.205808` | fail |
| AdamW restart | `0.00104862` | `0.000241299` | `0.00676557` | pass |
| L-BFGS neutral | `1.60220e-5` | `1.45241e-5` | `3.83964e-5` | pass |
| L-BFGS restart | `1.00893e-5` | `4.75842e-7` | `1.79828e-5` | pass |

All replay, occurrence-count, E12 reproduction, sealed E13 ceiling, objective
integrity, finiteness, coverage, and boundary gates pass. Grouped and literal
weighted objectives match, all-ones weights recover E13, and cross-block
gradients are exactly zero.

The classification uses registered precedence. It must not hide that both
L-BFGS arms pass. The causal reading is:

- deterministic weighting alone does not rescue neutral AdamW;
- the E12 checkpoint places fresh AdamW in a recoverable basin; and
- the componentwise L-BFGS package recovers from either registered start.

This does not prove pure optimizer causality or broad initialization
robustness.

## Evidence seal

- execution HEAD:
  `b94062980e70965d951d0051f9db6d50772305c7`;
- bundle:
  `3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a`;
- compact result:
  `e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d`;
- detached manifest:
  `1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311`;
- raw replicate:
  `f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed`;
- canonical replicate:
  `eec9fbc5bca20fc7c94600217dd13cc55d18fc779cae2b4990e0aed2be191758`;
- focused E15 tests: `40/40`;
- related E12-E15 tests: `111/111`;
- full clean-HEAD unit suite: pass;
- independent pre-execution review: GO;
- independent erratum review: GO;
- independent post-result review: GO.

The first invocation at HEAD `4d14a071...` stopped before publication due only
to an oracle API dispatch defect. The contract's pre-execution erratum records
the repair; it produced no scientific result or durable evidence.

## Boundary

E15 reads 768 training and 256 validation trajectories, with zero held-out
reads. It performs zero encoder updates and uses zero provider calls, routing
paths, labels, or source bypasses. The evidence is synthetic periodic scalar
linear only; nonlinear and particle dynamics remain unqualified.

## Next arc

Preregister one minimal E16 robustness gate. Preserve the two viable packages:
componentwise L-BFGS from neutral and E12-to-deterministic-AdamW repair. Test
them across multiple frozen training realizations or an equivalently strong
predeclared perturbation set. Add joint full-objective L-BFGS only to isolate
the componentwise curriculum.

If one practical rule remains stable, stop optimizer archaeology and move to a
nonlinear shared-latent generator contract. If neither is stable, keep the
linear result as a representation/identifiability ceiling and redesign the
training rule without touching the encoder or routing.
