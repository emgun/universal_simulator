# Canonical Latent E15 Training-Package Result

Date: 2026-07-26
Status: `deterministic_objective_adamw_restart_repairs_e12_checkpoint_only`

## Decision

E15 independently classifies
`deterministic_objective_adamw_restart_repairs_e12_checkpoint_only`.

The complete recovery vector is:

- schedule-weighted AdamW from the canonical neutral state: fail;
- schedule-weighted AdamW with fresh optimizer moments from the frozen E12
  checkpoint: pass;
- schedule-weighted componentwise strong-Wolfe L-BFGS from neutral: pass; and
- schedule-weighted componentwise strong-Wolfe L-BFGS from the E12 checkpoint:
  pass.

Within the frozen smooth periodic linear advection-diffusion family, the E12
checkpoint is therefore repairable without changing the E10 encoder, the
52-coefficient latent, the full structured generator, the data, or any
qualification threshold. Deterministic schedule weighting alone is not enough
for AdamW from neutral, while a warm start changes its outcome. The
componentwise L-BFGS package succeeds from either registered start.

This does not isolate pure optimizer causality: the L-BFGS arm is a frozen
componentwise strong-Wolfe package, and the successful AdamW arm inherits a
checkpoint produced by E12's stochastic scheduled training. It also does not
establish initialization robustness beyond the two registered starts.

## Registered comparison

E15 freezes the E10 projection, E12 trajectories and schedules, the full-skew
additive generator, and all E12/E13 qualification gates. It compares four new
arms under the literal schedule-occurrence-weighted objective:

| Arm | Start | Package |
| --- | --- | --- |
| AdamW neutral | canonical neutral generator | deterministic grouped objective, fresh AdamW |
| AdamW restart | frozen E12 checkpoint | deterministic grouped objective, fresh AdamW |
| L-BFGS neutral | canonical neutral generator | x, y, then diffusion strong-Wolfe L-BFGS |
| L-BFGS restart | frozen E12 checkpoint | x, y, then diffusion strong-Wolfe L-BFGS |

The grouped objective is not assumed equivalent. Before optimization, E15
proves grouped-versus-literal output and loss equality at neutral, the E12
checkpoint, the oracle, and a synthetic source probe; weighted gradients match
for trainable probes. Replacing occurrence counts by all ones reproduces the
E13 objective. Cross-block gradients are exactly zero, and joint gradients
match the mean of the three component losses.

All seven replay locks, the three occurrence-count records, both sealed E13
ceilings, the exact E12 reproduction, objective integrity, and coverage pass.

## Results

| Control | Basis-action NRMSE | Composite rollout NRMSE | Final HF NRMSE | Recovery |
| --- | ---: | ---: | ---: | --- |
| E12 AdamW replay | `0.0999215` | `0.00536402` | `0.261281` | fail |
| Weighted AdamW neutral | `0.0760865` | `0.00399506` | `0.205808` | fail |
| Weighted AdamW restart | `0.00104862` | `0.000241299` | `0.00676557` | pass |
| Weighted componentwise L-BFGS neutral | `1.60220e-5` | `1.45241e-5` | `3.83964e-5` | pass |
| Weighted componentwise L-BFGS restart | `1.00893e-5` | `4.75842e-7` | `1.79828e-5` | pass |
| Oracle | `0` | `1.42147e-14` | `8.32714e-13` | pass |

The neutral AdamW arm improves over E12 but still misses both decisive gates:
basis-action NRMSE remains above `0.05`, and final high-frequency NRMSE remains
above `0.15`. The restart arm clears every gate by a wide margin. Its generator
relative Frobenius errors are `0.0002286` for x advection, `0.0001488` for y
advection, and `0.0001604` for diffusion.

Both componentwise L-BFGS arms also clear every gate. The neutral result closely
matches the independently sealed E13 deterministic ceiling despite using the
literal E12 schedule-occurrence weighting. This weakens any claim that
population weighting itself explains E12's failure.

Coverage contains all five recovery-training controls, six generator
identification cells, 24 validation cells, 10,584 unique mode-resolved records,
and 30 literal argmax cells. State reads are exactly 768 training trajectories,
256 validation trajectories, and zero held-out trajectories.

## Pre-execution repair

The first invocation from HEAD
`4d14a07117f5c6b32a2e5ff3262b888b97d39e49` stopped as
`e15_execution_incomplete` before publication or a scientific conclusion.
Traceback-only replay showed that the oracle integrity probe invoked the frozen
`FixedGenerator` through an absent `forward` method instead of its registered
`step(..., rule="combined")` API.

The contract records a pre-execution erratum authorizing only that dispatch
repair and its direct regression test. Independent review returned GO, the
focused E15 suite passed `40/40`, the related E12-E15 suite passed `111/111`,
and the entire unit suite passed from the new clean execution HEAD.

## Evidence and provenance

The scientific run executed from clean HEAD
`b94062980e70965d951d0051f9db6d50772305c7` under Python `3.12.7`, PyTorch
`2.7.0`, deterministic float64 CPU execution, and one intra-op and inter-op
thread.

The canonical artifact directory contains exactly:

| Output | Bytes | Raw SHA-256 |
| --- | ---: | --- |
| Evidence bundle | `1,589,176` | `3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a` |
| Compact result | `593,358` | `e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d` |
| Detached manifest | `1,748` | `1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311` |

The two raw replicate files are byte-identical at SHA-256
`f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed`.
Their canonical payload SHA-256 is
`eec9fbc5bca20fc7c94600217dd13cc55d18fc779cae2b4990e0aed2be191758`.
Removing the replication record from the complete result yields the replicate
object exactly.

Independent post-result review recomputed the outer and member hashes and byte
counts, deterministic gzip header, archive order and metadata, source bindings,
raw and canonical replicate identity, finiteness, gate precedence, coverage,
state reads, and boundary. It returned GO with no P0/P1 blocker.

## Boundaries

E15 changes no encoder weights and uses no routing path, representation label,
task label, source bypass, held-out read, or provider call. It qualifies no
nonlinear or particle dynamics and is not public or claim-grade evidence. It
does not prove that AdamW restart is compute-matched to L-BFGS, that stochastic
noise caused E12, or that the componentwise L-BFGS package scales to a broader
basis or nonlinear generator.

## Next gate

Do not reopen the encoder, add routing, or tune these four arms post hoc.
Preregister one compact E16 robustness gate before nonlinear expansion:

1. freeze E10, the coefficient semantics, the full generator, E12/E15 gates,
   and the two viable training packages;
2. test componentwise L-BFGS neutral recovery and E12-to-deterministic-AdamW
   repair across multiple frozen, independently generated training
   realizations or an equivalently strong preregistered perturbation set;
3. include a joint full-objective L-BFGS challenger only if it is used to
   separate second-order optimization from the componentwise curriculum;
4. use all registered recovery bits rather than the precedence label alone;
5. stop optimizer archaeology after that robustness decision; and
6. open a nonlinear shared-latent dynamics contract only if the practical
   recovery rule is stable without threshold relaxation.
