# Universal-Value Roadmap Amendment

**Status:** Active strategy and sequencing authority from 2026-07-13 onward.

**Preservation rule:** This amendment does not edit or replace the frozen
`strat-v1` data release, the 2026-07-13 `strat-v1` contract, the completed A4
validation artifacts, or their recorded hashes. It prospectively supersedes
the goal, exit criteria, and sequencing in Sections 2–9 of
`2026-07-09-universal-baseline-experimentation-roadmap.md`. The A4 scorecard
binds its generated execution-plan hash, not this roadmap file. Keeping the
earlier roadmap as historical context avoids retroactively rewriting the
rationale that preceded the measurements.

## 1. North star

Build one shared simulator that approaches credible task-specialist accuracy
on known systems and earns its shared design through at least one measured
advantage: positive transfer, lower data needed to adapt, or material
operational consolidation. All results must come from trajectory-disjoint,
regime-honest protocols with validation-only selection and reserved held-out
tests.

Training a model is not itself roadmap progress. Every run must resolve one of
these questions:

1. **Reference:** what can a credible specialist learn on this task?
2. **Architecture:** can a shared backbone learn all tasks without destructive
   interference?
3. **Transfer:** does prior shared learning reduce the data or optimization
   needed for another task or regime?
4. **Conditioning:** can one declared interface handle known and inferred
   physical regimes?
5. **Economics:** is one maintained artifact cheaper or simpler than the
   specialist ensemble at comparable quality?

External FNO and UNO runs answer the reference question. Poseidon and the
one-shot `tier_b` retrial answer architecture questions. Explicit and inferred
parameter conditioning answer the regime question. None is the product merely
because it trains successfully.

## 2. Evidence interpretation and protocol erratum

### A4 is a pipeline-calibration wall

The completed three-epoch A4 validation matrix proves that the locked data,
training, evaluation, metric, evidence-return, and teardown path works. Its
numbers remain frozen and reportable as `calibration-wall-v1`.

The learned rows are not claim-grade specialist references: they use three
epochs, one seed, small width, and no convergence criterion. UNO's macro
`0.677776` is therefore an engineering comparison point, not the final B2
threshold. No A4 calibration model receives held-out access.

### `strat-v1.1` metric erratum is required before promotion

The frozen contract's rule that no regime NRMSE may exceed `1.5x` its task mean
is ill-conditioned for Darcy. Per-regime NRMSE normalizes each beta slice by
that slice's own target scale; low-magnitude slices can produce values in the
hundreds while the pooled task NRMSE remains near one. Persistence exhibits the
same behavior, so this is a metric defect rather than candidate evidence.

The data membership, hashes, task metric, and raw per-regime NRMSE reporting
remain unchanged. Before candidate selection, publish a versioned
`strat-v1.1` metric addendum that adds both:

- globally normalized per-regime RMSE, using the frozen task-level validation
  target scale; and
- per-regime error ratio versus persistence under the same normalization.

The promotion gate will use the globally normalized measure or persistence
ratio specified in that addendum. Raw slice-normalized NRMSE stays diagnostic.
No test data may be used to select the corrected gate.

## 3. Universal-value gates

All gates are validation-only until the final measurement step.

| Gate | Requirement | Why it exists |
| --- | --- | --- |
| R0 — recipe adequacy | FNO and UNO only; fixed maximum budget, checkpointing, and stopping declared before training. A recipe is `adequate` only when its validation curve plateaus under the declared rule; otherwise label it `budget-capped`, never `strong`. Use three seeds for the selected architecture before held-out. | Prevents a toy recipe from becoming the claim wall. |
| U1 — useful shared artifact | Exactly one shared checkpoint; beats persistence on every task; macro NRMSE no worse than `1.10x` the oracle per-task specialist macro; no task worse than `1.20x` its specialist. | Tests accuracy without allowing an aggregate to hide task collapse. |
| U2 — negative-transfer control | Compare the joint model with same-architecture single-task ablations at matched update and compute budgets. Joint macro may be at most `1.05x` the ablation macro and no task may regress more than 10%. | Distinguishes sharing from merely packaging several weak specialists. |
| U3 — positive transfer | After pretraining on the other tasks, adapt with 10% and 25% of a task's train split. Versus scratch at equal samples, require at least 20% lower NRMSE, or matched target error with at most half the samples, on at least two task families. | Establishes a reason to prefer a reusable model. |
| U4 — regime capability | Run leave-one-regime-out validation. The known-parameter and inferred-parameter modes are reported separately; each must beat persistence on promoted regimes, and inferred mode must stay within 20% of the metadata-conditioned mode. | Tests regime handling rather than balanced in-distribution interpolation alone. |
| U5 — consolidation economics | Report total/trainable parameters, GPU-hours, examples-to-target, inference throughput, peak memory, and maintained artifact count against the specialist ensemble. Retained-task NRMSE may degrade at most 10% as tasks are added. | Makes operational value visible and stops “one artifact” from being a purely cosmetic claim. |

U1 is the minimum baseline gate. At least one of U3 or a material U5 advantage
must also pass before the project expands to `universal-v1` breadth. U2 and U4
are mandatory safety gates. Failure is an architecture result, not permission
for an unbounded sweep.

## 4. Revised execution order

### Phase V0 — repair measurement semantics

1. Freeze A4 as calibration evidence; do not run its held-out sequence.
2. Implement and validate the `strat-v1.1` regime metrics on existing
   validation predictions or validation-only reruns.
3. Freeze the metric addendum and promotion rule before new candidate numbers
   are used for selection.

### Phase V1 — establish only the references we need

1. Pre-register a validation-only recipe-adequacy ladder for FNO and UNO.
2. Use the smallest checkpoints that can establish a learning plateau under a
   fixed budget. Do not optimize U-Net or CNO unless later evidence makes one
   decision-relevant.
3. Select one architecture by validation, then run its remaining seeds and
   freeze the claim-grade specialist recipe. Still no held-out access.

### Phase V2 — test the actual shared-model hypotheses

Run in parallel after V0:

- Poseidon channel-lift Option A and task-modulated Option B on the locked
  train/validation data;
- the one allowed `tier_b` retrial;
- an explicit conditioning contract that includes task identity, parameter
  identity/value/presence, boundaries, resolution, and equation signature.

Every candidate must emit per-task, corrected per-regime, parameter-use,
compute, and artifact-identity evidence. Choose no winner on macro NRMSE alone.

### Phase V3 — test whether sharing creates value

For the best shared candidate only:

1. run matched single-task ablations for U2;
2. run the 10%/25% adaptation matrix for U3;
3. run known-parameter and inferred-parameter leave-one-regime-out validation
   for U4; and
4. materialize the U5 cost and artifact scorecard.

If U1 fails, stop and change the architecture or narrow the product. If U1
passes but neither U3 nor U5 shows value, describe the result as a consolidated
multi-task model, not a universal or foundation simulator.

### Phase V4 — spend held-out access once

Only after R0 and U1–U4 are frozen:

1. select exactly one claim-grade specialist recipe and one shared candidate;
2. pre-register their artifact identities, commands, metrics, gates, and unique
   measurement keys;
3. execute one guarded held-out sequence; and
4. publish negative or positive results without replacement runs.

### Phase V5 — breadth

Expand into additional PDEBench families and The Well only after U1, U2, U4,
and either U3 or U5 pass. Begin with one validation-only held-out-family pilot;
then scale to 8–12 PDE families, one mesh task, and one particle task. The
principal breadth claim is the k-shot adaptation curve, not a task-count total.

## 5. Stop rules and budget policy

- No held-out access for calibration or under-converged recipes.
- No broad baseline zoo: strengthen FNO and UNO first; add another specialist
  only when it could change a decision.
- `tier_b` receives one retrial. Additional in-house sweeps require a passing
  U1 signal or a new written hypothesis.
- Poseidon LoRA is allowed only when the declared frozen-adapter result is
  within 10% of U1 and the missing capacity mechanism is identified.
- No breadth spend after an accuracy-only win; sharing must also demonstrate
  transfer or consolidation value.
- Vast instances use ephemeral scratch and are destroyed after evidence
  collection. Durable object storage holds immutable source/release artifacts;
  training should stage locally rather than stream random HDF5 reads over the
  network.
- Re-estimate the GPU budget after V1 recipe-adequacy plans are frozen. The
  earlier approximately $5 estimate covered toy reproductions and is not a
  defensible claim-grade budget.

## 6. Immediate implementation queue

1. Publish the `strat-v1.1` metric erratum and tests.
2. Add a validation-only FNO/UNO recipe-adequacy planner with checkpoint and
   convergence evidence.
3. Add one lock-bound shared candidate configuration whose conditioning
   sources explicitly include `beta` and `nu` values and presence masks.
4. Add dedicated validation plans for `tier_b` and Poseidon A/B; do not reuse
   legacy `light-v1` launchers.
5. Add a shared-versus-ablated transfer/value scorecard for U2–U5.
6. Implement the final measurement runner only after a specialist and shared
   candidate have passed their validation gates.
