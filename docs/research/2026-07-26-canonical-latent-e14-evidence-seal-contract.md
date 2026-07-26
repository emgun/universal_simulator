# Canonical Latent E14 Evidence-Seal Contract

Date: 2026-07-26
Status: independently reviewed GO; frozen before sealing

## Reason for a new experiment number

E13 completed two byte-identical scientific runs and independently recomputed
the registered classification
`full_parameterization_deterministic_recovery_succeeds`. Independent result
review nevertheless issued NO-GO because E13's runner:

- stored the canonical minified pre-replication payload hash
  `d04d71e7b528d2332ca9de98bb64d6f065dff4614d82a2416ec5330ce3a41f19`
  under raw-replicate and complete-result hash labels;
- did not bind the raw pretty-printed replicate files, which actually share
  SHA-256
  `e0bc7b0575a8c6982ecc9fbff3899be0792643a57575a2349468a0244e7e6d9c`;
- did not bind the raw combined file, whose SHA-256 is
  `f65cdc9b6b965b5e77715ae5b00d1ecc21dd683b7ffe09c8ce19235d3283159a`;
- emitted no compact reviewed artifact or detached manifest.

The frozen E13 contract forbids editing its source after sampled-state access
and requires a new E-number for repair. E14 is therefore a zero-state
evidence-sealing audit. It does not rerun, tune, replace, or reinterpret any
E13 scientific control.

## Question

Can the already captured E13 raw bytes be:

1. verified against exact preregistered hashes;
2. shown to contain two byte-identical replicate objects;
3. independently recomputed to the same gates and classification;
4. archived durably with deterministic member bytes; and
5. bound by a compact artifact and detached manifest whose hash labels refer to
   the literal files or explicitly named canonical payloads?

Only an affirmative result permits recording E13's scientific interpretation.

## Frozen inputs

Read exactly these three existing files:

- `replicate_a/result.json`;
- `replicate_b/result.json`;
- `complete_result.json`;

under the E13 run directory supplied on the command line. The accepted run
directory is:

`/private/tmp/canonical_latent_e13_identifiability_audit_fe47b93`.

Require these literal raw-file SHA-256 values before JSON parsing:

- replicate A:
  `e0bc7b0575a8c6982ecc9fbff3899be0792643a57575a2349468a0244e7e6d9c`;
- replicate B:
  `e0bc7b0575a8c6982ecc9fbff3899be0792643a57575a2349468a0244e7e6d9c`;
- combined:
  `f65cdc9b6b965b5e77715ae5b00d1ecc21dd683b7ffe09c8ce19235d3283159a`.

Require replicate A and B raw bytes to be exactly equal.

After JSON parsing with no numeric coercion or normalization, require replicate
A and B objects to be equal. Canonical JSON means UTF-8
`json.dumps(value, sort_keys=True, separators=(",", ":"))` with default
Python numeric formatting. Require:

- canonical replicate object SHA-256:
  `d04d71e7b528d2332ca9de98bb64d6f065dff4614d82a2416ec5330ce3a41f19`;
- canonical combined object SHA-256:
  `022f8867fab4f9b28f84da501245928ee3ef3f074dc6d1a84ed04c56bad7abbb`.

Require the combined object with its top-level `replication` field removed to
equal the replicate object exactly. Treat E13's embedded replication hashes as
known mislabeled fields: record them but do not trust or copy them into E14
integrity decisions.

## Frozen source and zero-state boundary

Before reading any E13 result byte, require a source-only preflight:

- this contract, the E14 sealer, the E14 tests, the E13 contract, and the E13
  runner are byte-identical to a clean committed Git HEAD;
- Git HEAD exists;
- the worktree is clean;
- the frozen E13 execution commit
  `fe47b937205f47a2dba93f0ecbeee83015824c09` exists; and
- all eight source paths used by E13 exist at that commit with the exact
  preregistered committed hashes.

Only after that source preflight may E14 read the three E13 result files,
require their literal raw hashes, and parse them. After parsing, independently
validate the complete E13 provenance record:

- the source set is exactly
  `{contract,runner,e12_lock,e12_artifact,e12_runner,e11_runner,e7_runner,latent_evaluation}`;
- every working hash equals its committed hash and the frozen expected hash;
- every source record has `matches_git_head=true`;
- the execution HEAD is exactly the frozen E13 commit, a Git HEAD is present,
  and the E13 execution worktree was clean;
- the E13 config canonical hash is exactly
  `08077c555b3476c51e13b861065835f914fdd6c5ae44e1a22dd8d08ec24d12dc`;
- the E12 replay config hash is exactly
  `cd428d490ad9d5505f88ead66b41fdb25e25830f45d0eb21f451c5dbea261934`;
  and
- E13 boundary counts for held-out reads, provider calls, routing paths,
  representation-label inputs, task-label inputs, and source bypass are zero.

E14 may read only the three E13 result files and repository source/document
bytes. It must not construct a physical state, trajectory, coefficient
dataset, schedule tensor, model, generator, optimizer, oracle, or observation
geometry. Record all E14 state reads and optimizer updates as zero.

If source/Git preflight fails, write no output and exit nonzero. If any E13
input hash or object check fails, write no output and exit nonzero.

## Frozen independent recomputation

Recompute from the replicate object rather than trusting E13's stored pass
booleans.

### Reproduction and preflight

Recompute the E13 provenance, parameterization, oracle, replay-lock, and E12
reproduction results from stored leaves; do not use their aggregate `passed`
booleans to qualify E14. Require:

- exact E12 elementary checkpoint and three generator hashes;
- exact E12 first/final losses, update count, examples per regime, basis action,
  and zero-shot rollout values from the E13 replay lock;
- the observed and expected replay dataset records both equal the frozen seven
  dataset records exactly;
- the observed and expected replay schedule records both equal the frozen five
  schedule records exactly;
- full-skew, support-sparse, and mode-tied parameter counts of `2304`, `90`,
  and `12`, exact mask hashes and nonzero counts, structural invariants, and
  maximum oracle phase below pi;
- exactly the 18 frozen E12 case names in their frozen order for both oracle
  preflight tables;
- oracle and E11 projection-closure maxima, ranks, finiteness, and structure
  recomputed from their 18 stored records;
- all three input covariances rank `48`;
- exactly the groups `A_x` and `A_y`, 21 rotation-plane Grams per group, all
  shape `[2,2]` and rank `2`;
- mode-tied oracle Jacobian rank `12`.

For every rank record, recompute the tolerance as
`max(shape) * eps(float64) * largest_singular_value`, then recompute numerical
rank and condition number from the stored singular values. Require input
covariance shapes `[48,48]` and mode-tied Jacobian shape `[301056,12]`. Do not
recompute from a model or state.

### Evidence coverage and argmax

Require:

- six exact control names;
- six generator-identification records;
- 24 validation records over six controls and exactly
  `{composite,x_advection,y_advection,diffusion}`;
- 10,584 mode records with 10,584 unique
  `(control,basis_index,case_name,horizon)` keys;
- exactly 49 basis indices, 18 E12 case names, and horizons `{1,8}` for every
  control;
- 30 argmax records over the five frozen metrics.

Independently scan all mode records and require every recorded argmax value and
identity to equal the literal maximum.

### Recovery gates

For every learned control independently recompute:

- all finite;
- exact structure residuals and parameter count;
- maximum relative Frobenius error `<=0.10`;
- maximum supported-entry relative error `<=0.20`;
- maximum off-support leakage `<=0.10`;
- maximum diffusion-rate relative error `<=0.20`;
- maximum normalized commutator `<=0.02`;
- one-step maximum decoded basis-action NRMSE `<=0.05`;
- composite final high-frequency NRMSE `<=0.15`;
- maximum elementary one-step decoded NRMSE `<=0.03`;
- maximum elementary eight-step decoded NRMSE `<=0.08`;
- composite zero-shot eight-step decoded NRMSE `<=0.20`;
- zero-shot/persistence ratio using `1.5584481380508215` `<=0.75`.

Require the recomputed aggregate `recovery_pass` to equal the stored value for
every control.

Recompute classification precedence only from the independently recomputed
preflight, reproduction, excitation, coverage, and recovery results. Keep
excitation record integrity separate from the observed full-rank outcome:

- preflight comprises provenance, parameterization, oracle, coverage, and
  excitation record-integrity checks;
- the E12 reproduction input comprises both exact replay-lock agreement and
  exact E12 reproduction leaves; and
- the excitation outcome input is the recomputed required-rank result, not
  excitation record integrity.

Apply this precedence:

1. preflight failure;
2. E12 reproduction failure;
3. either full-skew deterministic recovery passes;
4. support-sparse passes after both full-skew controls fail;
5. mode-tied passes after support-sparse and full-skew fail;
6. excitation rank deficiency;
7. no recovery control qualifies.

Require
`full_parameterization_deterministic_recovery_succeeds`.

## Frozen eight-step caveat

The E13 contract explicitly records but does not gate the eight-step per-basis
relative maximum. E14 must preserve, not average away, this caveat:

- full-skew neutral: basis `41`, `composite_c`, horizon `8`, decoded NRMSE
  `0.3595701757249822`;
- full-skew polish: basis `41`, `composite_c`, horizon `8`, decoded NRMSE
  `0.37147804128286543`.

Require those exact identities and values. Record that the target is strongly
attenuated under maximum diffusion; do not relabel the metric as an absolute
error or add a post-hoc pass threshold.

## Frozen durable outputs

The requested final output directory must:

- not exist at the start of sealing;
- be outside and distinct from the E13 input directory; and
- have an existing parent directory on the same filesystem used for staging.

Publish exactly three outputs inside it:

1. `canonical_latent_e14_evidence_bundle.tar.gz`;
2. `canonical_latent_e14_evidence_seal_result.json`;
3. `canonical_latent_e14_evidence_seal_manifest.json`.

### Deterministic evidence bundle

The gzip-compressed tar archive contains exactly:

- `replicate_a_result.json`;
- `replicate_b_result.json`;
- `complete_result.json`.

Member bytes are the untouched E13 raw bytes. Sort members in the order above.
For every tar member freeze:

- mode `0o644`;
- uid/gid `0`;
- uname/gname empty;
- mtime `0`;
- no PAX headers.

Use GNU tar format and gzip `mtime=0`, empty original filename, and compression
level `9`. Record the archive raw SHA-256 and every member raw SHA-256.

### Compact seal result

The compact JSON records:

- E14 schema and classification `e13_scientific_result_sealed`;
- underlying E13 classification;
- E13 execution HEAD and E14 sealing HEAD;
- all raw and canonical input hashes with unambiguous labels;
- complete independent-recomputation checks;
- E12 reproduction, excitation-rank, coverage, gate, and key metric summaries;
- the eight-step caveat;
- deterministic archive SHA-256 and member hashes;
- zero E14 state/provider/held-out/routing/optimizer counts;
- the boundary of the scientific interpretation.

Serialize with sorted keys, two-space indentation, and a trailing newline.

### Detached manifest

The detached manifest records:

- E14 source and Git provenance;
- literal input paths and raw hashes;
- canonical payload hashes under distinct names;
- compact-result raw SHA-256 and byte count;
- evidence-bundle raw SHA-256 and byte count;
- archive member names, byte counts, and raw hashes;
- final classifications and zero-state boundary.

The manifest must not claim to hash itself. Serialize with sorted keys,
two-space indentation, and a trailing newline.

Construct all three file payloads in memory first. Create a temporary sibling
directory, write and fsync each file there, reopen every staged file, recompute
every declared raw-file hash and byte count, reopen the archive, and verify:

- gzip has `mtime=0` and no original-filename header flag;
- every tar member is a regular file with empty PAX headers;
- member ordering, metadata, byte counts, hashes, and bytes are exact;
- every manifest output and input byte count and raw hash is exact; and
- the staged directory contains exactly the three frozen filenames.

Fsync the staged directory and atomically rename the complete sibling directory
to the absent final output directory, then fsync the parent. A sibling
publication lock created with exclusive creation serializes conforming E14
publishers. Refuse any existing final directory entry, including a broken
symlink, both after acquiring the lock and immediately before rename. The
single-publisher invariant requires all E14 publishers for this output name to
honor that lock. Publication, lock close, lock unlink, and the final parent
fsync share one outer success/failure boundary. Clean up the staging directory
and lock on every failure. If any failure occurs after rename but before all
lock-finalization steps succeed, remove the renamed directory so no durable
partial or unverified seal remains. Lock cleanup is best-effort and must not
mask the primary failure.

The compact result and manifest must explicitly record:

- `e13_original_evidence_status` as scientifically recomputed but originally
  nonconforming;
- `seal_does_not_modify_e13=true`; and
- `sealed_raw_input_hashes` using the literal three-file raw hashes.

E14 is a separate evidence record that binds immutable E13 bytes. It does not
retroactively make E13's original output contract-complete.

## Classification

Classify `e13_scientific_result_sealed` only if every frozen input, source,
recomputation, output, and reopen check passes.

Otherwise emit no durable output and exit nonzero. There is no partial seal,
retry with changed bytes, optional check, or threshold relaxation.

## Interpretation boundary and next move

E14 may establish that the E14-sealed scientific result is evidence-complete
support for assigning the E12 failure to its frozen stochastic
AdamW/scheduled-batch optimization package rather than to E10 representation
insufficiency or generator-class non-identifiability.

It may not establish that:

- L-BFGS alone is causal;
- the full-skew control is robust on every nearly extinguished long-horizon
  mode;
- Fourier semantic tying transfers to other equations or bases;
- nonlinear, particle, or broad multiphysics dynamics are qualified.

If sealed, the next scientific experiment is E15: a preregistered practical
training-package challenger that keeps E10 and the full structured generator,
separates batching/blockwise/optimizer causes, and explicitly handles the
nearly extinguished eight-step mode diagnostic without inventing a post-hoc E13
gate.
