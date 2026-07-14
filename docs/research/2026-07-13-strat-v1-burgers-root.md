# `strat-v1` Burgers Root

Date: 2026-07-13

Status: complete and durably published. No
baseline, candidate, validation, or held-out model evaluation was run.

## Source and schema gate

All 12 official PDEBench Burgers training files were downloaded on one
ephemeral Vast.ai instance and verified against the committed DaRUS sizes and
MD5 checksums. The catalog is 98,795,619,744 bytes (about 92.0 GiB).

The first verified file, DaRUS file `268190`, froze the raw contract:

- `tensor`: uncompressed `float32 [10000,201,1024]`;
- raw layout: `[sample,time,x]`;
- canonical layout: `[sample,time,x,channel]`;
- `x-coordinate`: monotonic `[1024]`;
- `t-coordinate`: monotonic `[202]`, one longer than the stored 201 tensor
  states, so the adapter does not invent an unverified coordinate offset.

Inspection evidence is
`docs/research/artifacts/strat_v1_burgers_first_file_inspection.json`.

## Canonical source

The checksum-bound hydrator selected the same deterministic 36 source rows in
each viscosity regime and produced 432 finite trajectories with explicit
`source_file_id`, `source_sample_index`, and `nu` provenance.

- path: `data/pdebench_burgers_canonical/burgers1d_train.h5` (ignored local data);
- bytes: `198064650`;
- SHA-256: `746a6731160f1aa69db2344d5d2a28261829a541a4f4323c3e8406f12f7b69d7`;
- selection identity SHA-256:
  `ad975520a51cb2f16708f845e5a5c9a0e5c0ab495ddc52b1d1f705c74e4211bb`.

The full source record is
`docs/research/artifacts/strat_v1_burgers_source.json`.

## Balanced root

The portable local root is `data/pdebench_burgers_strat_v1/`:

| split | shape | per viscosity | bytes | SHA-256 |
|---|---:|---:|---:|---|
| train | `[288,201,1024,1]` | 24 | 136414903 | `9b7ae18e229641e2b75962673ca7699ff75fd2a51df4178ce2771d0c4ee4fd82` |
| val | `[72,201,1024,1]` | 6 | 32289060 | `496a66bc4366d88d83fbbf9842ae14e2c93c2b726a27d9e6ac26ccd4ada68e73` |
| test | `[72,201,1024,1]` | 6 | 26148529 | `b4d1f65a5ba9b60c97832dc843fb2d58a0a61f2e81a7ac39af3244b49bc1b0a9` |

Independent checks confirmed all fields are finite, all 12 viscosities are
exactly balanced, every provenance pair is unique within a split, and
cross-split provenance overlap is zero. The local rebuild's HDF5 container
hashes intentionally differ from the remote build because local manifests use
portable local source URIs; every substantive HDF5 dataset matched the remote
dataset-content hashes exactly.

## Compute and storage decision

The preparation used Vast contract `44740931`, a 250 GB ephemeral disk at
`$0.0875/hour`. The instance was destroyed and confirmed absent after local
verification. Credit moved from about `$0.6491` to `$0.5704`, so no `$5`
top-up was needed.

This validates the intended storage strategy: use cheap ephemeral compute for
large one-time raw hydration, retain only the approximately 388 MB canonical
and derived local artifacts, and publish those compact artifacts to durable
object storage before deleting or relying on local-only copies.

## Durable publication and fetch verification

The canonical source and all three shards are published at their immutable
SHA-256 object keys. Release controls are under
`strat-v1/releases/burgers1d/9120f76b0410aa1835821940d3d3b8461fbf8379e0bacaf127d704c8b5460115/`.
The combined universal training fetch and physically separate measurement
fetch both completed from B2 with zero cache hits and passed re-verification.
No model read the reserved test shard. A3 is frozen in
`docs/research/2026-07-13-strat-v1-contract.md`.
