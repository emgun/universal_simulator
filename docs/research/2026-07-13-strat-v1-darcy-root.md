# strat-v1 Darcy Root

Date: 2026-07-13

Status: complete and protocol-gated. No candidate was scored or trained.

## Source authority

The root was rebuilt from all five official PDEBench Darcy Train objects in
DaRUS release 8.0. The checked acquisition lock is
`docs/data/locks/pdebench_darcy_source_v1.training.lock.json`, with lock identity
`d2e04f82ec63d9e6ebdac6f4773f7fb6fc32c6d68a54671fafd6218790745254`.
All 6,553,622,984 source bytes passed exact size and MD5 verification.

The official mapping is a steady operator:

- `nu [N,128,128]`: heterogeneous coefficient input;
- `tensor [N,1,128,128]`: elliptic solution target;
- `beta`: physical regime parameter.

The canonical source contains 78 shared sample identities in each of the five
beta regimes, or 390 coefficient-to-solution pairs. Its HDF5 SHA-256 is
`5248096cfe1cd40cffcdecd0d57d33a4bfa993f12895eeb3a287927ca96f5bd8`.

## Split contract

The deterministic `strat-v1` split contains:

| Split | Samples | Per beta | Unique coefficient groups |
| --- | ---: | ---: | ---: |
| Train | 260 | 52 | 52 |
| Validation | 65 | 13 | 13 |
| Test | 65 | 13 | 13 |

PDEBench intentionally reuses a matched coefficient realization across beta
regimes. The protocol therefore treats `(coefficient, beta)` as the operator
sample identity while grouping the shared `source_sample_index` into one split.
It rejects duplicate coefficient-regime pairs and any shared coefficient group
across splits.

The recorded gates passed:

- zero train/validation, train/test, and validation/test coefficient overlap;
- unique composite provenance in every split;
- unique coefficient-regime pairs in every split;
- identical five-beta coverage and exact balance in every split;
- aligned coefficient and solution arrays with shape `[N,1,128,128,1]`.

Shard identities:

- train: `47945f27fa1f56f856733d3bc1aa1b0b5f498669a73cdb7352940292d71d09fe`;
- validation: `2b345a587f6f95a9ff4a12f6cce80ac4c8c83540a03c2a11f87ffdc91be1b595`;
- test: `ea170e3031a48ed400fdcc255fc8dd67e588388573275f0607d91ad7d3aafca6`.

The training lock identity is
`112fb8071ee7a02f04e9c107635eaa9144bcd88e1981a1e35d88a6d78e367701`.
The separate protocol-audit measurement lock is
`a61965c79a4b1cf30f30c42a2f4845b5e6ec68ff41f8212d52579231615cc5a1`.
The test shard was read only by the recorded split-integrity diagnostic under
measurement contract `strat-v1-darcy-protocol-audit`; no candidate, checkpoint,
or metric selection touched it.

## Runtime validation

Coefficient and solution normalization statistics were fitted independently
from the locked training shard. A normalized batch passed a spawned
`DataLoader` worker read with fields and targets shaped `[2,1,128,128,1]` and
explicit beta conditioning.

This completes the Darcy data gate. It does not authorize model selection or a
held-out model measurement before the full three-task `strat-v1` contract is
frozen.
