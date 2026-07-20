# Canonical Latent E6 Compressed-Locality Handoff

Date: 2026-07-20

## Decision

E6 closes as `compressed_locality_not_qualified`. Keep E5's physical-space,
quadrature-aware locality insight, but close the compact eight-token regional
codec. Do not add routing, an operator, another seed, a radius sweep, or an
anchor-count rung.

This is not a rejection of the universal-latent north star. It is evidence
that equal tensor shape and geometric anchor identity are insufficient when
the latent features themselves discard field information.

## Evidence

Compact artifact:
`docs/research/artifacts/canonical_latent_e6_compressed_locality_result.json`.

- strict `24.5x` compression from `196` source nodes to eight latent tokens;
- no original-source feature, coordinate, or measure bypass;
- exact E3 global checkpoint reproduction;
- local decoder parameters `7,010` versus global `9,089`;
- grid NRMSE `0.278208 -> 0.304720`, `9.53%` worse;
- mesh NRMSE `0.263731 -> 0.323675`, `22.73%` worse;
- high-frequency error worsens `17.58%` grid and `20.26%` mesh;
- both absolute and unseen-resolution local gates fail;
- coverage, mass, invariance, parameter, and no-truncation checks pass;
- effective latent rank is only about `3.9` in both families;
- three complete results and all checkpoints are byte-identical, including a
  final run from the published source bytes;
- no operator, held-out read, provider call, routing, labels, or source bypass.

## Interpretation

E5 showed that the sampled field contains enough information and that local
integral decoding can recover it. E6 shows that the eight-token regional
encoder does not transmit enough information across the bottleneck. The
failure is upstream of dynamics and cannot be repaired honestly with an
under-the-hood router.

## Next coherent arc

Freeze E7 as a deterministic function-space latent sufficiency test before
training another encoder:

1. represent the field by coefficients in one shared multiresolution physical
   basis evaluated from arbitrary coordinates;
2. project grids, meshes, and particles into those coefficients using
   quadrature, without modality labels;
3. decode only from coefficients and require absolute, spectral, refinement,
   invariance, and paired-state gates;
4. derive the coefficient budget before measurement from field bandwidth and
   approximation error;
5. only after this latent space qualifies, train a universal encoder to infer
   its coefficients and later consider a latent operator.

The immediate question is now the sufficiency of the common function space,
not routing and not operator capacity.
