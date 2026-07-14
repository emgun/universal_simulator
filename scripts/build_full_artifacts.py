#!/usr/bin/env python
"""Retired heuristic PDEBench builder.

This workflow copied training bytes into validation/test outputs when upstream
splits were absent, so it cannot satisfy the universal protocol.
"""

raise SystemExit(
    "Archived legacy workflow: heuristic split construction is forbidden. Use "
    "hydrate_official_canonical_source.py plus make_light_hdf5_shards.py and protocol gates."
)
