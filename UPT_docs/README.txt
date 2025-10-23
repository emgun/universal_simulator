# UPT YAML Templates (Small/Medium/Large)

These config files are designed as **starting points** for the ml-jku/UPT codebase (`main_train.py --hp <yaml>`).
They follow a conventional layout with sections for `dataset`, `model`, `trainer`, and `logging`.
If your local fork uses different key names, keep the values but map the keys accordingly.

## Quick start
```bash
# example
python main_train.py --devices 0 --hp /path/to/upt_small.yaml
# or
python main_train.py --devices 0,1 --hp /path/to/upt_medium.yaml
```

## Notes
- **Tokens & capacity**: scale in this order → `latent_dim`, `depth`, `n_latent_tokens`, `num_supernodes`.
- **Latent rollout**: enable inverse losses (encoding/decoding) for stable long-horizon inference.
- **Mixed precision** recommended; enable EMA for stable validation.
- If you see OOM, reduce `query_points_per_batch` first, then `num_supernodes`.
