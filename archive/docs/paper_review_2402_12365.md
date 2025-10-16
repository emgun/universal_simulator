# Review notes for arXiv:2402.12365

## Paper highlights (as summarised)
- Introduces a **multi-resolution latent simulator** that keeps coarse and fine grids active simultaneously, using axial transformers plus frequency-aware conditioning to stabilise stiff PDE rollouts.
- Trains the latent operator with a **curriculum of multi-step supervision** and emphasises long-horizon losses over single-step error so that diffusion correctors only need to clean residual noise.
- Augments test-time computation with a **critic-guided search**: analytical physics rewards are blended with a lightweight MLP critic and a scheduled perturbation budget to explore candidate beams efficiently.

## Alignment with the Universal Physics Stack (UPS)
- Our latent operator still runs on a single token grid (`latent.tokens=128`, `latent.dim=512`) and shallow PDE-Transformer depths, so we match the paper’s backbone only partially. Multi-step rollouts recently landed via `rollout_horizon` and `lambda_rollout`, which mirrors the paper’s curriculum component but without the extra coarse/fine heads.
- Diffusion residual training already leverages stochastic τ sampling; the beta prior we added ensures broader coverage, yet we still operate at one latent resolution and without the paper’s adaptive residual norm targets.
- TTC previously relied on purely analytical rewards. Adding critic support and per-step noise scheduling closes the gap to the paper’s search strategy, but we still need a fully trained critic checkpoint and better logging for critic confidence.

## Improvements implemented in this pass
- Added a **feature-based TTC critic** and a composite reward wrapper so analytical and learned signals can be mixed without bespoke code changes. Evaluation configs now expose the blend weight, hidden size, and dropout so the critic matches the paper’s setup once trained.
- Extended `TTCConfig` with a **noise schedule**; per-step logs now record the active noise variance, enabling the gradual exploration budgets recommended by the paper.
- Documented how to configure these additions in the TTC integration plan and the Burgers TTC presets, keeping validation and test recipes aligned.

## Recommended next steps
- Expand the latent operator to a true multi-resolution stack (e.g., dual token streams with cross-scale attention) so we fully adopt the paper’s coarse-to-fine pathway.
- Train and version a Burgers-specific critic checkpoint to unlock the blended reward path—until then the config keeps the critic weight at zero to avoid random behaviour.
- Track critic and analytical contributions in W&B (mean, std, disagreement rate) to monitor when the critic drifts and to match the paper’s diagnostics.
