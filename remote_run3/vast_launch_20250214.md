# Vast.ai Training Launch Attempt (2025-02-14)

- **Command:** `python scripts/vast_launch.py launch --gpu RTX_4090 --num-gpus 1 --disk 120 --datasets burgers1d_v3 --wandb-project universal-simulator --wandb-entity emgun --overrides TRAIN_CONFIG=configs/train_burgers_quality_v3.yaml,TRAIN_STAGE=all --auto-shutdown --repo-url https://github.com/emgun/universal_simulator.git`
- **Outcome:** Failed locally because the `vastai` CLI binary is not available in the execution environment. The launcher successfully generated the onstart script at `.vast/onstart.sh` before exiting.
- **Next Steps:** Install the Vast.ai CLI (`pip install vastai`) or run the command from an environment where the CLI is present, then rerun the launch command.
