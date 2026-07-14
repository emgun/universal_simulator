# Darcy FNO affine-head ablation D2

The pre-registered validation-only D2 run completed both matched seed-17
trajectories through epochs `3/6/12/24/48/96/192`. No measurement-lock or
held-out object was staged or read.

The affine output basis improved selected primary validation NRMSE from
`0.12244374883447355` to `0.10406253061282325` (`15.01%`) and improved the
beta-100 global-scale NRMSE from `0.2702953385456401` to
`0.231361037684217`. It did not pass the frozen specialist gate: maximum
corrected spread was `2.2232885969785094` versus `<=1.5`, and neither arm
plateaued by epoch 192. The registered interpretation is
`affine_head_not_validated`.

Two operational defects were repaired without changing measurement evidence:
full-validation CUDA forwards were replaced with bounded batches and v3
checkpoints now contain tensor-only model state; then the materializer's exact
decimal comparison was replaced by a tightly bounded check for float32-decoded
regime labels. The self-hashed repair manifest binds the original plan,
measurement commit, summary, and both verifier versions.

The recovered immutable bundle is
`b2://pdebench/remote-runs/darcy-fno-affine-head-ablation/immutable/sha256/c4322ce908538030018c46e48462232dc8d85c5c13198add400a6d058a695918/darcy_fno_affine_head_ablation_recovered_20260714T214812Z.tar.gz`
(`47,146,253` bytes). Its downloaded SHA-256 was rechecked as
`c4322ce908538030018c46e48462232dc8d85c5c13198add400a6d058a695918`.
Temporary resumable prefixes were purged after verification, and Vast contract
`44919095` was destroyed.

Next, stop extending this FNO head family. The result shows that beta-aware
capacity helps, but beta-100 absolute error and non-plateauing optimization are
not solved by a linear output basis. The next bounded branch should test a
regime-balanced objective or sampling policy under validation only, with a
stronger steady-operator architecture held as the secondary alternative.
