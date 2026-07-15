# Darcy FNO regime-balanced objective D3

The pre-registered validation-only D3 run completed both matched seed-17
trajectories through epochs `3/6/12/24/48/96/192/384`. The universal training
lock contained only train and validation objects; no measurement-lock or
held-out object was staged or read.

The regime-complete mean-loss control selected epoch 384 at primary validation
NRMSE `0.11694165553982801`, beta-100 global-scale NRMSE
`0.25762147974587746`, and maximum corrected spread `2.2029915564017024`.
It satisfied the frozen plateau rule at epoch 384.

The matched minimax candidate used the same direct beta-conditioned FNO,
sample order, optimizer, and update count. Its only change was the registered
objective `0.5 * mean(per-regime MSE) + 0.5 * max(per-regime MSE)`. It selected
epoch 384 at primary NRMSE `0.1230189209192746`, beta-100 NRMSE
`0.2706539139312041`, and spread `2.200099886332185`. It did not plateau.
Relative to the matched control, primary performance worsened `5.20%`,
beta-100 error worsened, and the spread reduction was negligible. The frozen
interpretation is `regime_balanced_objective_not_validated`.

The candidate still showed causal beta use: shuffled-beta relative degradation
was `9.922351624289874`, and counterfactual relative prediction RMS was
`0.9985636385951515`. The failure is therefore not missing parameter use. It
shows that a simple worst-regime penalty does not overcome the remaining
high-beta representation/capacity imbalance.

The immutable bundle was uploaded and read back with SHA-256
`399600af0d10df94ca91e4b1b270e4455718fe4e047c44292fdff4a4b6506b69` at
`b2://pdebench/remote-runs/darcy-fno-regime-balanced-objective/immutable/sha256/399600af0d10df94ca91e4b1b270e4455718fe4e047c44292fdff4a4b6506b69/darcy_fno_regime_balanced_objective_20260715T033430Z.tar.gz`.
Vast contract `44940764` was destroyed after publication, and its plan-keyed
temporary prefix was purged only after remote read-back verification.

Stop objective and sampling tuning on this FNO specialist. The next bounded
branch should compare the strongest direct conditioned FNO control against one
parameter-conditioned steady-operator architecture with materially different
multiscale capacity. Keep selection validation-only and preserve the same
spread, plateau, causal-parameter, and held-out-zero gates.
