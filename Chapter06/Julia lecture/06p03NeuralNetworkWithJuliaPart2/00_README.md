# 6.3 Neural Network with Julia (Part 2)

**Problem.** Build deeper and wider networks with Lux, and study generalisation: train only on part of the data and test how well the network predicts unseen data - both by interpolation (a random split) and by extrapolation (train on x < x0, predict beyond).

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_DeeperNetworkArchitectures.jl` | Chain/Dense syntax for multi-node, multi-layer networks. |
| `02_TrainValidationInterpolation.jl` | Random 50% train/validation split, Adam training, validation MSE (interpolation works well). |
| `03_Extrapolation.jl` | Train only on x < 2, predict beyond, validation MSE (extrapolation is poor). |
