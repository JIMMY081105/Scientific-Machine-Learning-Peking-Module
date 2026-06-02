# 6.2 Neural Network with Julia (Part 1)

**Problem.** Rebuild the workflow from 6.1 using Julia's `Lux.jl` library, which handles the network definition, parameter initialisation, and evaluation for you, then train these Lux networks to fit data.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_LuxLinearModel.jl` | Define `NN(x)=Wx+b` with `Chain(Dense(1=>1))`, set up parameters with `Lux.setup`, and plot. |
| `02_GradientDescentTraining.jl` | Fit `y=x+3` with an MSE loss, Zygote gradients, and hand-coded gradient descent. |
| `03_TanhActivation.jl` | Add a tanh activation: `Chain(Dense(1=>1, tanh))`. |
| `04_HiddenLayerGradientDescent.jl` | Two-layer network `W2*tanh(W1*x+b1)+b2`, trained by hand-coded gradient descent over all four parameters. |
| `05_AdamOptimizer.jl` | Retrain the hidden-layer network with `Optimization.jl` (Adam, AutoZygote) and a loss-logging callback. |
