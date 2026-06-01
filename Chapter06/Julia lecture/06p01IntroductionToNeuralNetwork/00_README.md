# 6.1 Introduction to Neural Networks

**Problem.** Understand what a neural network is (a parameterised function that can approximate any function) and learn to train one: fit the weights and biases so the network matches data, using a loss function and gradient descent with automatically-computed gradients.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_ActivationFunctions.jl` | Plot the identity, sigmoid, and tanh activation functions. |
| `02_SingleNeuronByHand.jl` | Tune a single neuron (linear `Wx+b`, then `tanh(Wx+b)`) by hand against data. |
| `03_HiddenLayerModel.jl` | Build the four-parameter hidden-layer network `W2*tanh(W1*x+b1)+b2` and view its flexibility. |
| `04_GradientDescentTraining.jl` | Train that network with a squared-error loss + Zygote gradients + hand-coded gradient descent. |
| `05_MultipleInputs.jl` | Extend a neuron to two inputs and visualise it as a contour plot. |
