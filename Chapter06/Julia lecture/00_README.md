# Chapter 6 - Introduction to Neural Networks

This final chapter introduces neural networks as flexible, tunable function approximators - the natural successor to polynomial fitting when the target function is complex. It builds up from the simplest possible network and ties together every earlier idea: a network NN(x; W, b) is just a function with parameters (weights W and biases b), fitting it means minimising a squared-error loss, and the gradients of that loss come from automatic differentiation (Zygote). Notebook 6.1 does everything by hand - from a one-input/one-output linear unit, through activation functions and a hidden layer, to training by gradient descent. Notebooks 6.2 and 6.3 switch to the `Lux.jl` library, which defines and evaluates the network for you, then trains it with `Optimization.jl`; 6.3 also introduces the training/validation split and shows the difference between interpolation and extrapolation.

## Notebooks in this chapter

### 6.1 Introduction to Neural Networks  →  `06p01IntroductionToNeuralNetwork.jl`

**Problem.** Understand what a neural network is (a parameterised function that can approximate any function) and learn to train one: fit the weights and biases so the network matches data, using a loss function and gradient descent with automatically-computed gradients.

**Procedure.**

1. Start with one input, one output: NN(x; W, b) = sigma(Wx + b); plot the activation functions (identity, sigmoid, tanh).
2. See a plain linear unit (sigma = identity) is just a line, then add nonlinearity with tanh; adjust W, b by hand and watch the curve change against some data.
3. Add a hidden layer: NN(x) = W2*tanh(W1*x + b1) + b2, giving four tunable parameters and a far more flexible shape.
4. Train it: define Loss(W, b) = sum of squared errors to a target g(x), get dLoss/dW and dLoss/db from Zygote.gradient, and run hand-coded gradient descent until the network fits the data.
5. Go wider and higher-dimensional: hidden layers with several nodes (more parameters), and a network with two inputs visualised as a contour plot.

### 6.2 Neural Network with Julia (Part 1)  →  `06p02NeuralNetworkWithJuliaPart1.jl`

**Problem.** Rebuild the workflow from 6.1 using Julia's `Lux.jl` library, which handles the network definition, parameter initialisation, and evaluation for you, then train these Lux networks to fit data.

**Procedure.**

1. Define the simplest Lux network NN(x) = Wx + b with `Chain(Dense(1 => 1))`; initialise its parameters and state randomly with `Lux.setup`, optionally overwriting W and b by hand (note Lux defaults to Float32).
2. Evaluate and plot the network over a grid; generate noisy data from y = x + 3 and compare the untrained prediction.
3. Define a mean-squared-error loss, get its gradient w.r.t. the Lux parameters with Zygote, and train by hand-coded gradient descent; read back the recovered W and b.
4. Add a tanh activation (`Dense(1 => 1, tanh)`), then a hidden-layer network NN(x) = W2*tanh(W1*x + b1) + b2 (a Chain of two Dense layers); fit it to g(x) = 2*sin(pi*x/5) with hand-coded gradient descent.
5. Retrain the same hidden-layer network with the `Optimization.jl` library (Adam, AutoZygote) instead of hand-coded descent, logging the loss through a callback.

### 6.3 Neural Network with Julia (Part 2)  →  `06p03NeuralNetworkWithJuliaPart2.jl`

**Problem.** Build deeper and wider networks with Lux, and study generalisation: split data into training and validation sets, train only on part of it, and test how well the network predicts unseen data - both by interpolation (a random split) and by extrapolation (train on x < x0, predict beyond).

**Procedure.**

1. Show how to construct multi-node / multi-layer networks with `Chain(Dense(...))`: a 1-hidden-layer (2 nodes) network and a 2-hidden-layer (4 and 6 nodes) network.
2. Generate noisy data from h(x) = 2*cos(pi*x)*exp(-x^2/5) on [0, 5].
3. Interpolation test: randomly pick 50% of the points as the training set, train a 2-hidden-layer (5, 5) network with Optimization.jl (Adam, AutoZygote), plot the prediction over all data, and compute the mean squared error on the held-out validation set.
4. Extrapolation test: retrain the same architecture using only data with x < x0 = 2, then predict for x > x0 and compute the validation MSE - showing the network interpolates well but extrapolates poorly.
