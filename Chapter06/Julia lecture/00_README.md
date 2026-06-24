# Chapter 6 - Introduction to Neural Networks

This final chapter introduces neural networks as flexible, tunable function approximators - the natural successor to polynomial fitting when the target function is complex. It builds up from the simplest possible network and ties together every earlier idea: a network NN(x; W, b) is just a function with parameters (weights W and biases b), fitting it means minimising a squared-error loss, and the gradients of that loss come from automatic differentiation (Zygote). The single notebook walks from a one-input/one-output linear unit, through activation functions and a hidden layer, to actually training a network by gradient descent, and finally to networks with wider hidden layers and multiple inputs.

## Notebooks in this chapter

### 6.1 Introduction to Neural Networks  →  `06p01IntroductionToNeuralNetwork.jl`

**Problem.** Understand what a neural network is (a parameterised function that can approximate any function) and learn to train one: fit the weights and biases so the network matches data, using a loss function and gradient descent with automatically-computed gradients.

**Procedure.**

1. Start with one input, one output: NN(x; W, b) = sigma(Wx + b); plot the activation functions (identity, sigmoid, tanh).
2. See a plain linear unit (sigma = identity) is just a line, then add nonlinearity with tanh; adjust W, b by hand and watch the curve change against some data.
3. Add a hidden layer: NN(x) = W2*tanh(W1*x + b1) + b2, giving four tunable parameters and a far more flexible shape.
4. Train it: define Loss(W, b) = sum of squared errors to a target g(x), get dLoss/dW and dLoss/db from Zygote.gradient, and run hand-coded gradient descent until the network fits the data.
5. Go wider and higher-dimensional: hidden layers with several nodes (more parameters), and a network with two inputs visualised as a contour plot.
