# Chapter 3 - Linear Regression / Curve Fitting

This chapter connects optimisation to machine learning: fitting a model to noisy data. The recipe is always the same - pick a model f_model(a, x) with unknown coefficients a, define a loss S(a) = (1/2) sum of squared errors between data and model, then find the coefficients a that minimise S. Notebook 3.1 does this with GRADIENT DESCENT (first hand-coded for a constant model, then with the library for a straight line). Notebook 3.2 shows that when the model is LINEAR in its coefficients you don't need iteration at all: setting all derivatives of S to zero gives the NORMAL EQUATIONS, a single linear system V'V a = V'y that you solve once. Throughout, data is generated from a known function plus noise so you can check the fitted coefficients against the truth.

## Notebooks in this chapter

### 3.1 Linear Regression with Gradient Descent  →  `03p01LinearLeastSquaresGradientDescent/` (split by goal — see its README)

**Problem.** Given noisy (x, y) data, find the model coefficients that best fit it by minimising the sum-of-squared-errors loss S with gradient descent. Done twice: a constant model f = a0, then a straight-line model f = a0 + a1*x.

**Procedure.**

1. Generate fictitious data from a known function plus noise (so the 'right' answer is known) and plot it.
2. Example 1 (constant model f = a0): write the loss S(a0) = (1/2) sum (yi - a0)^2 and its analytic derivative dS/da0, then hand-code gradient descent a0 <- a0 - alpha*dS/da0 for 1000 steps; recover a0 approximately 2.3.
3. Example 2 (line f = a0 + a1*x): write the model as a[1] + a[2]*x (Julia is 1-indexed) and the loss S(a, x_samples).
4. Visualise the loss surface S over (a0, a1) as a contour plot to see the single minimum at (3, 2).
5. Minimise S with the Optimization.jl library (Descent optimiser, AutoZygote gradient), then plot the fitted line over the data.

### 3.2 Linear Least Squares - Normal Equations  →  `03p02LinearLeastSquaresNormalEquations.jl`

**Problem.** Fit a quadratic y = a0 + a1*x + a2*x^2 to noisy data WITHOUT iterating. Because the model is linear in the coefficients, minimising the squared-error loss gives a single linear system (the normal equations) that is solved in one step.

**Procedure.**

1. Generate quadratic-plus-noise data and show that a hand-guessed set of coefficients fits poorly.
2. Derive the normal equations: set dS/da0 = dS/da1 = dS/da2 = 0, which yields a 3x3 linear system in the coefficients.
3. Recognise that system compactly as V'V a = V'y, where V (the Vandermonde matrix) has columns [1, x, x^2].
4. Build V with hcat(ones, x, x.^2) and solve a = V'V \ (V'y) in a single line - no iteration.
5. Plot the fitted quadratic; the recovered coefficients (approximately 0.5, 1.0, 3.0) match the data-generating function.
