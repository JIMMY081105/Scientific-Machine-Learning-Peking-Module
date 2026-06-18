# Chapter 4 - Differentiation and Automatic Differentiation

Every optimisation and fitting method so far needed derivatives. This chapter is about how to actually compute them. Notebook 4.1 surveys the four ways to differentiate (manual, symbolic, numerical, automatic), shows why numerical differencing suffers round-off error, and uses the ForwardDiff and Zygote libraries to get machine-precision derivatives, gradients, and Jacobians. Notebook 4.2 opens the black box and IMPLEMENTS forward-mode automatic differentiation from scratch using Dual numbers - a number carrying both a value and its derivative, propagated by the chain rule. Notebook 4.3 applies that hand-built Dual-number AD to real tasks (Newton-Raphson, a nonlinear system's Jacobian, a regression). Notebook 4.4 tackles implicit / inverse problems: finding an input parameter so that a quantity defined only by solving an equation hits a target, by differentiating straight through the linear solve.

## Notebooks in this chapter

### 4.1 Introduction to Differentiation  →  `04p01DifferentiationInJulia.jl`

**Problem.** Understand the different ways to compute derivatives (manual, symbolic, numerical, automatic) and learn to use Julia's ForwardDiff and Zygote packages to get exact derivatives, gradients, and Jacobians without differentiating by hand.

**Procedure.**

1. Symbolic differentiation with the Symbolics package (good for small problems only).
2. Numerical differentiation: forward/backward/central differences plus the complex-step trick; plot error vs step size h to see truncation and round-off error fight each other.
3. Automatic differentiation with ForwardDiff.derivative / Zygote.gradient - machine-precision derivatives; plot them against the analytic derivative to confirm.
4. Apply AD in place of hand derivatives: Newton-Raphson using ForwardDiff for f'(x), and gradient descent on Himmelblau using ForwardDiff.gradient.
5. Functions with parameters: note ForwardDiff needs a single (vector) argument while Zygote can differentiate w.r.t. several; fit f_model = a1(1 - exp(a2*x)) to data with Zygote gradients.
6. Vector-valued functions: compute Jacobians with ForwardDiff.jacobian / Zygote.jacobian, and see why Zygote rejects in-place array mutation.

### 4.2 Implementation of Automatic Differentiation (Forward Mode)  →  `04p02AutoDiffImplementationForwardMode.jl`

**Problem.** Demystify automatic differentiation by building forward-mode AD yourself. The idea: break a function into elementary operations and carry each value together with its derivative (a 'Dual number'), applying the chain rule at every step.

**Procedure.**

1. Motivate with a worked table: decompose f(x) = 3x + sin(x^2) into steps v1..v5 and propagate the derivative dot(v) alongside each value.
2. Define a Dual struct with fields v (value) and d (derivative).
3. Overload the elementary operations (+, -, *, ^, sin, ...) on Dual so each returns a new Dual with the correct chain-rule derivative.
4. Seed a variable as Dual(x, 1.0) and evaluate a function - the result's derivative field IS df/dx, obtained just by running the function.
5. Plot the Dual-number derivative against the analytic one to confirm they match.
6. Multivariate case: to get a partial derivative, seed the chosen variable with dot = 1 and the others with dot = 0; note this needs one pass per input (the limitation reverse mode fixes).

### 4.3 Application of Forward Mode Automatic Differentiation  →  `04p03ApplicationOfForwardMode.jl`

**Problem.** Use the hand-built Dual-number AD from the previous notebook to solve real problems, so you trust it does the same job as the libraries.

**Procedure.**

1. Extend the Dual type with the extra operations needed (/, cos, and more Number combinations).
2. Newton-Raphson via Dual numbers: find r where dA/dr = 0 for the cylinder, taking f(r).v and f(r).d straight from one Dual evaluation.
3. Nonlinear system: assemble the Jacobian of a vector function from Dual evaluations (seeding one variable at a time) and iterate Newton-Raphson to the solution.
4. Regression: fit a constant model a0 to noisy data by gradient descent, taking dS/da0 from the derivative field of a Dual evaluation of the loss.

### 4.4 Implicit Problems (Parameter Estimation): Linear Equations  →  `04p04ParameterEstimationLinearProblems.jl`

**Problem.** Solve an inverse problem where the quantity of interest is defined implicitly (only by solving a linear system): find the input voltage V1 that makes the circuit current i5 equal zero. This needs derivatives taken THROUGH a linear solve.

**Procedure.**

1. Recall the circuit linear system A i = C and solve it (backslash and LinearSolve).
2. Sweep V1 over a range, re-solving each time, and plot i5 (and i5^2) versus V1 to locate roughly where i5 = 0.
3. Define a loss(p) that, for a trial V1 = p, rebuilds C, solves the system, and returns i5^2 - so evaluating the loss requires solving the linear system.
4. Confirm ForwardDiff.gradient can differentiate the loss through the solve (positive/negative gradient matches the plot).
5. Minimise the loss for V1 with hand-coded gradient descent, then with the Optimization.jl library (Adam) and a callback that logs the loss.
