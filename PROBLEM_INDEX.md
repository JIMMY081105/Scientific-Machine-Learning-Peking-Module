# Problem Index — Exam / Homework Cheat Sheet

## Quick homework reference (what each one does + which file to copy)

**homework 2** → [day2_code.jl](<Homework/Day02/day2_code.jl>)
- 2 nonlinear equations, plot the zero-contours to see where they cross
- use newton raphson (hand Jacobian, `delta = -J\F` loop) to solve it
- check with NonlinearSolve

**homework 3** → [solution.jl](<Homework/Day03/solution.jl>) (also [one_dimensional_optimization.jl](<Homework/Day03/one_dimensional_optimization.jl>))
- plot f(x) = x·exp(-x²), say how many local max/min
- find the minimum with Optimization.jl (Descent, AutoZygote, x0 as a VECTOR)
- hand gradient descent with the analytic derivative, compare with (b)

**homework 4-5** → [homework_day4and5.jl](<Homework/Day04-05/homework_day4and5.jl>)
- generate noisy sin(x) data, plot it
- fit a cubic polynomial by gradient descent (Optimisers.Descent)
- compare Descent vs Momentum vs Adam, print the MSE of each

**homework 6** → [solution_homework_day06.jl](<Homework/Day06/solution_homework_day06.jl>)
- differentiate exp(-x²)cos(x) three ways: analytical, ForwardDiff, hand Dual numbers
- compare all three at one point

**homework 7** → [Solution_Homeworkday07.jl](<Homework/Day07/Solution_Homeworkday07.jl>)
- both partial derivatives of exp(-x1²)cos(3x2): by hand, Zygote.gradient, hand Duals
- seed the Dual you want with ∂=1.0, the other with 0.0

**homework 8-9** → [homework_day8and9.jl](<Homework/Day08-09/homework_day8and9.jl>)
- plot f(x;b) for several b values
- newton raphson to solve for the PARAMETER b (not x!) so f(10;b)=0
- check with NonlinearSolve

**homework 10-11** → [homework_day10and11.jl](<Homework/Day10-11/homework_day10and11.jl>)
- fit a degree-8 polynomial with normal equations `V'V\(V'y)` (general degree via `hcat([x.^k for k in 0:degree]...)`)
- fit the same data with a 1-node NN, then an 8-node NN, compare the fits and final losses

**homework 12** → [solutionday12.jl](<Homework/Day12/solutionday12.jl>) (run-up: [classexerciseday12.jl](<Homework/Day12/classexerciseday12.jl>))
- 2-INPUT neural network: `t_input = [x1data'; x2data']`, first layer `Dense(2 => ...)`
- plot the prediction with tricontourf! (NOT lines! — 2 inputs = a surface)

**homework 13-14** → [SolutionHomework1314.jl](<Homework/Day13-14/SolutionHomework1314.jl>)
- 4 datasets, same columns, different initial condition x(0) — load all, overlay scatter
- NN with x(0) as the 2nd input; train on more and more datasets (a → ab → abc → abcd)
- combine datasets with SPACES `[axzero' bxzero']`, time row is tdata NOT xdata
- after training, predict at x(0)=0.5 by rebuilding `t_input = [dxzero'; dtdata']`


























This is a "what pattern is this question testing, and what do I copy?" index, organized by **problem type** rather than by folder (for folder-by-folder navigation, see [`COURSE_INDEX.md`](COURSE_INDEX.md)). Every entry names the exact file (and line numbers, where a single line matters) that has a working version of the recipe, plus what to edit if the question's wording is slightly different from the original.

General rule that applies everywhere below: **the loss must compare the model to the target `y`, never to an input `x`.** This is the single most common self-inflicted bug when adapting one of these scripts to a new question — see the pitfalls box at the end of §7.

---

## 1. Root-finding & linear systems — Chapter01

| Pattern | Canonical code |
| --- | --- |
| Solve `A x = C` (backslash vs `LinearSolve`) | [`Chapter01/Julia lecture/01p04SolvingLinearAndNonlinearEquations/01_LinearEquations.jl`](Chapter01/Julia%20lecture/01p04SolvingLinearAndNonlinearEquations/01_LinearEquations.jl) |
| Newton-Raphson on **one** equation, by hand + `NonlinearSolve` | [`.../02_SingleNonlinearEquation.jl`](Chapter01/Julia%20lecture/01p04SolvingLinearAndNonlinearEquations/02_SingleNonlinearEquation.jl) — hand loop `x = x - f(x)/dfdx(x)` at lines 27-31, `NonlinearProblem` at lines 37-40 |
| Newton-Raphson on a **system** of 2+ equations (Jacobian) | [`.../03_NonlinearSystem.jl`](Chapter01/Julia%20lecture/01p04SolvingLinearAndNonlinearEquations/03_NonlinearSystem.jl) line 60 |

**Already-worked variants of the system case** (same recipe, different equations — good for seeing how much changes when the question changes):
- `Homework/Day02/day2_code.jl` (and identical `day2_homework.ipynb`) — solves `x³+y³=1`, `x²+y²=8` by hand Newton-Raphson with an explicit `Jacobian(x,y)`, then checks with `NonlinearSolve`, then plots the two zero-contours.
- `Assignment 1/Solution.jl` lines 44-79 (Question 1) — same skeleton, circle/cubic system `(x+1)²+(y-5)²=8`, `(y-4)²+x³=0`, tried from two different initial guesses.

**If the question differs:** rewrite the residual function (`FunctionStore`/`uandv`) and its `Jacobian` to match the new equations (Jacobian is just the 2×2 (or n×n) matrix of partial derivatives — do it by hand); keep the update loop `delta = -J\F; x += delta[1]; y += delta[2]` unchanged; change the initial guess if Newton diverges (read it off a contour/zero-crossing plot first, as all three worked examples do).

---

## 2. Optimization — finding a minimum — Chapter02

| Pattern | Canonical code |
| --- | --- |
| First-order library methods (Descent/Momentum/Nesterov/Adam via `Optimization.jl`) | [`Chapter02/Julia lecture/02p01BasicOptimizationInJulia/`](Chapter02/Julia%20lecture/02p01BasicOptimizationInJulia) — one file per test function |
| Second-order library methods (Newton/BFGS/LBFGS) | [`Chapter02/Julia lecture/02p02BasicOptimizationInJuliaContinued/`](Chapter02/Julia%20lecture/02p02BasicOptimizationInJuliaContinued) |
| Hand-coded GD / Momentum / Nesterov, 1-D | [`02p03GradientBasedMethods/01_Quadratic_GDMomentumNesterov.jl`](Chapter02/Julia%20lecture/02p03GradientBasedMethods/01_Quadratic_GDMomentumNesterov.jl) — momentum update at lines 36-41 (`v = beta*v - alpha*∇f(x); x += v`), Nesterov at 46-51 (gradient evaluated at `x + beta*v`) |
| Hand-coded GD / Momentum / Nesterov, 2-D + contour/arrow plots | [`02p04MultivariateExamples/`](Chapter02/Julia%20lecture/02p04MultivariateExamples) |
| Newton's method by hand (1-D and N-D via `x -= H\g`), and the "converges to a maximum" pitfall | [`02p05SecondOrderMethods/01_Newton1D_AndPitfall.jl`](Chapter02/Julia%20lecture/02p05SecondOrderMethods/01_Newton1D_AndPitfall.jl), [`02_NewtonND_QuadraticHimmelblau.jl`](Chapter02/Julia%20lecture/02p05SecondOrderMethods/02_NewtonND_QuadraticHimmelblau.jl) lines 31 & 80 |

**Homework/Assignment worked variants:**
- `Homework/Day03/one_dimensional_optimization.jl` — same function optimized 3 ways: exact (solve `f'(x)=0`), `Optimization.jl` Descent, and hand-coded GD; explicitly discusses when GD converges to the wrong stationary point depending on the starting guess.
- `Assignment 1/Solution.jl` lines 84-204 (Question 2) — 2-D function with **two** local minima; hand GD run from four different starting points to show which basin each one falls into.

**If the question differs:** swap in the new `f(x)`/gradient/Hessian (write the gradient/Hessian by hand — every worked example does); to test "does it find the right minimum," just change `x0` and rerun, comparing against a contour plot; to add momentum to a plain-GD answer, add one `v` variable and the two-line update above.

---

## 3. Curve fitting / linear least squares — Chapter03

| Pattern | Canonical code |
| --- | --- |
| Constant model `f=a0`, analytic derivative + hand GD | [`03p01LinearLeastSquaresGradientDescent/01_ConstantModelGradientDescent.jl`](Chapter03/Julia%20lecture/03p01LinearLeastSquaresGradientDescent/01_ConstantModelGradientDescent.jl) — loss at line 28, derivative at line 38, GD loop at lines 41-45 |
| Line model `f=a0+a1*x`, loss-surface contour + `Optimization.jl` | [`.../02_LineModelOptimization.jl`](Chapter03/Julia%20lecture/03p01LinearLeastSquaresGradientDescent/02_LineModelOptimization.jl) |
| **Normal equations** — any polynomial degree, no iteration | [`Chapter03/Julia/03p02LinearLeastSquaresNormalEquations.jl`](Chapter03/Julia/03p02LinearLeastSquaresNormalEquations.jl) lines 54-58: build `V` with `hcat(ones, x, x.^2, ...)`, solve `a = (V'*V)\(V'*y)` in one line |

**Homework worked variants:**
- `Homework/Day10-11/homework_day10and11.jl` lines 24-27 — the **general-degree** version: `V = hcat([xdata .^ k for k in 0:degree]...)`, so changing `degree = 8` to any other number is the entire edit needed for "fit a degree-N polynomial."
- `Homework/Day04-05/homework_day4and5.jl` — cubic model fit by hand-coded Descent **and** Momentum **and** Adam side by side (`Optimisers.Descent`, `Optimisers.Momentum`, `Optimisers.Adam` all called on the same `OptimizationProblem`), comparing final MSE.

**If the question differs:** for a different polynomial degree, only the `degree`/`hcat` columns change (see the general-degree line above); for a non-polynomial linear-in-coefficients basis (e.g. `[1, sin(x), cos(x)]`), just change what goes into `hcat(...)` — the `(V'V)\(V'y)` line never changes; for a **nonlinear**-in-coefficients model (e.g. `a1*exp(a2*x)`), normal equations don't apply — use §4's Zygote-gradient fit instead.

---

## 4. Differentiation & automatic differentiation — Chapter04

| Pattern | Canonical code |
| --- | --- |
| Symbolic differentiation | [`04p01DifferentiationInJulia/01_SymbolicDifferentiation.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/01_SymbolicDifferentiation.jl) |
| Numerical differencing (forward/backward/central/complex-step) | [`.../02_NumericalDifferentiation.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/02_NumericalDifferentiation.jl) |
| `ForwardDiff.derivative` / `Zygote.gradient` | [`.../03_AutomaticDifferentiation.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/03_AutomaticDifferentiation.jl) lines 19 & 22 |
| AD inside Newton-Raphson / gradient descent | [`.../04_ADForRootFindingAndDescent.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/04_ADForRootFindingAndDescent.jl) lines 15 & 40 |
| Multi-argument functions: `ForwardDiff` (one arg at a time) vs `Zygote` (several at once) | [`.../05_FunctionsWithParameters.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/05_FunctionsWithParameters.jl) |
| Fit a **nonlinear** model `a1*(1-exp(a2*x))` via `Zygote.gradient` + hand GD | [`.../06_ParameterEstimationFit.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/06_ParameterEstimationFit.jl) lines 36 & 41 |
| Jacobians of vector functions | [`.../07_JacobiansAndNonlinearSystems.jl`](Chapter04/Julia%20lecture/04p01DifferentiationInJulia/07_JacobiansAndNonlinearSystems.jl) — `ForwardDiff.jacobian`/`Zygote.jacobian` at lines 35-37 |
| **Hand-rolled forward-mode AD** (`Dual` numbers from scratch) | [`04p02AutoDiffImplementationForwardMode/01_DualNumbersUnivariate.jl`](Chapter04/Julia%20lecture/04p02AutoDiffImplementationForwardMode/01_DualNumbersUnivariate.jl) (univariate struct + operator overloads), [`02_MultivariablePartials.jl`](Chapter04/Julia%20lecture/04p02AutoDiffImplementationForwardMode/02_MultivariablePartials.jl) (partials: seed the wanted variable's `.∂` with `1.0`, the rest with `0.0`) |
| Applying hand-rolled `Dual` AD to real tasks | [`04p03ApplicationOfForwardMode/01_DualNewtonRaphson.jl`](Chapter04/Julia%20lecture/04p03ApplicationOfForwardMode/01_DualNewtonRaphson.jl), [`02_DualNonlinearSystem.jl`](Chapter04/Julia%20lecture/04p03ApplicationOfForwardMode/02_DualNonlinearSystem.jl) (Jacobian from Dual evals), [`03_DualRegression.jl`](Chapter04/Julia%20lecture/04p03ApplicationOfForwardMode/03_DualRegression.jl) (fit via a Dual-number derivative) |

**Homework worked variants:**
- `Homework/Day06/solution_homework_day06.jl` — differentiates `exp(-x²)cos(x)` three ways (analytical, `ForwardDiff`, hand `Dual`) and prints all three side by side — the cleanest template for a "compare differentiation methods" question.
- `Homework/Day07/Solution_Homeworkday07.jl` — `Zygote.gradient` **and** a hand `Dual` struct used to get both partial derivatives of a 2-variable function.
- `Homework/Day08-09/homework_day8and9.jl` — Newton-Raphson (hand + `NonlinearSolve`) to solve for a **parameter** `b` inside `f(x,b)=0` rather than for `x` itself — read this one if the question asks you to back out a constant instead of a root.

**If the question differs:** to support a new elementary operation in the hand-rolled `Dual` type, overload it the same way as the existing ones — `import Base: <op>; <op>(a::Dual) = Dual(<value formula>, <chain-rule derivative formula>)`; to get a different partial derivative from the multivariate version, change which variable is seeded with `∂ = 1.0`.

---

## 5. Implicit / inverse problems (differentiate through a `solve()`) — Ch04§4 + Ch05§4-5

| Pattern | Canonical code |
| --- | --- |
| Sweep an input, re-solving a linear system each time, to locate a target | [`Chapter04/Julia lecture/04p04ParameterEstimationLinearProblems/01_CircuitSolveAndSweep.jl`](Chapter04/Julia%20lecture/04p04ParameterEstimationLinearProblems/01_CircuitSolveAndSweep.jl) |
| Minimize `loss(p) = (solve(p) - target)^2`, hand GD then `Optimization.jl` (Adam), differentiating **through** the solve | [`.../02_OptimizeThroughSolve.jl`](Chapter04/Julia%20lecture/04p04ParameterEstimationLinearProblems/02_OptimizeThroughSolve.jl) |
| Same pattern, BVP version: unknown material constant `k` | [`Chapter05/Julia lecture/05p04ParameterEstimationBVP01/01_ParameterSweepScan.jl`](Chapter05/Julia%20lecture/05p04ParameterEstimationBVP01/01_ParameterSweepScan.jl) (brute-force sweep), [`02_OptimizeWithAdam.jl`](Chapter05/Julia%20lecture/05p04ParameterEstimationBVP01/02_OptimizeWithAdam.jl) (Adam + `AutoForwardDiff`) |
| Fit an unknown **function** `w(x)=a0+a1*x` from measured data, where the model is only defined by solving a BVP | [`Chapter05/Julia lecture/05p05ParameterEstimationBVP02.jl`](Chapter05/Julia%20lecture/05p05ParameterEstimationBVP02.jl) |

**Assignment worked variant:**
- `Assignment 2/Solution.jl` Question 1 (lines 34-110) — identical BVP-inverse-for-a-scalar recipe (`solve_q1(p)` rebuilds and solves the tridiagonal system for a trial `p`, `loss_q1(p) = (y0(p)-target)^2`), including a worked note on resonances (the system going ill-conditioned near certain `p`) that made the naive sweep unreliable.
- `Assignment 2/Solution.jl` Question 3 (lines 203-308) — the same "loss solves the system" skeleton, but the unknown is now a whole **neural network** `q(x)` instead of a couple of scalars (see §7's NN+physics row — this is the same idea, just fewer/more parameters).

**If the question differs:** keep the skeleton `function loss(p, target); <rebuild the system with p>; <solve it>; return (result - target)^2; end`; only what's inside "rebuild the system with p" changes (a scalar parameter → few coefficients → a whole NN, in increasing order of the section-7 examples); pick `AutoForwardDiff` when the unknown is a handful of scalars, `AutoZygote` once it's a Lux network's `ComponentArray`.

---

## 6. Boundary value problems — Chapter05

| Pattern | Canonical code |
| --- | --- |
| Uniform / non-uniform heating, tridiagonal finite-difference assembly | [`05p01IntroBoundaryValueProblem/01_UniformHeatingTridiagonal.jl`](Chapter05/Julia%20lecture/05p01IntroBoundaryValueProblem/01_UniformHeatingTridiagonal.jl), [`02_NonUniformHeating.jl`](Chapter05/Julia%20lecture/05p01IntroBoundaryValueProblem/02_NonUniformHeating.jl) (only the RHS `C` changes) |
| BVP with **both** first- and second-derivative terms, dense vs sparse solve | [`05p02BoundaryValueProblem01/01_DenseFiniteDifference.jl`](Chapter05/Julia%20lecture/05p02BoundaryValueProblem01/01_DenseFiniteDifference.jl), [`02_SparseSolveSpeedup.jl`](Chapter05/Julia%20lecture/05p02BoundaryValueProblem01/02_SparseSolveSpeedup.jl) |
| **Nonlinear** BVP (can't be written as a matrix) via `NonlinearProblem` | [`05p03NonlinearBoundaryValueProblem.jl`](Chapter05/Julia%20lecture/05p03NonlinearBoundaryValueProblem.jl) line 62 |

All of these assemble `Tridiagonal(dl, d, du)` — see `A = Tridiagonal(dl, d, du)` in every file above — and differ only in what goes in the three diagonals and the boundary rows.

**If the question differs:**
- **Dirichlet** boundary (`y(a)=v`): that row is `d[1]=1.0; C[1]=v` (or the last row for the other end).
- **Neumann** boundary (`y'(a)=0`): one-sided difference, e.g. `dl[n-1]=-1.0; d[n]=1.0; C[n]=0.0` for a right-end condition (see `Chapter05/Julia lecture/05p01IntroBoundaryValueProblem/01_UniformHeatingTridiagonal.jl` lines 41-43), or `d[1]=-1.0; du[1]=1.0; C[1]=0.0` for a left-end condition (see `Assignment 2/Solution.jl` line 44).
- **Interior equation changes** (different ODE): re-derive the three coefficients multiplying `y[i-1], y[i], y[i+1]` from the central-difference formulas for `y''` (`1/Δ², -2/Δ², 1/Δ²`) and `y'` (`-1/(2Δ), 0, 1/(2Δ)`), then add whatever the ODE's own `y` coefficient contributes to the middle term (see `Assignment 2/Solution.jl` lines 47-52 for a worked `p*y` term).
- **Nonlinear in the unknown:** don't build a matrix at all — write a residual function `F(y,p)` (each entry is "LHS − RHS", zero at the solution) and hand it to `NonlinearProblem`, exactly as in `05p03NonlinearBoundaryValueProblem.jl`.

---

## 7. Neural networks — Chapter06

| Pattern | Canonical code |
| --- | --- |
| Activation functions, single neuron, hidden layer, hand GD, 2-input contour — **all by hand, no Lux** | [`06p01IntroductionToNeuralNetwork/01-05`](Chapter06/Julia%20lecture/06p01IntroductionToNeuralNetwork) — the 2-input grid-and-contour pattern is at `05_MultipleInputs.jl` lines 19-22: `grid = [[x1i,x2i] for x1i in x1range, x2i in x2range]` then `contour!(ax, x1range, x2range, only.(f.(grid)))` |
| Lux `Chain`/`Dense` workflow: linear model → tanh → hidden layer → hand GD → `Optimization.jl` Adam | [`06p02NeuralNetworkWithJuliaPart1/01-05`](Chapter06/Julia%20lecture/06p02NeuralNetworkWithJuliaPart1) — canonical training recipe (`OptimizationProblem`→`solve`) at `05_AdamOptimizer.jl` lines 49 & 53 |
| Deeper/wider `Chain(Dense(...))` architectures | [`06p03NeuralNetworkWithJuliaPart2/01_DeeperNetworkArchitectures.jl`](Chapter06/Julia%20lecture/06p03NeuralNetworkWithJuliaPart2/01_DeeperNetworkArchitectures.jl) |
| Train/validation split — **interpolation** (random split) | [`.../02_TrainValidationInterpolation.jl`](Chapter06/Julia%20lecture/06p03NeuralNetworkWithJuliaPart2/02_TrainValidationInterpolation.jl) line 36: `training_indices = shuffle(eachindex(xdata))[1:round(Int, 0.5*N_SAMPLES)]` |
| Train/validation split — **extrapolation** (train on `x < x0`, test beyond) | [`.../03_Extrapolation.jl`](Chapter06/Julia%20lecture/06p03NeuralNetworkWithJuliaPart2/03_Extrapolation.jl) |

**Homework worked variants (multi-input NN and NN+physics — the newest, most exam-relevant material):**
- `Homework/Day10-11/homework_day10and11.jl` — 1-node vs 8-node network, compared against the §3 normal-equations polynomial fit on the *same* data (good template for "fit with a polynomial and also a NN, compare").
- `Homework/Day12/solutionday12.jl` — **2-input** NN (`t_input = [x1data'; x2data']`, `Dense(2 => 16, tanh)`), trained with `Optimization.jl`/Adam, visualized with `tricontourf!` at line 64 — the right template whenever the question has two independent inputs and you need a filled-contour comparison against scattered data (not `lines!`, which only works for a single, sorted, single-valued `x`).
- `Homework/Day12/classexerciseday12.jl` — extends the 2-input idea so the **initial condition `x(0)`** itself is the second input (`t_input=[xzero';tdata']`), trained jointly across several datasets that share the same underlying ODE but different initial conditions — the template for "one network, many related datasets."
- `Homework/Day13-14/SolutionHomework1314.jl` — loads four datasets that share the same `tdata`/`xdata`/`xzero` columns but differ in one parameter (initial condition `x(0) ∈ {0, 1, 2, 0.5}`, one CSV per value: `DataHomeworkDay13a-d.csv`) and overlays them on one plot before the multi-initial-condition network training that follows (same idea as `classexerciseday12.jl`, §7 above). This is the template for "load and compare N structurally-identical datasets that differ in one parameter."
- `Chapter06/Julia/0605.jl` and `Chapter06/Jupyter Notebooks/06p05NeuralNetworkScientificModelsLinear.ipynb` — **NN combined with a physics solve**: the network's output becomes the right-hand side of a `Tridiagonal`/`LinearSolve` system, and the loss compares the *solved* result to data (not the network's raw output). This is the newest chapter material (no PDF/slides filed yet).
- `Assignment 2/Solution.jl` Question 3 (lines 203-308) — the fullest worked version of NN+physics: a `Chain(Dense(1=>16,tanh), Dense(16=>16,tanh), Dense(16=>1))` produces a source term `q(x)`, which feeds into a `Tridiagonal` BVP solve each loss evaluation, trained with `AutoForwardDiff` (not Zygote — differentiating through the `LinearSolve.solve` call needs it).

**If the question differs:**
- **More inputs** (2 → 3+): stack rows the same way — `t_input = [x1'; x2'; x3']` — and change only the first layer's arity, `Dense(3 => ...)`.
- **Different data files:** every NN script here reads a `DataXX.csv` with `CSV.read`; swap the filename/path only.
- **Want interpolation vs extrapolation:** swap the split rule in §7's train/validation row (random `shuffle` vs a threshold on `x`) — nothing else in the training loop changes.
- **NN's output feeds a physics solve instead of being the final answer:** reuse the §5 "solve inside loss" skeleton, but the "few coefficients" become the NN's `ComponentArray(parameters)`, and you must use `AutoForwardDiff` or `AutoZygote` — whichever the solve library you're differentiating through actually supports (check `0605.jl`/Assignment 2 Q3 for the one that already works).

### Common pitfalls when adapting any of the above (checklist: **L-I-S-S-C**)
1. **L**oss must compare the prediction to the **target `y`**, never to an input `x` — the single most common bug.
2. Match the plot to the **I**nput dimension: a 2-input model is a surface — use `contourf!`/`tricontourf!` or colored `scatter!`, not `lines!`.
3. `lines!` needs **S**orted, single-valued `x` — otherwise use `scatter!` or a sorted grid.
4. Check input/output **S**hapes: features×samples in (e.g. `xdata'`), `1×N` out.
5. **C**heck the loss curve first — loss trending down but the picture still wrong means a plotting bug, not a training bug.

**Extra pitfall when loading a series of similar datasets (a, b, c, d, ...):** if each dataset's block is built by copy-pasting the previous one (as in `Homework/Day13-14/SolutionHomework1314.jl`), it's easy to duplicate the extraction lines but forget to rename the source dataframe — e.g. `dtdata = dfa[:, :tdata]` left over instead of `dfd[:, :tdata]`. The tell: two "different" series turn out pixel-identical and one appears to have vanished from the plot (it's actually just been overplotted by the duplicate). Always check that the `dfX` in each extraction line matches the `dfX = CSV.read(...)` immediately above it.

---

## 8. Homework Day → PDF → Solution → Pattern cross-reference

| Day | Handout | Solution file | Chapter / pattern(s) exercised |
| --- | --- | --- | --- |
| 02 | [`Homework/Day02/HomeworkDay02.pdf`](Homework/Day02/HomeworkDay02.pdf) | `Homework/Day02/day2_code.jl` / `day2_homework.ipynb` | §1 — nonlinear system, Jacobian Newton-Raphson |
| 03 | [`Homework/Day03/HomeworkDay03.pdf`](Homework/Day03/HomeworkDay03.pdf) | `Homework/Day03/one_dimensional_optimization.jl` | §2 — 1-D optimization, exact vs hand GD vs library |
| 04-05 | [`Homework/Day04-05/HomeworkDays4and5.pdf`](Homework/Day04-05/HomeworkDays4and5.pdf) | `Homework/Day04-05/homework_day4and5.jl` | §3 — polynomial regression, Descent/Momentum/Adam compared |
| 06 | [`Homework/Day06/HomeworkDay06.pdf`](Homework/Day06/HomeworkDay06.pdf) | `Homework/Day06/solution_homework_day06.jl` | §4 — differentiation methods compared |
| 07 | [`Homework/Day07/HomeworkDay07.pdf`](Homework/Day07/HomeworkDay07.pdf) | `Homework/Day07/Solution_Homeworkday07.jl` (+ `draft.jl`, scratch) | §4 — Jacobian/partials via Zygote + hand Dual |
| 08-09 | [`Homework/Day08-09/HomeworkDays8and9.pdf`](Homework/Day08-09/HomeworkDays8and9.pdf) | `Homework/Day08-09/homework_day8and9.jl` | §4/§1 — Newton-Raphson for a parameter, not a root |
| 10-11 | [`Homework/Day10-11/HomeworkDays10and11.pdf`](Homework/Day10-11/HomeworkDays10and11.pdf) | `Homework/Day10-11/homework_day10and11.jl` | §3 + §7 — normal-equations polynomial vs 1-node/8-node NN |
| 12 | [`Homework/Day12/HomeworkDay12.pdf`](Homework/Day12/HomeworkDay12.pdf) | `Homework/Day12/solutionday12.jl` (+ `classexerciseday12.jl`, in-class run-up) | §7 — multi-input NN, `tricontourf!` |
| 13-14 | [`Homework/Day13-14/HomeworkDays13And14.pdf`](Homework/Day13-14/HomeworkDays13And14.pdf) | `Homework/Day13-14/SolutionHomework1314.jl` | §7 — comparing multiple same-shape datasets that differ in one parameter (`x(0)`), feeding a multi-initial-condition network |
