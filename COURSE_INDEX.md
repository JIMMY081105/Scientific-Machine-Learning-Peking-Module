# Course Index

This index groups the repository material by study activity so the lecture PDFs, scripts, notebooks, and assignments are easier to navigate. For an exam-prep cheat sheet organized by *problem type* instead of by folder, see [`PROBLEM_INDEX.md`](PROBLEM_INDEX.md).

## Lecture Notes

Each chapter separates its material into `Jupyter Notebooks/`, `PDFs/`, `Julia lecture/`, and `Julia/` (smaller standalone examples). Within `Julia lecture/`, each notebook-derived lesson is broken into a per-lesson subfolder of small scripts split by goal/strategy (each runs on its own, with a `00_README.md` listing them); single-goal lessons remain a single script. Chapters that use datasets keep the `DataXX.csv` files at the chapter root, referenced from scripts and notebooks via relative paths. Click a notebook below to open it directly.

### [`Chapter01/`](Chapter01) — motivation, Julia basics, plotting, solving linear/nonlinear equations
- [01p01IntroductionAndMotivation.ipynb](<Chapter01/Jupyter Notebooks/01p01IntroductionAndMotivation.ipynb>)
- [01p02JuliaProgrammingBasics.ipynb](<Chapter01/Jupyter Notebooks/01p02JuliaProgrammingBasics.ipynb>)
- [01p03PlottingFunctionsInJulia.ipynb](<Chapter01/Jupyter Notebooks/01p03PlottingFunctionsInJulia.ipynb>)
- [01p04SolvingLinearAndNonlinearEquations.ipynb](<Chapter01/Jupyter Notebooks/01p04SolvingLinearAndNonlinearEquations.ipynb>)

### [`Chapter02/`](Chapter02) — optimization: first-order/second-order library methods, hand-coded gradient methods, multivariate examples
- [02p01BasicOptimizationInJulia.ipynb](<Chapter02/Jupyter Notebooks/02p01BasicOptimizationInJulia.ipynb>)
- [02p02BasicOptimizationInJuliaContinued.ipynb](<Chapter02/Jupyter Notebooks/02p02BasicOptimizationInJuliaContinued.ipynb>)
- [02p03GradientBasedMethods.ipynb](<Chapter02/Jupyter Notebooks/02p03GradientBasedMethods.ipynb>)
- [02p04MultivariateExamples.ipynb](<Chapter02/Jupyter Notebooks/02p04MultivariateExamples.ipynb>)
- [02p05SecondOrderMethods.ipynb](<Chapter02/Jupyter Notebooks/02p05SecondOrderMethods.ipynb>)

### [`Chapter03/`](Chapter03) — linear least squares: gradient descent and normal equations
- [03p01LinearLeastSquaresGradientDescent.ipynb](<Chapter03/Jupyter Notebooks/03p01LinearLeastSquaresGradientDescent.ipynb>)
- [03p02LinearLeastSquaresNormalEquations.ipynb](<Chapter03/Jupyter Notebooks/03p02LinearLeastSquaresNormalEquations.ipynb>)

### [`Chapter04/`](Chapter04) — differentiation, forward-mode AD, parameter estimation through a linear solve
- [04p01DifferentiationInJulia.ipynb](<Chapter04/Jupyter Notebooks/04p01DifferentiationInJulia.ipynb>)
- [04p02AutoDiffImplementationForwardMode.ipynb](<Chapter04/Jupyter Notebooks/04p02AutoDiffImplementationForwardMode.ipynb>)
- [04p03ApplicationOfForwardMode.ipynb](<Chapter04/Jupyter Notebooks/04p03ApplicationOfForwardMode.ipynb>)
- [04p04ParameterEstimationLinearProblems.ipynb](<Chapter04/Jupyter Notebooks/04p04ParameterEstimationLinearProblems.ipynb>)

### [`Chapter05/`](Chapter05) — boundary value problems, nonlinear BVPs, parameter estimation for BVPs
- [05p01IntroBoundaryValueProblem.ipynb](<Chapter05/Jupyter Notebooks/05p01IntroBoundaryValueProblem.ipynb>)
- [05p02BoundaryValueProblem01.ipynb](<Chapter05/Jupyter Notebooks/05p02BoundaryValueProblem01.ipynb>)
- [05p03NonlinearBoundaryValueProblem.ipynb](<Chapter05/Jupyter Notebooks/05p03NonlinearBoundaryValueProblem.ipynb>)
- [05p04ParameterEstimationBVP01.ipynb](<Chapter05/Jupyter Notebooks/05p04ParameterEstimationBVP01.ipynb>)
- [05p05ParameterEstimationBVP02.ipynb](<Chapter05/Jupyter Notebooks/05p05ParameterEstimationBVP02.ipynb>)

### [`Chapter06/`](Chapter06) — neural networks, from a hand-built single neuron up to multivariate networks and NN+physics (`06p05`)
- [06p01IntroductionToNeuralNetwork.ipynb](<Chapter06/Jupyter Notebooks/06p01IntroductionToNeuralNetwork.ipynb>)
- [06p02NeuralNetworkWithJuliaPart1.ipynb](<Chapter06/Jupyter Notebooks/06p02NeuralNetworkWithJuliaPart1.ipynb>)
- [06p03NeuralNetworkWithJuliaPart2.ipynb](<Chapter06/Jupyter Notebooks/06p03NeuralNetworkWithJuliaPart2.ipynb>)
- [06p04TrainingMultivariateNeuralNetworks.ipynb](<Chapter06/Jupyter Notebooks/06p04TrainingMultivariateNeuralNetworks.ipynb>)
- [06p05NeuralNetworkScientificModelsLinear.ipynb](<Chapter06/Jupyter Notebooks/06p05NeuralNetworkScientificModelsLinear.ipynb>)

## Class Work

- `class 2/` and `class 3/` contain the raw in-class live-coding demos from the second and third class sessions (plotting, nonlinear systems, circuit equations, gradient descent/momentum) that led up to the Day02/Day03 homework, including a raw copy of the optimization notebook: [class 3/02p01BasicOptimizationInJulia.ipynb](<class 3/02p01BasicOptimizationInJulia.ipynb>). They mostly duplicate — in rougher form — material now polished into `Chapter01/`/`Chapter02/`. `class 2/` also carries its own `Project.toml`/`Manifest.toml` environment snapshot (see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). The actual Day02/Day03 homework *deliverables* now live in `Homework/Day02/` and `Homework/Day03/` (see below), not here.

## Coursework

- `Assignment 1/` — nonlinear system solving (Newton-Raphson + `NonlinearSolve`), 2-D gradient descent with multiple local minima, and nonlinear curve fitting (exponential decay / damped oscillation) by hand-coded gradient descent.
- `Assignment 2/` — inverse/parameter-estimation BVP (finite-difference + gradient descent for an unknown coefficient), a multi-input neural network trained across several material-conductivity datasets, and a neural network combined with a linear BVP solve to recover a source term.
- `Homework/` is organized as one subfolder per class day, each holding the handout PDF (where one exists) alongside the corresponding solution script/notebook and any data/figures it needs:
  - `Day02/` — nonlinear system of two equations (Newton-Raphson + `NonlinearSolve` + zero-contour plot): [day2_homework.ipynb](<Homework/Day02/day2_homework.ipynb>). Ch01 material.
  - `Day03/` — 1-D optimization: hand-coded gradient descent vs `Optimization.jl`, with the local-max/min pitfall. Ch02 material.
  - `Day04-05/` — polynomial regression via hand-coded Descent/Momentum/Adam, comparing convergence and fit. Ch03 material.
  - `Day06/` — differentiating `exp(-x^2)*cos(x)` three ways: analytical, `ForwardDiff`, hand-rolled Dual numbers. Ch04 material.
  - `Day07/` — partial derivatives / Jacobian of a two-variable function via `Zygote` and Dual numbers (`draft.jl` is an earlier scratch attempt at a related nonlinear-solve + `ForwardDiff` optimization, kept for reference). Ch04 material.
  - `Day08-09/` — Newton-Raphson to fit a parameter `b` in a nonlinear equation, both hand-coded and via `NonlinearSolve`. Ch04 material.
  - `Day10-11/` — polynomial fit (normal equations) vs. 1-node and 8-node neural networks on the same data. Ch06 material.
  - `Day12/` — multivariate neural network with the initial condition `x(0)` as a second input, trained across combined datasets (`classexerciseday12.jl` is the in-class run-up; `solutionday12.jl` is the submitted solution on `DataHomeworkDay12.csv`). Ch06 material.
- `Archive/` — two superseded/incomplete scratch scripts kept for reference rather than deleted: `combine.jl` (early draft of the Chapter03 quadratic-regression demo) and `findmodel.jl` (incomplete — references an undefined `N_SAMPLES` — early draft of the same, using `Archive/Datasets/class 3/Data02.csv`).

## Julia Environment

- `Project.toml` and `Manifest.toml` define the root Julia environment.
- `class 2/Project.toml` and `class 2/Manifest.toml` preserve the class-specific environment snapshot.
