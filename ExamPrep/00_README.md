# Exam Prep — one folder, five files, open-book survival kit

## Panic protocol (do this when you read a question)

1. **Identify the level** (it tells you which tools you're allowed):
   - **Level 1** — no packages (except linear solvers `A\C` / LinearSolve). You must hand-code derivatives, Newton loops, gradient descent loops, Duals.
   - **Level 2** — AD allowed (ForwardDiff, Zygote) + linear solvers. Same loops as Level 1, but you may replace hand derivatives with `ForwardDiff.derivative` / `Zygote.gradient`.
   - **Level 3** — everything allowed (NonlinearSolve, Optimization.jl, Lux, ...).
2. **Identify the pattern** from the question wording, open the matching file:

| Question says... | Open | Section |
| --- | --- | --- |
| "using Dual numbers", "implement forward-mode AD", "differentiate without packages" | `01_Duals.jl` | all |
| "find the root", "solve f(x)=0", "Newton-Raphson", "find x where..." | `02_NewtonRaphson.jl` | pick your level |
| "find the minimum/maximum", "optimize", "gradient descent / momentum / Adam", "fit a model / curve to data" | `03_GradientDescent_And_Fitting.jl` | pick your level |
| "solve the linear system", "circuit", "boundary value problem", "temperature/beam ODE", "two equations two unknowns", "Himmelblau / contour" | `04_SolveEquations.jl` | pick your level |
| "neural network", "Lux", "train/predict", "multiple datasets", "initial condition as input" | `05_NeuralNetworks.jl` | Level 3 |

3. **Copy the skeleton, rename the variables, swap the function/data.** Do not invent new structure under pressure.
4. Before submitting a fit/plot answer, run the **L-I-S-S-C** checklist at the bottom of `05_NeuralNetworks.jl`.

## Cheap points to remember

- **Normal equations are Level 1 legal** (only `\` needed): `V = hcat(ones(N), x, x.^2); a = V'V\(V'y)` — no iteration, no packages. If a fitting question is linear in the coefficients and Level 1, this is the answer.
- **BVPs are Level 1/2 legal** — assembling `Tridiagonal(dl,d,du)` and solving with `\`/LinearSolve is "linear solvers".
- Every Newton/GD loop needs: an initial guess, a step (learning rate for GD), and ~a `for` loop with `global` on the updated variable (script top level).
- A minimum is where the derivative is zero → an optimization question can become a root-finding question (`f'(x)=0` via Newton) and vice versa.

## Where the originals live (for more context)

- Duals: `Chapter04/Julia lecture/04p02.../04p03...`, `Homework/Day06/`, `Homework/Day07/`
- Newton-Raphson: `Chapter01/Julia lecture/01p04...`, `Homework/Day02/`, `Homework/Day08-09/`, `Assignment 1/`
- Gradient descent & fitting: `Chapter02/Julia lecture/`, `Chapter03/`, `Homework/Day03/`, `Homework/Day04-05/`, `Assignment 1/` Q2-Q3
- Equations & BVPs: `Chapter01/`, `Chapter05/Julia lecture/`, `Assignment 2/` Q1
- Neural networks: `Chapter06/`, `Homework/Day10-11/`, `Homework/Day12/`, `Homework/Day13-14/`, `Assignment 2/` Q2-Q3

Full pattern-by-pattern index with line numbers: [`../PROBLEM_INDEX.md`](../PROBLEM_INDEX.md)
