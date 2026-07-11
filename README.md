<div align="center">
  <img src="docs/assets/peking-university.svg" alt="Peking University seal" width="110">
  <h1>Scientific Machine Learning - Peking Module</h1>
  <p>
    Lecture notes, Julia demonstrations, notebooks, assignments, and generated figures
    for a scientific machine learning study module.
  </p>
</div>

## Overview

This repository collects the materials used while studying scientific machine learning with a Julia-based workflow. The content starts with Julia programming fundamentals, then moves into plotting, linear and nonlinear systems, and numerical optimization.

Core topics include:

- Julia syntax, arrays, functions, types, and basic scripting.
- Plotting and numerical visualization with CairoMakie and Plots.
- Linear systems, nonlinear equations, Newton-Raphson iterations, and contour analysis.
- Optimization workflows, including gradient-based and second-order methods.
- Linear least-squares modeling with gradient descent and normal equations.
- Coursework artifacts, homework notebooks, CSV data, and lecture PDFs.

## Repository Structure

| Path | Contents |
| --- | --- |
| [`Chapter01/`](Chapter01) | Introduction, Julia basics, plotting, and equation-solving materials organized into `Jupyter Notebooks`, `PDFs`, and `Julia`. |
| [`Chapter02/`](Chapter02) | Optimization materials organized into `Jupyter Notebooks`, `PDFs`, and `Julia`, including gradient-based and second-order methods. |
| [`Chapter03/`](Chapter03) | Linear least-squares materials organized into `Jupyter Notebooks`, `PDFs`, and `Julia`. |
| [`class 2/`](class%202) | Julia class demos for linear systems, nonlinear solving, plotting, homework, and generated PNG outputs. |
| [`class 3/`](class%203) | Optimization notebook and Julia demos for gradient descent and momentum-style methods. |
| [`Assignment 1/`](Assignment%201) | Assignment PDF and `Q3Data.csv`. |
| [`Homework/`](Homework) | Homework Day 02, Day 03, and Days 4-5 handouts, plus one-dimensional optimization code and figures. |
| [`JuliaPrograms/`](JuliaPrograms) | Introductory Julia program examples. |

For a study-oriented navigation guide, see [`COURSE_INDEX.md`](COURSE_INDEX.md).

## Visual Overview

| Optimization landscape | Algorithm comparison | Rosenbrock valley |
| --- | --- | --- |
| <img src="https://raw.githubusercontent.com/JIMMY081105/Scientific-Machine-Learning-Peking-Module/main/docs/assets/notebook-output-06.png" alt="Quadratic optimization contour field" width="290"> | <img src="https://raw.githubusercontent.com/JIMMY081105/Scientific-Machine-Learning-Peking-Module/main/docs/assets/notebook-output-09.png" alt="Optimization trajectories over a contour field" width="290"> | <img src="https://raw.githubusercontent.com/JIMMY081105/Scientific-Machine-Learning-Peking-Module/main/docs/assets/notebook-output-11.png" alt="Rosenbrock optimization landscape with trajectories" width="290"> |

These notebook-derived figures give the repository a more scientific visual summary: objective landscapes, contour geometry, and optimizer trajectories for gradient-based methods.

## Getting Started

Install the local tools first:

- [Julia](https://julialang.org/install/) for running scripts and notebooks.
- [Visual Studio Code](https://code.visualstudio.com/download) with the [Julia extension](https://code.visualstudio.com/docs/languages/julia), which provides the integrated REPL, plot pane, code completion, debugging, and Julia workspace tools.
- The [Jupyter extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for `.ipynb` notebook editing and execution. The extension also installs its common notebook helper extensions, including Jupyter Keymap, Notebook Renderers, Cell Tags, and Slide Show support.

Install the recommended VS Code extensions from PowerShell:

```powershell
code --install-extension julialang.language-julia
code --install-extension ms-toolsai.jupyter
```

Download the Julia packages for the repository environment from the repository root:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

The same setup can also be run from the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The `class 2/` folder has its own Julia environment snapshot. Instantiate it separately when running those examples:

```powershell
julia --project="class 2" -e "using Pkg; Pkg.instantiate()"
```

Run selected scripts from PowerShell:

```powershell
julia "Chapter01\Julia\01p02.jl"
julia "class 2\nonlinear.jl"
julia "class 3\classdemo3.jl"
```

Open notebooks directly in VS Code, Jupyter, or another Julia-compatible notebook environment:

```text
Chapter02/Jupyter Notebooks/02p01BasicOptimizationInJulia.ipynb
Chapter03/Jupyter Notebooks/03p02LinearLeastSquaresNormalEquations.ipynb
```

## Main Dependencies

The Julia environment is defined in [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml). Major packages include:

- `CairoMakie` and `Plots` for visualization.
- `LinearSolve` and `NonlinearSolve` for equation solving.
- `Optim`, `Optimization`, `OptimizationOptimJL`, and `OptimizationOptimisers` for numerical optimization.
- `ForwardDiff` and `Zygote` for automatic differentiation.
- `CSV` and `DataFrames` for tabular data workflows.

For environment setup and generated-output notes, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Notes

- Lecture materials are stored as PDFs for reference and review.
- Notebooks may be larger than the scripts because they can include embedded outputs.
- The Peking University seal is included from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Peking_University.svg), which identifies the original source as `pku.edu.cn` and notes that trademark or insignia restrictions may apply.
