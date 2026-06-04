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
- Coursework artifacts, homework notebooks, CSV data, and lecture PDFs.

## Repository Structure

| Path | Contents |
| --- | --- |
| [`Chapter01/`](Chapter01) | Introduction, Julia basics, plotting, and equation-solving lecture PDFs and slides. |
| [`Chapter02/`](Chapter02) | Optimization lectures, slides, gradient-based methods, multivariate examples, and second-order methods. |
| [`class 2/`](class%202) | Julia class demos for linear systems, nonlinear solving, plotting, homework, and generated PNG outputs. |
| [`class 3/`](class%203) | Optimization notebook and Julia demos for gradient descent and momentum-style methods. |
| [`Assignment 1/`](Assignment%201) | Assignment PDF and `Q3Data.csv`. |
| [`Homework/`](Homework) | Homework Day 02 handout. |
| [`JuliaPrograms/`](JuliaPrograms) | Introductory Julia program examples. |

## Visual Overview

| Function behavior | Nonlinear system contours | Circuit sweep |
| --- | --- | --- |
| <img src="class%202/classdemo1_output.png" alt="Function plot from class 2" width="290"> | <img src="class%202/day2_check.png" alt="Contour plot for nonlinear system" width="290"> | <img src="class%202/nonlinear_check.png" alt="Current i5 versus source voltage V1" width="290"> |

These figures are generated from the Julia class demos and show the main numerical workflow: define a mathematical model, visualize it, solve or optimize it, and inspect the result.

## Getting Started

Install Julia, then instantiate the project environment from the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Run selected scripts from PowerShell:

```powershell
julia "Chapter01\01p02.jl"
julia "class 2\nonlinear.jl"
julia "class 3\classdemo3.jl"
```

Open notebooks directly in VS Code, Jupyter, or another Julia-compatible notebook environment:

```text
class 2/day2_homework.ipynb
class 3/02p01BasicOptimizationInJulia.ipynb
```

## Main Dependencies

The Julia environment is defined in [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml). Major packages include:

- `CairoMakie` and `Plots` for visualization.
- `LinearSolve` and `NonlinearSolve` for equation solving.
- `Optim`, `Optimization`, `OptimizationOptimJL`, and `OptimizationOptimisers` for numerical optimization.
- `ForwardDiff` and `Zygote` for automatic differentiation.
- `CSV` and `DataFrames` for tabular data workflows.

## Notes

- Lecture materials are stored as PDFs for reference and review.
- Notebooks may be larger than the scripts because they can include embedded outputs.
- The Peking University seal is included from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Peking_University.svg), which identifies the original source as `pku.edu.cn` and notes that trademark or insignia restrictions may apply.
