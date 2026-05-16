# Course Index

This index groups the repository material by study activity so the lecture PDFs, scripts, notebooks, and assignments are easier to navigate.

## Lecture Notes

- `Chapter01/` covers motivation, Julia basics, plotting, and solving linear and nonlinear equations.
- `Chapter02/` covers optimization foundations, gradient-based methods, multivariate examples, and second-order methods.
- `Chapter03/` covers linear least-squares methods using gradient descent and normal equations.
- `Chapter04/` covers differentiation in Julia and forward-mode automatic differentiation.
- `Chapter05/` covers boundary value problems, nonlinear BVPs, and parameter estimation for BVPs.
- `Chapter06/` covers neural networks and their implementation in Julia.

Each chapter separates its material into `Jupyter Notebooks/`, `PDFs/`, `Julia lecture/`, and `Julia/` (smaller standalone examples). Within `Julia lecture/`, each notebook-derived lesson is broken into a per-lesson subfolder of small scripts split by goal/strategy (each runs on its own, with a `00_README.md` listing them); single-goal lessons remain a single script. Chapters that use datasets keep the `DataXX.csv` files at the chapter root, referenced from scripts and notebooks via relative paths.

## Class Work

- `class 2/` contains Julia demos for plotting, nonlinear systems, circuit equations, and a homework notebook.
- `class 3/` contains optimization notebooks and Julia scripts for descent-based methods.

## Coursework

- `Assignment 1/` contains the assignment sheet and supporting CSV data.
- `Homework/` contains additional homework material.

## Julia Environment

- `Project.toml` and `Manifest.toml` define the root Julia environment.
- `class 2/Project.toml` and `class 2/Manifest.toml` preserve the class-specific environment snapshot.
