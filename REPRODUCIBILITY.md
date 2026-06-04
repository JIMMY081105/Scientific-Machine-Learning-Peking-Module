# Reproducibility Notes

The repository includes Julia project files so examples can be run from a consistent package environment.

## Recommended Setup

From the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

For the class 2 environment snapshot:

```julia
using Pkg
Pkg.activate("class 2")
Pkg.instantiate()
```

## Generated Outputs

The checked-in PNG files under `class 2/` and `docs/assets/` are generated outputs used for review and README presentation. New temporary output should go into `figures/` or `output/`, which are ignored by Git unless intentionally added.
