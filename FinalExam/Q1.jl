using CSV
using DataFrames
using CairoMakie
using Statistics
using Random
using Lux
using Zygote
using Optimization
using OptimizationOptimisers
using ComponentArrays

data_path = joinpath(@__DIR__, "DataHomeworkDay12.csv")
df = CSV.read(data_path, DataFrame)