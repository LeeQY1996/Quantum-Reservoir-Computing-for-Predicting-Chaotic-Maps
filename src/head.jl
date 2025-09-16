push!(LOAD_PATH,"../package/QuantumCircuits_demo/src","../package/VQC_demo/src")
using QuantumCircuits, QuantumCircuits.Gates
using VQC, VQC.Utilities
using Flux:train!
using Flux
using Random
using Statistics
using StatsBase
using LinearAlgebra
using SparseArrays


include("auxiliary.jl")
include("logisticmap.jl")
include("core.jl")
include("circuitQR.jl")