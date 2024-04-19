module GANExample

using Flux: Flux, gradient, Optimise.update!, params, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, NNlib.gelu, NNlib.elu, NNlib.sigmoid,
Chain, @functor, train!, cpu, gpu, softplus, params, BatchNorm, normalise
using Random: Random
using Distributions: Distributions, MixtureModel, Normal, MvNormal
using Makie: Makie, Axis
using CUDA

include("gan.jl")

end # module GANExample
