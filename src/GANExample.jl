module GANExample

using Flux: Flux, gradient, Optimise.update!, params, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, NNlib.gelu, NNlib.elu, NNlib.sigmoid,
Chain, @functor, train!, cpu, gpu, softplus, params, BatchNorm, normalise
using Zygote: Zygote
using Optimisers: Optimisers, setup, update, destructure
using Random: Random
using Distributions: Distributions, MixtureModel, Normal, MvNormal
using Makie: Makie, Axis
using GLMakie
using CUDA

include("gan.jl")
include("utils.jl")

function main()
    train_gan()
end

end # module GANExample
