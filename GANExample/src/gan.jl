struct GAN
    z_dim::Integer
    generator::Any
    discriminator::Any
end

function construct_training_setup()
    function decoder_gt(z)
        tanh.(1.5z)
    end

    rng = Random.MersenneTwister(1)

    training_config = (;
        optimizer = Flux.ADAM(0.001),
        n_epochs = 200,
        batchsize = 128,
        n_datapoints = 100_000,
        device = gpu
    )

    dims = (; dim_x = 1, dim_hidden = 32, dim_z = 1) # dim_x: data dimension dim_z: 
    dataset = randn(rng, dims.dim_z, training_config.n_datapoints) |> decoder_gt |> training_config.device
    data_batch_iterator = Flux.Data.DataLoader(dataset; training_config.batchsize)

    (; rng, training_config, dims, dataset, data_batch_iterator)
end

function setup_gan(set_up = construct_training_setup(); generator = nothing, discriminator = nothing)
    discriminator = isnothing(discriminator) ? Chain(
        Dense(set_up.dims.dim_x, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, 1, sigmoid; init = glorot_uniform(set_up.rng)),
    ) : encoder
    generator = isnothing(generator) ? Chain(
        Dense(set_up.dims.dim_z, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_x; init = glorot_uniform(set_up.rng)),
    ) : generator
    GAN(set_up.dims.dim_z, generator, discriminator) |> set_up.training_config.device
end