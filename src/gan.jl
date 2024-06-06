struct GAN
    z_dim::Integer
    generator::Any
    discriminator::Any
end

function train_gan(; set_up = construct_training_setup(), training_log_sample_size = 1000)
    gan = setup_gan(set_up)
    generator = gan.generator
    discriminator = gan.discriminator
    fixed_ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, training_log_sample_size) # fixed noise samples for computing loss
    fixed_data = set_up.dataset[:, 1:training_log_sample_size]
    generator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, generator)
    discriminator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, discriminator)
    losses = Vector{Float64}()
    for epoch in 1:set_up.training_config.n_epochs
        println("Epoch $epoch")
        ii = 0
        for mini_batch in set_up.data_batch_iterator
            num_samples = size(mini_batch)[2]
            if ii % (set_up.training_config.time_difference_k + 1) != 0
                # update discriminator
                ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, num_samples) # noise samples
                fake_data = generator(ϵ)
                # explicit style of gradient computation
                grads = Zygote.gradient(discriminator) do model
                    loss = get_discriminator_loss(model)
                    loss(mini_batch, fake_data)
                end
                ∇m = grads[1]
                discriminator_optimizer_setup, discriminator = Optimisers.update(discriminator_optimizer_setup, discriminator, ∇m)
            else
                # update generator
                ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, num_samples) # noise samples
                # explicit style of gradient computation
                grads = Zygote.gradient(generator) do model
                    loss = get_generator_loss(model, discriminator)
                    loss(ϵ)
                end
                ∇m = grads[1]
                generator_optimizer_setup, generator = Optimisers.update(generator_optimizer_setup, generator, ∇m)
            end
            ii += 1
        end
        current_loss = (sum(log.(discriminator(fixed_data) .+ 1e-6)) 
        + sum(log.(1 .- discriminator(generator(fixed_ϵ)) .+ 1e-6))) / training_log_sample_size
        @info "loss: $(current_loss)"
        push!(losses, current_loss)
    end
    plot_loss_curve(losses)
    plot_generated_samples(generator; set_up, gan.z_dim)
end

function get_generator_loss(generator, discriminator)
    function loss(ϵ)
        generated_fake_data = generator(ϵ)
        # sum(log.(1 .- discriminator(generated_fake_data) .+ 1e-6)) / size(ϵ)[2] # According to Goodfellow et al., use log(G(ϵ)) instead for improved gradient signal
        - sum(log.(discriminator(generated_fake_data) .+ 1e-6)) / size(ϵ)[2]
    end
end

function get_discriminator_loss(discriminator)
    function loss(real_data, fake_data)
        - (sum(log.(discriminator(real_data) .+ 1e-6)) + sum(log.(1 .- discriminator(fake_data) .+ 1e-6))) / size(real_data)[2]
    end
end

function construct_training_setup()
    function decoder_gt(z)
        tanh.(1.5z)
    end

    rng = Random.MersenneTwister(1)

    training_config = (;
        optimizer = Optimisers.Adam(0.0001, (0.9, 0.999), 1.0e-8),
        n_epochs = 500,
        batchsize = 128,
        n_datapoints = 100_000,
        device = gpu,
        time_difference_k = 3, # difference of the update frequency between the generator and the discriminator
    )

    dims = (; dim_x = 1, dim_hidden = 32, dim_z = 1) # dim_x: data dimension dim_z: 
    # construct dataset
    # dataset = randn(rng, dims.dim_z, training_config.n_datapoints) |> decoder_gt |> training_config.device
    sample_distribution = MixtureModel(Normal, [(-3, 1), (3, 1)])
    dataset = rand(rng, sample_distribution, dims.dim_z, training_config.n_datapoints) |> training_config.device
    data_batch_iterator = Flux.Data.DataLoader(dataset; training_config.batchsize, shuffle = true, rng)

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