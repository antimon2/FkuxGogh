module FluxGogh

using Flux: Flux, Chain, Conv, relu
using LinearAlgebra
using Metalhead: Colorant, Metalhead, VGG19, channelview
using NNlib
using Images
# using Random
using Statistics: mean

loadmodel() = VGG19()

function preprocess(image::AbstractMatrix{<:Colorant})
    # Convert from CHW (Image.jl's channel ordering) to WHCN (Flux.jl's ordering)
    # and enforce Float32, as that seems important to Flux
    Float32.(permutedims(channelview(RGB.(image)), (3, 2, 1))[:, :, :, :] .* 255 .- 120)
end
preprocess(x) = x

loadimage(image) = preprocess(Metalhead.load(image))

saveimage(file, image::AbstractArray{<:Float32}) = save(file, clamp.(permutedims(image[:, :, :], (2, 1, 3)) .+ 120, 0, 255) ./ 255f0)

# function convertmodel(model::VGG19)
#     Relu(x) = relu.(x)
#     layers1 = Chain(model.layers[1], Conv(weight=model.layers[2].weight, bias=model.layers[2].bias))
#     layers2 = Chain(Relu, model.layers[3:4], Conv(weight=model.layers[5].weight, bias=model.layers[5].bias))
#     layers3 = Chain(Relu, model.layers[6:9], Conv(weight=model.layers[10].weight, bias=model.layers[10].bias))
#     layers4 = Chain(Relu, model.layers[11:14], Conv(weight=model.layers[15].weight, bias=model.layers[15].bias))
#     [layers1, layers2, layers3, layers4]
# end

function choosemids(model::VGG19, x)
    layers1 = Chain(model.layers[1], Conv(weight=model.layers[2].weight, bias=model.layers[2].bias))
    x1 = layers1(x)
    layers2 = Chain(x -> relu.(x), model.layers[3:4], Conv(weight=model.layers[5].weight, bias=model.layers[5].bias))
    x2 = layers2(x1)
    layers3 = Chain(x -> relu.(x), model.layers[6:9], Conv(weight=model.layers[10].weight, bias=model.layers[10].bias))
    x3 = layers3(x2)
    layers4 = Chain(x -> relu.(x), model.layers[11:14], Conv(weight=model.layers[15].weight, bias=model.layers[15].bias))
    x4 = layers4(x3)
    [x1, x2, x3, x4]
end

# function choosemids(model::Vector{<:Chain}, x)
#     _x = x
#     results = Vector{Any}(undef, length(model))
#     for i = 1:length(model)
#         _x = results[i] = model[i](_x)
#     end
#     results
# end

function get_matrix(y)
    width, height, ch_num, batch_size = size(y)
    y_reshaped = reshape(y, (width * height, ch_num, batch_size))
    # result = similar(y, (ch_num, ch_num, batch_size))
    # result = Array{eltype(y)}(undef, ch_num, ch_num, batch_size)
    # @inbounds for i = 1:batch_size
    #     @views LinearAlgebra.mul!(result[:, :, i], y_reshaped[:, :, i]', y_reshaped[:, :, i])
    # end
    # result = NNlib.batched_mul(NNlib.batched_transpose(y_reshaped), y_reshaped)
    result = NNlib.batched_mul(permutedims(y_reshaped, (2,1,3)), y_reshaped)
    # result = [dot(y_reshaped[:, i, k], y_reshaped[:, j, k]) for i=1:ch_num, j=1:ch_num, k=1:batch_size]
    result ./ Float32(width * height * ch_num)
end

# function get_matrix(y)
#     width, height, ch_num, batch_size = size(y)
#     y_reshaped = reshape(y, (width * height, ch_num, batch_size))
#     results = [
#         y_reshaped[:, :, i]' * y_reshaped[:, :, i]
#         for i = 1:batch_size
#     ]
#     cat(results...; dims=3) ./ Float32(width * height * ch_num)
# end

function generate_image(model, img_orig, img_style, width, nw, nh, max_iter, ir)
    img_gen = rand(Float32, (nw, nh, 3, 1)) .* 40 .- 20
    generate_image(model, img_orig, img_style, width, nw, nh, max_iter, ir, img_gen)
end

function generate_image(model, img_orig, img_style, width, nw, nh, max_iter, ir, img_gen)
    # _model = convertmodel(model)  # .|> Flux.gpu
    _model = model .|> Flux.gpu
    mids_orig = choosemids(model, img_orig) .|> Flux.gpu
    mids_style = choosemids(model, img_style)  # .|> Flux.gpu
    style_mats = get_matrix.(mids_style) .|> Flux.gpu  # == [get_matrix(y) for y in mids_orig]

    function loss(x, (y1, y2)::NTuple{2, Any})
        ŷ = choosemids(_model, x)
        l = length(ŷ)
        s1 = l - l ÷ 2 + 1
        L1 = sum(Flux.Losses.mse.(ŷ[s1:end], y1[s1:end]))
        L2 = mean(Flux.Losses.mse.(get_matrix.(ŷ), y2))
        0.005f0 * L1 + L2  # TODO: 係数を外部parameter化
    end

    _img_gen = img_gen |> Flux.gpu
    ps = Flux.params(_img_gen)
    opt = Flux.ADAM(1f0, (0.9f0, 0.999f0))
    for i in 1:5000
        # Flux.train!(loss, ps, [(_img_gen, (mids_orig, style_mats))], opt)
        # @info _img_gen
        gs = Flux.gradient(ps) do
            loss(_img_gen, (mids_orig, style_mats))
        end
        # @info gs
        Flux.update!(opt, ps, gs)
        # @info _img_gen
        if i % 100 == 0
            ŷ = choosemids(_model, _img_gen)
            l = length(ŷ)
            s1 = l - l ÷ 2 + 1
            L1 = Flux.Losses.mse.(ŷ[s1:end], mids_orig[s1:end])
            L2 = Flux.Losses.mse.(get_matrix.(ŷ), style_mats)
            L = 0.005f0 * sum(L1) + mean(L2)
            @info L1, L2, L
            saveimage("out/sample_$(lpad(i, 4, '0')).png", _img_gen |> Flux.cpu)
        end
    end
    # _img_gen
end

end # module
