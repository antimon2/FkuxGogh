module Config

export Configuration

Base.@kwdef struct Configuration
    batch_size::Int = 1
    iteration::Int = 5000
    lr::Float32 = 1f0
    lam::Float32 = 0.05
    width::Int = 300
    height::Int = 300
    outputdir::AbstractString = "_output"
end

shape(c::Configuration) = (c.width, c.height, 3, c.batch_size)  # WHCN (Flux.jl's ordering)
function Base.getproperty(c::Configuration, name::Symbol)
    if name === :shape
        shape(c)
    else
        getfield(c, name)
    end
end

end # module
