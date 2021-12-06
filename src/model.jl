mutable struct Model{G <: AbstractGraph, P <: OffsetVector{<:HFSPPolynomial}}
    g::G
    p::P
    function Model(g::AbstractGraph, p::OffsetVector{HFSPPolynomial})
        N, M = maxdegree(g), last(DP.nvariables(last(p)))
        if !iszero(M) && N != M
            error("provided polynomial must have as many variables as the graph's largest degree: expected $N, got $M")
        end
        new{typeof(g), typeof(p)}(g, p)
    end
end

function Model(g::AbstractGraph, p::HFSPPolynomial)
    ps = HFSPPolynomial[p]
    n = max(maxdegree(g), last(DP.nvariables(p)))
    for i in 2:n
        push!(ps, project(ps[i-1]))
    end
    Model(g, OffsetVector(reverse(ps), 1:n))
end

Base.length(m::Model) = nv(m.g)
ninputs(m::Model) = nvariables(last(m.p).g)
inputspace(m::Model) = StateSpace(ninputs(m))
space(m::Model) = StateSpace(length(m))
maxdegree(g::LightGraphs.AbstractGraph) = maximum(LightGraphs.degree(g))
maxdegree(m::Model) = maxdegree(m.g)

Base.copy(m::Model) = Model(copy(m.g), deepcopy(m.p))

function (m::Model)(inputs::AbstractVector{F₂}, state::AbstractVector{F₂})
    result = similar(state)
    neighborhood = zeros(F₂, maxdegree(m))
    for i in eachindex(state)
        ns = neighbors(m.g, i)
        p = m.p[length(ns)]
        fill!(neighborhood, 0)
        copyto!(neighborhood, state[ns])
        result[i] = apply(p, inputs, state[i:i], neighborhood)
    end
    result
end

function update!(dst::AbstractVector{F₂}, m::Model, input::AbstractVector{<:AbstractVector{F₂}}, n::Int)
    @views for t in 1:n
        i = mod(t, length(input))
        dst[:] = m(input[i], dst)
    end
    dst
end

function update!(dst::AbstractVector{F₂}, m::Model, input::Function, n::Int)
    @views for t in 1:n
        dst[:] = m(input(t), dst)
    end
    dst
end

function update!(dst::AbstractVector{F₂}, m::Model, input::AbstractVector{F₂}, n::Int)
    @views for t in 1:n
        dst[:] = m(input, dst)
    end
    dst
end

function update(m::Model, init, args...; kwargs...)
    update!(copy(init), m, args...; kwargs...)
end

function trajectory!(dst::AbstractMatrix{F₂}, m::Model, input::AbstractVector{<:AbstractVector{F₂}})
    @views for t in 1:size(dst,2)-1
        i = mod1(t, length(input))
        dst[:,t+1] = m(input[i], dst[:,t])
    end
    dst
end

function trajectory!(dst::AbstractMatrix{F₂}, m::Model, input::Function)
    @views for t in 1:size(dst,2)-1
        dst[:,t+1] = m(input(t), dst[:,t])
    end
    dst
end

function trajectory!(dst::AbstractMatrix{F₂}, m::Model, input::AbstractVector{F₂})
    @views for t in 1:size(dst,2)-1
        dst[:,t+1] = m(input, dst[:,t])
    end
    dst
end

function trajectory!(dst::AbstractMatrix{F₂}, m::Model, init, args...; kwargs...)
    dst[:,1] = init
    trajectory!(dst, m, args...; kwargs...)
end

function trajectory(m::Model, init, input, t, args...; kwargs...)
    trajectory!(Matrix{F₂}(undef, length(m), t + 1), m, init, input, args...; kwargs...)
end

function ensemble(m::Model, input, t, n, args...; kwargs...)
    ensemble = Array{F₂}(undef, length(m), t, n)
    @views for i in 1:n
        ensemble[:,1,i] = rand(F, length(m))
        trajectory!(ensemble[:,:,i], m, input, args...; kwargs...)
    end
    ensemble
end

function ensemble(m::Model, init::AbstractVector{F₂}, input, t, n, args...; kwargs...)
    ensemble = Array{F₂}(undef, length(m), t, n)
    @views for i in 1:n
        trajectory!(ensemble[:,:,i], m, init, input, args...; kwargs...)
    end
    ensemble
end

function finalensemble(m::Model, init::AbstractVector{F₂}, input, t, n, args...; kwargs...)
    ensemble = Array{F₂}(undef, length(m), n)
    @views for i in 1:n
        ensemble[:,i] = rand(F, length(m))
        update!(ensemble[:,i], m, t, args...; kwargs...)
    end
    ensemble
end

function finalensemble(m::Model, init::AbstractVector{F₂}, input, t, n, args...; kwargs...)
    ensemble = Array{F₂}(undef, length(m), n)
    @views for i in 1:n
        ensemble[:,i] = init
        update!(ensemble[:,i], m, input, t, args...; kwargs...)
    end
    ensemble
end

#  polynomials(g::AbstractGraph) = g |> LightGraphs.degree |> maximum |> polynomials
#  polynomials(n::Int) = first(extend(F, n))

function setup(filename::AbstractString = datadir("exp_pro", "sam.jld2"))
    @unpack graph = load(filename)
    graph
end

function model0(graph; p=0.01)
    g = CompletePolynomial(@SVector F₂[0, 1])
    h = CompletePolynomial(@SVector F₂[0, 1])

    f00 = RandomChoice(p, 1, 0)
    f01 = 1
    f10 = 0
    f11 = 1

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model1(graph; p=0.01)
    n = maxdegree(graph)

    g = CompletePolynomial(@SVector F₂[0, 1])
    h = CompletePolynomial(@SVector F₂[0, 1])

    f00 = RandomChoice(p, 1, WeakMajorityRule{n}())
    f01 = 1
    f10 = 0
    f11 = 1

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model2(graph; p = 0.01)
    n = maxdegree(graph)

    g = CompletePolynomial(@SVector F₂[0, 1])
    h = CompletePolynomial(@SVector F₂[0, 1])

    f00 = RandomChoice(p, 1, WeakMajorityRule{n}())
    f01 = 1
    f10 = WeakMajorityRule{n}()
    f11 = 1

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model3(graph; p=0.01, q=0.01)
    n = maxdegree(graph)

    g = CompletePolynomial(@SVector F₂[0, 1])
    h = CompletePolynomial(@SVector F₂[0, 1])

    f00 = RandomChoice(p, 1, WeakMajorityRule{n}())
    f01 = 1
    f10 = 0
    f11 = RandomChoice(q, 0, 1)

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function collate(ens::AbstractArray{F₂,3})
   μ = mean(Array{Int}(ens); dims=1)[1,:,:]
   M, N = size(μ)
   DataTable(run=repeat(1:N; inner=M), timestep=repeat(1:M; outer=N), expression=vec(μ))
end

function collate(ens::AbstractArray{F₂,2})
   μ = mean(Array{Int}(ens); dims=1)[1,:]
   N = length(μ)
   DataTable(timestep=1:N, expression=vec(μ))
end

function ensembleplot(args...; title="Ensemble", kwargs...)
    @vlplot(
        x = :timestep,
        title = title,
        width = 454,
        height = 347,
    ) +
    @vlplot(
        {
            :line,
            clip = true,
        },
        y = {
            :expression,
            scale = {
                domain = [0,1],
            },
        },
        detail = {
            "run:o",
            legend = false,
        },
        opacity = {
            value = 0.1
        },
        color = {
            value = "black"
        }
    ) +
    @vlplot(
        mark = {
            :errorband,
            extent = :stddev
        },
        y = {
            :expression,
            title="expression level (fraction of cells)"
        },
        color = {
            value = "red"
        }
    ) +
    @vlplot(
        :line,
        y = "mean(expression)",
        strokeWidth = {
            value = 3,
        },
        color = {
            value = "black"
        }
    )
end

function runplot(args...; title = "Trajectory", kwargs...)
    @vlplot(
        x = :timestep,
        title = title,
        width = 454,
        height = 347,
    ) +
    @vlplot(
        :line,
        y = :expression,
        stroke = {
            "run:o",
            legend=false,
        }
    )
end

ensembleplot(ens::AbstractArray{F₂,3}, args...; kwargs...) = collate(ens) |> ensembleplot(args...; kwargs...)
runplot(ens::AbstractArray{F₂,2}, args...; kwargs...) = collate(ens) |> runplot(args...; kwargs...)
