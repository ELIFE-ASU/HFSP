struct Model{G <: AbstractGraph, P <: OffsetVector{<:HFSPPolynomial}}
    g::G
    p::P
    function Model(g::AbstractGraph, p::OffsetVector{HFSPPolynomial})
        N, M = maximum(LightGraphs.degree(g)), last(nvariables(last(p)))
        if N != M
            error("provided polynomial must have as many variables as the graph's largest degree: expected $N, got $M")
        end
        new{typeof(g), typeof(p)}(g, p)
    end
end

function Model(g::AbstractGraph, p::HFSPPolynomial)
    ps = HFSPPolynomial[p]
    _, n = nvariables(p)
    for i in 1:n
        push!(ps, project(ps[i]))
    end
    Model(g, OffsetVector(reverse(ps), 0:n))
end

Base.length(m::Model) = nv(m.g)
ninputs(m::Model) = nvariables(last(m.p).g)
inputspace(m::Model) = StateSpace(ninputs(m))
space(m::Model) = StateSpace(length(m))
maxneighbors(m::Model) = maximum(LightGraphs.degree(m.g))

function (m::Model)(inputs::AbstractVector{F₂}, state::AbstractVector{F₂})
    result = similar(state)
    neighborhood = zeros(F₂, maxneighbors(m))
    for i in eachindex(state)
        ns = neighbors(m.g, i)
        p = m.p[length(ns)]
        fill!(neighborhood, 0)
        copyto!(neighborhood, state[ns])
        result[i] = p(inputs, state[i:i], neighborhood)
    end
    result
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
        ensemble[:,1,i] = rand(F₂, length(m))
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

polynomials(g::AbstractGraph) = g |> LightGraphs.degree |> maximum |> polynomials
polynomials(n::Int) = first(extend(F, n))

function setup(filename::AbstractString = datadir("exp_pro", "sam.jld2"))
    @unpack graph = load(filename)

    G = polynomials(1)
    H = polynomials(graph)

    graph, G, H
end

function model0(graph, G, H)
    n = nvariables(H)

    g = CompletePolynomial(G, @SVector F₂[0, 1])
    h = CompletePolynomial(G, @SVector F₂[0, 1])

    f00 = RandomChoice(0.01, H(1), H(0))
    f01 = H(1)
    f10 = H(0)
    f11 = H(1)

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model1(graph, G, H)
    n = nvariables(H)

    g = CompletePolynomial(G, @SVector F₂[0, 1])
    h = CompletePolynomial(G, @SVector F₂[0, 1])

    f00 = RandomChoice(0.01, H(1), WeakMajorityRule{H}())
    f01 = H(1)
    f10 = H(0)
    f11 = H(1)

    Model(graph, HFSPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model2(graph, G, H)
    n = nvariables(H)

    g = CompletePolynomial(G, @SVector F₂[0, 1])
    h = CompletePolynomial(G, @SVector F₂[0, 1])

    f00 = RandomChoice(0.1, H(1), WeakMajorityRule{H}())
    f01 = H(1)
    f10 = WeakMajorityRule{H}()
    f11 = H(1)

    Model(graph, HSFPPolynomial(g, h, (f00, f01, f10, f11)))
end

function model3(graph, G, H)
    n = nvariables(H)

    g = CompletePolynomial(G, @SVector F₂[0, 1])
    h = CompletePolynomial(G, @SVector F₂[0, 1])
    f00 = RandomChoice(0.01,  H(1), WeakMajorityRule{H}())
    f01 = H(1)
    f10 = H(0)
    f11 = RandomChoice(0.01, H(0), H(1))

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

function ensembleplot()
    @vlplot(
        x = :timestep,
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
        stroke = {
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
            extent = :ci
        },
        y = {
            :expression,
            title="expression level (fraction of cells)"
        }
    ) +
    @vlplot(
        :line,
        y = "mean(expression)"
    )
end

function runplot()
    @vlplot(
        x = :timestep,
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

ensembleplot(ens::AbstractArray{F₂,3}) = collate(ens) |> ensembleplot()
runplot(ens::AbstractArray{F₂,2}) = collate(ens) |> runplot()
