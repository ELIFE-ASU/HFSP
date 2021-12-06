const Polynomial = DP.Polynomial{true, F₂}

function apply(p::Polynomial)
    if iszero(DP.nvariables(p))
        isempty(p.a) ? zero(F) : p.a[1]
    else
        p(StaticArrays.fill(0, DP.nvariables(p)))
    end
end
function apply(p::Polynomial, args::AbstractVector{F₂})
    δ = DP.nvariables(p) - length(args)
    if δ < zero(δ)
        p(args[1:end+δ])
    elseif δ > zero(δ)
        p([args; StaticArrays.fill(0,δ)])
    else
        p(args)
    end
end
apply(p::Polynomial, args::F₂...) = apply(p, collect(args))
apply(p::Polynomial, args::T...) where {T <: Integer} = apply(p, collect(F₂, args))
apply(p::F₂) = p
apply(p::F₂, args...) = apply(p)
apply(p::Integer) = convert(F₂, p)
apply(p::Integer, args...) = apply(p)

project(p::Polynomial) = subs(p, DP.variables(p)[end] => 0)
project(p::Real) = p
project(p::F₂) = p

DP.nvariables(::Real) = 0
DP.nvariables(::F₂) = 0

@inline function generators(n::Int)
    if n == 0
        DP.PolyVar{true}[]
    else
        DP.@polyvar x[1:n]
        x
    end
end

function sigma(gens::Vector{DP.PolyVar{true}})
    if isempty(gens)
        Polynomial[1]
    else
        Polynomial[1; map(prod, combinations(gens))]
    end
end

function rho(gens::Vector{DP.PolyVar{true}}, i::Int)
    sum(Iterators.map(prod, combinations(gens, i)))
end

rho(gens::Vector{DP.PolyVar{true}}) = Polynomial[rho(gens, i) for i in 0:length(gens)]

abstract type CompletePolynomial{N} end

function CompletePolynomial(coeff::AbstractVector{F₂})
    N = floor(Int, log2(length(coeff)))
    gens = generators(N)
    dot(coeff, sigma(gens))
end

Base.rand(::Type{CompletePolynomial{N}}) where N = CompletePolynomial(rand(F₂, 2^N))

abstract type SymmetricPolynomial{N} end

function SymmetricPolynomial(coeff::AbstractArray{F₂})
    gens = generators(length(coeff) - 1)
    dot(coeff, rho(gens))
end

Base.rand(::Type{SymmetricPolynomial{N}}) where N = SymmetricPolynomial(rand(F₂, N + 1))

abstract type MajorityRule{N} end

apply(p::MajorityRule, args...) = apply(p.p, args...)

mutable struct StrongMajorityRule{N} <: MajorityRule{N}
    p::Polynomial
end
function StrongMajorityRule{N}() where N
    (N < 1 || N > 8) && error("StrongMajorityRule{N} is only defined for 1 ≤ N ≤ 8")
    coeff = Dict(
        1 => F₂[0, 1],
        2 => F₂[0, 0, 1],
        3 => F₂[0, 0, 1, 0],
        4 => F₂[0, 0, 0, 1, 1],
        5 => F₂[0, 0, 0, 1, 1, 0],
        6 => F₂[0, 0, 0, 0, 1, 0, 0],
        7 => F₂[0, 0, 0, 0, 1, 0, 0, 0],
        8 => F₂[0, 0, 0, 0, 0, 1, 1, 1, 1],
    )
    StrongMajorityRule{N}(SymmetricPolynomial(coeff[N]))
end
DP.nvariables(::StrongMajorityRule{N}) where N = N
DP.nvariables(::Type{StrongMajorityRule{N}}) where N = N
project(::StrongMajorityRule{N}) where N = StrongMajorityRule{N-1}()

Base.copy(p::StrongMajorityRule) = StrongMajorityRule(copy(p))

mutable struct WeakMajorityRule{N} <: MajorityRule{N}
    p::Polynomial
end
function WeakMajorityRule{N}() where N
    (N < 1 || N > 8) && error("WeakMajorityRule{N} is only defined for 1 ≤ N ≤ 8")
    coeff = Dict(
        1 => F₂[0, 1],
        2 => F₂[0, 1, 1],
        3 => F₂[0, 0, 1, 0],
        4 => F₂[0, 0, 1, 0, 1],
        5 => F₂[0, 0, 0, 1, 1, 0],
        6 => F₂[0, 0, 0, 1, 1, 0, 0],
        7 => F₂[0, 0, 0, 0, 1, 0, 0, 0],
        8 => F₂[0, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    WeakMajorityRule{N}(SymmetricPolynomial(coeff[N]))
end
DP.nvariables(::WeakMajorityRule{N}) where N = N
DP.nvariables(::Type{WeakMajorityRule{N}}) where N = N
project(::WeakMajorityRule{N}) where N = WeakMajorityRule{N-1}()

Base.copy(p::WeakMajorityRule) = WeakMajorityRule(copy(p.p))

mutable struct RandomChoice{F, G}
    p::Float64
    f::F
    g::G
    function RandomChoice(p::Real, f::F, g::G) where {F, G}
        !(zero(p) ≤ p ≤ one(p)) && error("probability must be in [0.0, 1.0]")
        new{F, G}(p, f, g)
    end
end

apply(r::RandomChoice) = rand() < r.p ? r.f : r.g
apply(r::RandomChoice, args...) = apply(apply(r), args...)

DP.nvariables(r::RandomChoice) = max(DP.nvariables(r.f), DP.nvariables(r.g))

Base.copy(p::RandomChoice) = RandomChoice(p.p, copy(f), copy(g))

project(r::RandomChoice) = RandomChoice(r.p, project(r.f), project(r.g))

negate(p::Polynomial) = one(p) + p

space(p::Polynomial) = StateSpace(DP.nvariables(p))

tablecols(v) = tuple(map(Symbol ∘ repr, v)...)
tablecols(p::Polynomial) = tablecols(DP.variables(p))

function table(p::Polynomial; name = Symbol(repr(p)))
    cols = tuple(tablecols(p)..., name)
    table = NamedTuple{cols, NTuple{length(cols), F₂}}[]
    for state in space(p)
        f = p(state...)
        push!(table, NamedTuple{cols}(tuple(state..., f)))
    end
    DataTable(table)
end

mutable struct HFSPPolynomial{G, H, T}
    g::G
    h::H
    f::T
end

function apply(p::HFSPPolynomial, garg, harg)
    g = p.g(garg)
    h = p.h(harg)
    f = if iszero(g)
        iszero(h) ? p.f[1] : p.f[2]
    else
        iszero(h) ? p.f[3] : p.f[4]
    end
end

apply(p::HFSPPolynomial, garg, harg, farg) = apply(apply(p, garg, harg), farg)

Base.copy(p::HFSPPolynomial) = HFSPPolynomial(copy(p.g), copy(p.h), copy(p.f))

project(p::HFSPPolynomial) = HFSPPolynomial(p.g, p.h, project.(p.f))

DP.nvariables(p::HFSPPolynomial) = DP.nvariables(p.g), maximum(DP.nvariables.(p.f))

tablecols(prefix, v) = tuple(map(Symbol ∘ (x -> prefix * x) ∘ repr, v)...)
tablecols(prefix, p::Polynomial) = tablecols(prefix, generators(p))
tablecols(p::HFSPPolynomial) = (tablecols("g_", p.g)..., tablecols("h_", p.h)..., tablecols("f_", p.f[1])...)

function table(p::HFSPPolynomial; name = Symbol("f(x)"))
    cols = (tablecols(p)..., name)
    table = NamedTuple{cols, NTuple{length(cols), F₂}}[]
    for (input, h, state) in product(space(p.g), space(p.h), space(p.f[1]))
        f = apply(p, input, h, state)
        push!(table, NamedTuple{cols}(tuple(input..., h..., state..., f)))
    end
    DataTable(table)
end
