struct BPoly
    var::PolyVar{true}
    zero::Union{F₂, BPoly}
    one::Union{F₂, BPoly}
end

function compile(p::Polynomial{true, F₂})
    if isempty(variables(p))
        convert(F₂, p)
    else
        v = first(variables(p))
        zero = subs(p, v => F₂(0))
        one = subs(p, v => F₂(1))
        BPoly(v, compile(zero), compile(one))
    end
end

eval(p::F₂, args::F₂...) = p
eval(p::F₂, args::AbstractVector{F₂}) = p
@inline function eval(p::BPoly, arg::F₂, args::F₂...)
    if iszero(arg)
        eval(p.zero, args...)
    else
        eval(p.one, args...)
    end
end
@inline function eval(p::BPoly, args::AbstractVector{F₂})
    if iszero(args[1])
        eval(p.zero, @view args[2:end])
    else
        eval(p.one, @view args[2:end])
    end
end
(p::F₂)(args::F₂...) = p
(p::F₂)(args::AbstractVector{F₂}) = p
(p::BPoly)(args::F₂...) = eval(p, args...)
(p::BPoly)(args::AbstractVector{F₂}) = eval(p, args)

MP.nvariables(p::BPoly) = length(variables(p))
MP.nvariables(p::Integer) = 0
MP.nvariables(p::F₂) = 0

MP.variables(p::BPoly) = sort!(union([p.var], variables(p.zero), variables(p.one)); rev=true)
MP.variables(p::Integer) = PolyVar{true}[]
MP.variables(p::F₂) = PolyVar{true}[]

space(p::BPoly) = StateSpace(nvariables(p))

project(p::F₂, args...) = p
function project(p::BPoly, var::PolyVar{true})
    if p.var == var
        p.zero
    else
        BPoly(p.var, project(p.zero), project(p.one))
    end
end
project(p::BPoly) = project(p, last(variables(p)))

todp(p::F₂) = p
todp(p::BPoly) = p.var * todp(p.one) + (F(1) + p.var) * todp(p.zero)

Base.show(io::IO, p::BPoly) = show(io, todp(p))

@inline function generators(n::Int)
    if n == 0
        PolyVar{true}[]
    else
        @polyvar x[1:n]
        x
    end
end

function sigma(gens::Vector{PolyVar{true}})
    if isempty(gens)
        Polynomial{true, F₂}[1]
    else
        Polynomial{true, F₂}[1; map(prod, combinations(gens))]
    end
end

function rho(gens::Vector{PolyVar{true}}, i::Int)
    sum(Iterators.map(prod, combinations(gens, i)))
end

function rho(gens::Vector{PolyVar{true}})
    Polynomial{true, F₂}[rho(gens, i) for i in 0:length(gens)]
end

abstract type CompletePolynomial{N} end

function CompletePolynomial(coeff::AbstractVector{F₂})
    N = floor(Int, log2(length(coeff)))
    gens = generators(N)
    compile(dot(coeff, sigma(gens)))
end

Base.rand(::Type{CompletePolynomial{N}}) where N = CompletePolynomial(rand(F₂, 2^N))

abstract type SymmetricPolynomial{N} end

function SymmetricPolynomial(coeff::AbstractArray{F₂})
    gens = generators(length(coeff) - 1)
    compile(dot(coeff, rho(gens)))
end

Base.rand(::Type{SymmetricPolynomial{N}}) where N = SymmetricPolynomial(rand(F₂, N + 1))

abstract type MajorityRule{N} end

(p::MajorityRule)(args...) = p.p(args...)

mutable struct StrongMajorityRule{N} <: MajorityRule{N}
    p::BPoly
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
MP.nvariables(::StrongMajorityRule{N}) where N = N
MP.nvariables(::Type{StrongMajorityRule{N}}) where N = N
project(::StrongMajorityRule{N}) where N = StrongMajorityRule{N-1}()

Base.copy(p::StrongMajorityRule) = StrongMajorityRule(copy(p))

mutable struct WeakMajorityRule{N} <: MajorityRule{N}
    p::BPoly
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
MP.nvariables(::WeakMajorityRule{N}) where N = N
MP.nvariables(::Type{WeakMajorityRule{N}}) where N = N
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

(r::RandomChoice)() = rand() < r.p ? r.f : r.g
(r::RandomChoice)(args...) = r()(args...)

MP.nvariables(r::RandomChoice) = max(nvariables(r.f), nvariables(r.g))

Base.copy(p::RandomChoice) = RandomChoice(p.p, copy(f), copy(g))

project(r::RandomChoice) = RandomChoice(r.p, project(r.f), project(r.g))

negate(p::Polynomial{true, F₂}) = one(p) + p

space(p::Polynomial{true, F₂}) = StateSpace(nvariables(p))

tablecols(v) = tuple(map(Symbol ∘ repr, v)...)
tablecols(p::Polynomial{true, F₂}) = tablecols(variables(p))

function table(p::Polynomial{true, F₂}; name = Symbol(repr(p)))
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

function (p::HFSPPolynomial)(garg, harg)
    g = p.g(garg)
    h = p.h(harg)
    f = if iszero(g)
        iszero(h) ? p.f[1] : p.f[2]
    else
        iszero(h) ? p.f[3] : p.f[4]
    end
end

(p::HFSPPolynomial)(garg, harg, farg) = p(garg, harg)(farg)

Base.copy(p::HFSPPolynomial) = HFSPPolynomial(copy(p.g), copy(p.h), copy(p.f))

project(p::HFSPPolynomial) = HFSPPolynomial(p.g, p.h, project.(p.f))

MP.nvariables(p::HFSPPolynomial) = nvariables(p.g), maximum(nvariables.(p.f))

tablecols(prefix, v) = tuple(map(Symbol ∘ (x -> prefix * x) ∘ repr, v)...)
tablecols(prefix, p::BPoly) = tablecols(prefix, variables(p))
tablecols(p::HFSPPolynomial) = (tablecols("g_", p.g)..., tablecols("h_", p.h)..., tablecols("f_", p.f[1])...)

function table(p::HFSPPolynomial; name = Symbol("f(x)"))
    cols = (tablecols(p)..., name)
    table = NamedTuple{cols, NTuple{length(cols), F₂}}[]
    for (input, h, state) in product(space(p.g), space(p.h), space(p.f[1]))
        f = p(input, h, state)
        push!(table, NamedTuple{cols}(tuple(input..., h..., state..., f)))
    end
    DataTable(table)
end
