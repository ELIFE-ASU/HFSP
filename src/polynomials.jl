mutable struct StateSpace{N, M}
    StateSpace{N}() where N = new{N, 2^N}()
end
StateSpace(n::Int) = StateSpace{n}()

Base.length(space::StateSpace{N,M}) where {N, M} = M
Base.eltype(::StateSpace{N}) where N = SVector{N, F₂}
Base.IteratorEltype(::StateSpace) = Base.HasEltype()

@inline Base.in(n::Int, space::StateSpace{N,M}) where {N, M} = 1 ≤ n ≤ M

@inline function encode(::StateSpace{N}, state::SVector{N, F₂}) where N
    n = 0
    for x in state
        n = (n << 1) ⊻ x.n
    end
    n + 1
end

@inline function encode(space::StateSpace{N}, state::AbstractVector{F₂}) where N
    length(state) != N && ArgumentError("value is not in space")
    n = 0
    for x in state
        n = (n << 1) ⊻ x.n
    end
    n + 1
end

@inline function decode(space::StateSpace{N}, n::Int) where N
    !(n ∈ space) && DomainError("invalid encoding $n")
    n -= 1
    state = @MVector zeros(F₂, N)
    for i in Iterators.reverse(eachindex(state))
        state[i] = n & 1;
        n >>= 1
    end
    SVector(state)
end

Base.iterate(space::StateSpace) = (decode(space, 1), 2)

function Base.iterate(space::StateSpace, n)
    if n in space
        decode(space, n), n + 1
    end
end

project(p::T) where {T <: GF.AbstractExtensionField} = T(p.coeffs[1])

sigma(T::Type{<:GF.PrimeField}) = [T(1)]

function sigma(T::Type{<:GF.AbstractExtensionField})
    gens = generators(T)
    [T(1); map(prod, combinations(gens))]
end

function rho(T::Type{<:GF.AbstractExtensionField}, i::Int)
    gens = generators(T)
    sum(Iterators.map(prod, combinations(gens, i)))
end

rho(T::Type{<:GF.AbstractExtensionField}) = T[rho(T, i) for i in 0:nvariables(T)]

abstract type CompletePolynomial{T<:GF.AbstractExtensionField} end

function CompletePolynomial(T::Type{<:GF.AbstractExtensionField}, coeff::AbstractVector{F₂})
    dot(coeff, sigma(T))
end

Base.rand(::Type{CompletePolynomial{T}}) where T = rand(T)

abstract type SymmetricPolynomial{T<:GF.AbstractExtensionField} end

function SymmetricPolynomial(T::Type{<:GF.AbstractExtensionField}, coeff::AbstractArray{F₂})
    dot(coeff, rho(T))
end

Base.rand(::Type{SymmetricPolynomial{T}}) where T = SymmetricPolynomial(T, rand(F, nvariables(T) + 1))

abstract type MajorityRule{T<:GF.AbstractExtensionField} end

mutable struct StrongMajorityRule{T} <: MajorityRule{T}
    n::Int
    p::T
    function StrongMajorityRule{T}(n::Int) where {T <: GF.AbstractExtensionField}
        coeff = Dict(
            0 => F₂[0, 0, 0, 0, 0, 0, 0, 0, 0],
            1 => F₂[0, 1, 0, 0, 0, 0, 0, 0, 0],
            2 => F₂[0, 0, 1, 0, 0, 0, 0, 0, 0],
            3 => F₂[0, 0, 1, 0, 0, 0, 0, 0, 0],
            4 => F₂[0, 0, 0, 1, 1, 0, 0, 0, 0],
            5 => F₂[0, 0, 0, 1, 1, 0, 0, 0, 0],
            6 => F₂[0, 0, 0, 0, 1, 0, 0, 0, 0],
            7 => F₂[0, 0, 0, 0, 1, 0, 0, 0, 0],
            8 => F₂[0, 0, 0, 0, 0, 1, 1, 1, 1],
        )
        new{T}(n, SymmetricPolynomial(T, coeff[n]))
    end
end
StrongMajorityRule{T}() where {T <: GF.AbstractExtensionField} = StrongMajorityRule{T}(nvariables(T))

Base.copy(p::StrongMajorityRule) = StrongMajorityRule(p.n, copy(p.p))

mutable struct WeakMajorityRule{T} <: MajorityRule{T}
    n::Int
    p::T
    function WeakMajorityRule{T}(n::Int) where {T <: GF.AbstractExtensionField}
        coeff = Dict(
            0 => F₂[0, 0, 0, 0, 0, 0, 0, 0, 0],
            1 => F₂[0, 1, 0, 0, 0, 0, 0, 0, 0],
            2 => F₂[0, 1, 1, 0, 0, 0, 0, 0, 0],
            3 => F₂[0, 0, 1, 0, 0, 0, 0, 0, 0],
            4 => F₂[0, 0, 1, 0, 1, 0, 0, 0, 0],
            5 => F₂[0, 0, 0, 1, 1, 0, 0, 0, 0],
            6 => F₂[0, 0, 0, 1, 1, 0, 0, 0, 0],
            7 => F₂[0, 0, 0, 0, 1, 0, 0, 0, 0],
            8 => F₂[0, 0, 0, 0, 1, 0, 0, 0, 1],
        )
        new{T}(n, SymmetricPolynomial(T, coeff[n]))
    end
end
WeakMajorityRule{T}() where {T <: GF.AbstractExtensionField} = WeakMajorityRule{T}(nvariables(T))

(p::MajorityRule)(args...) = p.p(args...)

Base.copy(p::WeakMajorityRule) = WeakMajorityRule(p.n, copy(p.p))

project(p::T) where {T <: MajorityRule} = T(p.n - 1)

mutable struct RandomChoice{F, G}
    p::Float64
    f::F
    g::G
    function RandomChoice(p::Real, f::F, g::G) where {F, G}
        !(zero(p) ≤ p ≤ one(p)) && error("probability must be in [0.0, 1.0]")
        new{F, G}(p, f, g)
    end
end

(r::RandomChoice)(args...) = r()(args...)
(r::RandomChoice)() = rand() < r.p ? r.f : r.g

Base.copy(p::RandomChoice) = RandomChoice(p.p, copy(f), copy(g))

nvariables(T::Type{<:GF.AbstractGaloisField}) = depth(T)
nvariables(::T) where {T <: GF.AbstractGaloisField} = depth(T)
nvariables(r::RandomChoice{T}) where T = depth(T)

project(r::RandomChoice) = RandomChoice(r.p, project(r.f), project(r.g))

negate(p::GF.AbstractExtensionField) = one(p) + p

space(::T) where {T <: GF.AbstractExtensionField} = space(T)
space(T::Type{<:GF.AbstractExtensionField}) = StateSpace(nvariables(T))

tablecols(v) = tuple(map(Symbol ∘ repr, v)...)
tablecols(p::GF.AbstractExtensionField) = tablecols(generators(p))

function table(p::GF.AbstractExtensionField; name = Symbol(repr(p)))
    cols = tuple(tablecols(p)..., name)
    table = NamedTuple{cols, NTuple{length(cols), F₂}}[]
    for state in space(p)
        f = p(state)
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

nvariables(p::HFSPPolynomial) = nvariables(p.g), nvariables(p.f[1])

tablecols(prefix, v) = tuple(map(Symbol ∘ (x -> prefix * x) ∘ repr, v)...)
tablecols(prefix, p::GF.AbstractExtensionField) = tablecols(prefix, generators(p))
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
