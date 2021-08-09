using GaloisFields, StaticArrays
import MultivariatePolynomials, DynamicPolynomials

const GF = GaloisFields
const DP = DynamicPolynomials
const MP = MultivariatePolynomials

const F = @GaloisField 2
const F₂ = GaloisFields.PrimeField{Int8,2}

Base.conj(x::F₂) = x

const MinimalCoeff = @SVector [0, 1, 1]

generators(T::Type{<:GF.PrimeField}) = tuple()
generators(T::Type{<:GF.BinaryField}) = (GF.gen(T),)
generators(T::Type{<:GF.ExtensionField}) = promote.(generators(GF.basefield(T))..., GF.gen(T))
generators(::T) where {T <: GF.AbstractGaloisField} = generators(T)

generatornames(T::Type{<:GF.PrimeField}) = tuple()
generatornames(T::Type{<:GF.BinaryField}) = (GF.genname(T),)
generatornames(T::Type{<:GF.ExtensionField}) = (generatornames(GF.basefield(T))..., GF.genname(T))
generatornames(::T) where {T <: GF.AbstractGaloisField} = generatornames(T)

topoly(x::F₂) = x
topoly(x::GF.PrimeField) = F(x.n)
function topoly(x::T) where {T <: GF.BinaryField}
    v = DP.PolyVar{true}(string(GF.genname(T)))
    F(x.n & 0x1) + F(x.n >> 1) * v
end
function topoly(x::T) where {T <: GF.ExtensionField}
    v = DP.PolyVar{true}(string(GF.genname(T)))
    c₀, c₁ = x.coeffs
    if iszero(x)
        topoly(c₀)
    else
        topoly(c₀) + topoly(c₁) * v
    end
end

MP.print_coefficient(io::IO, x::GF.PrimeField) = print(io, x)

Base.show(io::IO, mime::MIME"text/latex", x::GF.AbstractExtensionField) = show(io, mime, topoly(x))
Base.show(io::IO, x::GF.AbstractExtensionField) = show(io, topoly(x))

function extend(T::Type{<:GF.PrimeField}, n::Int)
    if n < 1
        T, generators(T)
    else
        S, _ = GaloisField(T, Symbol("x[1]") => MinimalCoeff)
        extend(S, n - 1)
    end
end

function extend(T::Type{<:GF.AbstractExtensionField}, n::Int)
    if n < 1
        T, generators(T)
    else
        base, indices = base_name_indices(T)
        length(indices) != 1 && error("cannot extend field with generator $(GF.gen(T))")
        var = Symbol("$(base)[$(indices[1] + 1)]")
        S, _ = GaloisField(T, var => MinimalCoeff)
        extend(S, n - 1)
    end
end

base_name_indices(T::Type{<:GF.AbstractExtensionField}) = base_name_indices(GF.genname(T))
base_name_indices(s::Symbol) = base_name_indices(string(s))
function base_name_indices(s::AbstractString)
    splits = split(s, r"[\[,\]]\s*"; keepempty=false)
    splits[1], parse.(Int, splits[2:end])
end

(p::GF.AbstractExtensionField)(values::AbstractVector{F₂}) = substitute(p, reverse(values))

function substitute(p::GF.BinaryField, values::AbstractVector{F₂})
    F(p.n & 0x1) + F(p.n >> 1) * values[1]
end

function substitute(p::GF.ExtensionField, values::AbstractVector{F₂})
    c₀, c₁ = p.coeffs
    @views if iszero(values[1])
        substitute(c₀, values[2:end])
    else
        substitute(c₀, values[2:end]) + substitute(c₁, values[2:end])
    end
end

depth(::Type{<:GF.PrimeField}) = 0
depth(T::Type{<:GF.AbstractExtensionField}) = 1 + depth(GF.basefield(T))
depth(::T) where {T <: GF.AbstractGaloisField} = depth(T)

function Base.show(io::IO, F::Type{<:GF.ExtensionField})
    print(io, "𝔽[$(depth(F))]")
end