const GF = GaloisFields
const DP = DynamicPolynomials
const MP = MultivariatePolynomials

const F = @GaloisField 2
const F₂ = GaloisFields.PrimeField{Int8,2}

Base.conj(x::F₂) = F(x)
Base.abs(x::F₂) = F(x)
Base.isless(x::F₂, y::F₂) = isless(x.n, y.n)
Base.convert(T::Type{<:Real}, x::F₂) = convert(T, x.n)
Base.convert(T::Type{F₂}, n::Type{<:Integer}) = F₂(n)
