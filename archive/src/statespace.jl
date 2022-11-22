struct StateSpace{N, M}
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
