__precompile__(false)
module HFSP

export Model, ninputs, inputspace, space, maxneighbors
export update!, update, trajectory!, trajectory, ensemble, finalensemble
export setup
export model0, model1, model2, model3
export collate, ensembleplot, runplot

export StateSpace, encode, decode
export CompletePolynomial, SymmetricPolynomial
export MajorityRule, StrongMajorityRule, WeakMajorityRule
export RandomChoice
export HFSPPolynomial
export project, nvariables, negate, space, tablecols
export table

export F, F₂
export generators, generatornames, topoly, extend, base_name_indices, substitute, depth

include("core.jl")

end
