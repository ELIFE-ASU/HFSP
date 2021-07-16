using Test, DrWatson
@quickactivate

include(srcdir("hfsp.jl"))

@test greet("Doug") == "Hello Doug"
