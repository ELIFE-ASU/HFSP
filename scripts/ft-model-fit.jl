using DrWatson
using HFSP
using Metaheuristics
using Queryverse
using StaticArrays
using Statistics

const DEFAULT_GRAPH = datadir("exp_pro", "sam.jld2")
const DEFAULT_DATA  = datadir("exp_pro", "2021-08-10_expression-levels", "ft.csv")

const MODELFILE = datadir("sims", "2021-08-10_expression-levels", "model.jld2")
const SIMFILE = datadir("sims", "2021-08-10_expression-levels", "sim.csv")
const PLOTFILE = plotsdir("2021-08-10_expression-levels.png")

stderr(A::AbstractArray) = isempty(A) ? zero(Float64) : std(A) / sqrt(length(A))

struct Config{G, S, T}
    graph::G
    G::S
    H::T
    data::DataFrame
    function Config(; graphfile=DEFAULT_GRAPH, datafile=DEFAULT_DATA)
        graph, G, H = setup(graphfile)
        data = loaddata(datafile)
        new{typeof(graph), typeof(G), typeof(H)}(graph, G, H, data)
    end
end

tomodel(config::Config, p, q) = model3(config.graph, config.G, config.H; p, q)

function loaddata(datafile, scaling=x -> x)
    df = load(datafile) |>
        @groupby([_.Cold, _.Warm]) |>
        @map({
            Cold     = first(key(_)),
            Warm     = last(key(_)),
            MeanFT   = mean(_.FT),
            VarFT    = var(_.FT),
            StdErrFT = stderr(_.FT)
        }) |>
        DataFrame
    df.Level = scaling(df.MeanFT) ./ maximum(df.MeanFT)
    df
end

function runmodel(model, data, n)
    df = DataFrame(Trial=Int[], Cold=Int[], Warm=Int[], ExpLevel=Float64[], Level=Float64[])
    init = zeros(F, length(model))
    for row in eachrow(data)
        cold, warm, exp_level = row.Cold, row.Warm, row.Level
        if !iszero(cold) || !iszero(warm)
            input = t -> t ≤ cold ? [F(0)] : [F(1)]
            ens = finalensemble(model, init, input, cold + warm, n)
            levels = mean(Array{Float64}(ens); dims=1)[1,:]
            for (trial, level) in enumerate(levels)
                push!(df, [trial, cold, warm, exp_level, level])
            end
        end
    end
    df.Treatment = string.(df.Cold) .* "/" .* string.(df.Warm)
    df
end

function evaluatemodel(model, data, n; cond=(c, w) -> !iszero(c) || !iszero(w))
    init = zeros(F, length(model))
    loss = 0.0
    for row in eachrow(data)
        cold, warm, exp_level = row.Cold, row.Warm, row.Level
        if cond(cold, warm)
            input = t -> t ≤ cold ? [F(0)] : [F(1)]
            ens = finalensemble(model, init, input, cold + warm, n)
            levels = mean(Array{Float64}(ens); dims=1)[1,:]
            μ, se = mean(levels), stderr(levels)
            loss += (exp_level - μ)^2
        end
    end
    return loss
end

function setparams!(model, p, q)
    for poly in model.p
        poly.f[1].p = p
        poly.f[4].p = q
    end
    model
end

function setp!(model, p)
    for poly in model.p
        poly.f[1].p = p
    end
    model
end

function setq!(model, q)
    for poly in model.p
        poly.f[4].p = q
    end
    model
end

function jointloss(config::Config, n)
    basemodel = model3(config.graph, config.G, config.H; p=0.0, q=0.0)
    params -> begin
        model = copy(basemodel)
        setparams!(model, params...)
        evaluatemodel(model, config.data, n; cond=(c, w) -> !iszero(c) || !iszero(w))
    end
end

function coldloss(config::Config, n; q=0.0)
    basemodel = model3(config.graph, config.G, config.H; p=0.0, q)
    params -> begin
        model = copy(basemodel)
        setp!(model, params...)
        evaluatemodel(model, config.data, n; cond=(c, w) -> !iszero(c) && iszero(w))
    end
end

function warmloss(config::Config, n; p=0.0)
    basemodel = model3(config.graph, config.G, config.H; p, q=0.0)
    params -> begin
        model = copy(basemodel)
        setq!(model, params...)
        evaluatemodel(model, config.data, n; cond=(c, w) -> !iszero(w))
    end
end

function fitmodel(loss::Function, npop::Int, x₀, σ; debug=false, kwargs...)
    debug && @info "Building Population"
    X = [x₀ .+ σ * randn(length(x₀)) for _ in 2:npop]
    push!(X, x₀)
    pop = [Metaheuristics.create_child(x, loss(x)) for x in X]

    debug && @info "Building Algorithm"
    algo = ECA(N = length(pop), adaptive=true, options=Options(; debug, kwargs...))
    algo.status = State(Metaheuristics.get_best(pop), pop)

    debug && @info "Running Optimization"
    bounds = [zeros(length(x₀)) ones(length(x₀))]'
    coldresult = optimize(loss, bounds, algo)

    debug && @info "Done"
    minimizer(coldresult)
end

function fitmodel(loss::Function, npop::Int; bounds=[0 1.]', debug=false, kwargs...)
    debug && @info "Building Algorithm"
    algo = ECA(N = npop, adaptive=true, options=Options(; debug, kwargs...))

    debug && @info "Running Optimization"
    coldresult = optimize(loss, bounds, algo)

    debug && @info "Done"
    minimizer(coldresult)
end

function fitmodel(config::Config, n::Int, npop::Int; debug=false, kwargs...)
    debug && @info "Building Cold Loss Function"
    cold = coldloss(config, n; q=0.0)

    debug && @info "Fitting Cold Parameter"
    p = fitmodel(cold, npop; debug, kwargs...)

    debug && @info "Building Warm Loss Function"
    warm = warmloss(config, n; p=p[1])

    debug && @info "Fitting Warm Parameter"
    q = fitmodel(warm, npop; debug, kwargs...)

    debug && @info "Building Joint Loss Function"
    loss = jointloss(config, n)

    debug && @info "Fitting Joint Parameters"
    params = fitmodel(loss, npop, [p..., q...], 0.001; debug, kwargs...)

    debug && @info "Building Final Model"
    model = tomodel(config, params...)

    model, evaluatemodel(model, config.data, n), params
end

function plot(df)
    df |>
        @vlplot(
            x = {
                "Treatment:n",
                sort = false,
                title = "Treatment (Days Cold / Warm)",
            },
            title = "Final FT Expression by Treatment",
            width = 400,
            height = 300,
        ) +
        @vlplot(
            :bar,
            y = {
                "mean(ExpLevel)",
                scale = {
                    zero = false,
                },
                title = "FT Expression Level",
            },
        ) +
        @vlplot(
            mark = {
                :point,
                filled = true,
            },
            y = {
                "mean(Level)",
            },
            color = {
                value = :black,
            },
        ) +
        @vlplot(
            mark = {
                :errorbar,
                extent = :stddev,
            },
            y = {
                "Level:q",
            },
        )
end

function main()
    debug = true
    enssize, npop = 100, 10
    iterations = 100

    config = Config()
    model, value, params = fitmodel(config, enssize, npop; debug, iterations)

    tagsave(MODELFILE, Dict("model" => model))

    df = runmodel(model, config.data, enssize)
    save(SIMFILE, df)

    plot(df) |> save(PLOTFILE)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
