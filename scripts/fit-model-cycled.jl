using ArgParse
using BlackBoxOptim
using DrWatson
using HFSP
using Queryverse
using StaticArrays
using Statistics

stderr(A::AbstractArray) = isempty(A) ? zero(Float64) : std(A) / sqrt(length(A))

struct Config{G}
    graph::G
    data::DataFrame
    function Config(graphfile, datafile)
        graph = setup(graphfile)
        data = loaddata(datafile)
        new{typeof(graph)}(graph, data)
    end
end

tomodel(config::Config, p, q) = model3(config.graph; p, q)

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

function jointloss(config::Config, n)
    basemodel = tomodel(config, 0.0, 0.0)
    params -> begin
        model = copy(basemodel)
        setparams!(model, params...)
        evaluatemodel(model, config.data, n; cond=(c, w) -> !iszero(c) || !iszero(w))
    end
end

function fitmodel(loss::Function, args...; debug=false, kwargs...)
    debug && @info "Running Optimization"
    TraceMode = debug ? :compact : :silent
    result = bboptimize(loss, args...; TraceMode, kwargs...)

    debug && @info "Done"
    best_candidate(result)
end

function fitmodel(config::Config, n::Int; debug=false, kwargs...)
    debug && @info "Building Loss Function"
    loss = jointloss(config, n)

    debug && @info "Fitting Parameters"
    params = fitmodel(loss; debug, SearchRange=(0.0, 1.0), NumDimensions=2, kwargs...)

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

const DEFAULT_GRAPH = datadir("exp_pro", "sam.jld2")

s = ArgParseSettings()
@add_arg_table! s begin
    "--data", "-d"
        help = "the path to the data to fit the model to, CSV format"
        arg_type = String
        required = true
    "--graph", "-g"
        help = "path to the graph file, JLD2 format"
        arg_type = String
        default = DEFAULT_GRAPH
    "--out", "-o"
        help = "the path to the directory in which to store the model and simulation data"
        arg_type = String
        default = datadir("sims")
    "--plot", "-p"
        help = "the path to the directory in which to store plots"
        arg_type = String
        default = plotsdir()
    "--enssize", "-e"
        help = "the size of the ensembled used to compute average behavior"
        arg_type = Int
        default = 100
    "--maxtime", "-t"
        help = "the maximum number of evaluations to run"
        arg_type = Int
        default = 1000
    "--method", "-m"
        help = "the optimization method used"
        arg_type = String
        default = "adaptive_de_rand_1_bin_radiuslimited"
    "--verbose", "-v"
        help = "run in verbose mode"
        action = :store_true
end

function main()
    args = parse_args(s)

    graphfile = args["graph"]
    datafile = args["data"]
    debug = args["verbose"]
    enssize = args["enssize"]
    MaxTime = args["maxtime"]
    Method = Symbol(args["method"])

    name, _ = splitext(basename(datafile))
    modelfile = joinpath(args["out"], name * "_model.jld2")
    simfile = joinpath(args["out"], name * "_sim.csv")
    plotfile = joinpath(args["plot"], name * ".png")

    config = Config(graphfile, datafile)
    model, value, params = fitmodel(config, enssize; debug, MaxTime, Method)

    debug && @info "Saving model to" modelfile
    mkpath(dirname(modelfile))
    tagsave(modelfile, Dict("model" => model))

    df = runmodel(model, config.data, enssize)
    debug && @info "Saving sim data to" simfile
    mkpath(dirname(simfile))
    save(simfile, df)

    debug && @info "Saving plot to" plotfile
    mkpath(dirname(plotfile))
    plot(df) |> save(plotfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
