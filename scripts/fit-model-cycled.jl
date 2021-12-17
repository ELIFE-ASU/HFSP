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
    raw::DataFrame
    data::DataFrame
    function Config(graphfile, datafile)
        graph = setup(graphfile)
        raw, data = loaddata(datafile)
        new{typeof(graph)}(graph, raw, data)
    end
end

tomodel(config::Config, p, q) = model3(config.graph; p, q)

function loaddata(datafile, scaling=x -> x)
    raw = DataFrame(load(datafile))
    processed = raw |>
        @groupby([_.Cold, _.Warm]) |>
        @map({
            Cold        = first(key(_)),
            Warm        = last(key(_)),
            MeanLevel   = mean(_.Level),
            Variance    = var(_.Level),
            StdError    = stderr(_.Level)
        }) |>
        DataFrame
    processed.ScaledLevel = scaling(processed.MeanLevel) ./ maximum(processed.MeanLevel)
    raw, processed
end

function tempschedule(prep, cold, warm; coldunit=20, warmunit=4)
    coldunit + warmunit != 24 && error("coldunit + warmunit must be 24")

    prepwarm = ones(F, prep)

    unit = if iszero(warm)
        zeros(F, coldunit + warmunit)
    elseif isone(warm)
        [zeros(F, coldunit); ones(F, warmunit)]
    else
        error("warm must be 0 or 1")
    end

    outer, additional = iszero(warm) ? divrem(cold, 24) : divrem(cold, coldunit)

    map(collect, [prepwarm; repeat(unit; outer); ones(F, additional)])
end

function runmodel(model, data, n)
    df = DataFrame(Trial=Int[], Cold=Int[], Warm=Int[], ExpLevel=Float64[], Level=Float64[])
    for row in eachrow(data)
        cold, warm, exp_level = row.Cold, row.Warm, row.ScaledLevel
        init, input = if iszero(cold) && iszero(warm)
            rand(F, length(model)), tempschedule(1680, cold, warm)
        else
            zeros(F, length(model)), tempschedule(0, cold, warm)
        end
        ens = finalensemble(model, init, input, length(input), n)
        levels = mean(Array{Float64}(ens); dims=1)[1,:]
        for (trial, level) in enumerate(levels)
            push!(df, [trial, cold, warm, exp_level, level])
        end
    end
    df.Treatment = string.(Int.(df.Cold / 168)) .* "W_C" .* map(x -> x == 1 ? "W" : "", df.Warm)
    df
end


function evaluatemodel(model, data, n)
    run = runmodel(model,data,n)
    df = run |>
        @groupby([_.Cold, _.Warm]) |>
        @map({
            Loss = (first(_.ExpLevel) - mean(_.Level))^2
        }) |>
        DataFrame

    mean(df.Loss)
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
        evaluatemodel(model, config.data, n)
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
