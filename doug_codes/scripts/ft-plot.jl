using DrWatson
using Queryverse
using Statistics

const infile = datadir("exp_pro", "2021-08-10_expression-levels", "ft.csv")
const outfile = plotsdir("2021-10-20_ft-expression.png")

load(infile) |>
    @groupby([_.Cold, _.Warm]) |>
    @map({Treatment=(x-> string(first(x)) * "/" * string(last(x)))(key(_)), MeanFT=mean(_.FT)}) |>
    @vlplot(
        :bar,
        x = {
            :Treatment,
            type="nominal",
            sort=false,
            title="Treatment (Days Cold/Warm)"
        },
        y = {
            :MeanFT,
            title = "Mean FT Expression"
        },
        title="FT Expression by Treatment",
        width=400,
        height=300
    ) |> save(outfile)
