using DrWatson
using Graphs
using MetaGraphs
using Queryverse

function main(filename, out)
    table = DataTable(load(filename))
    indices = Dict{Int,Int}()
    g = MetaGraph(SimpleGraph())
    for row in table
        i = get!(indices, row.label, length(indices) + 1)
        j = get!(indices, row.neighbor, length(indices) + 1)
        if !has_vertex(g, i)
            add_vertex!(g)
            set_prop!(g, i, :label, row.label)
        end
        if !has_vertex(g, j)
            add_vertex!(g)
            set_prop!(g, j, :label, row.neighbor)
        end
        add_edge!(g, i, j)
        set_prop!(g, Edge(i, j), :area, row[Symbol("wall area")])
    end
    tagsave(out, Dict("graph" => g))
    g
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(datadir("exp_raw", "sam.csv"), datadir("exp_pro", "sam.jld2"))
end
