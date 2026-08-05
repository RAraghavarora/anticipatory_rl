# Cumulative PDDL cost vs. number of tasks, via Makie.jl (CairoMakie backend).
#
# Same computation as scripts/restaurant/plot_canonical_cost.py: at each
# task_index, cost is averaged across the 10 canonical sequences, then
# cumsum'd over task_index, per (method, checkpoint_seed). For the two DQN
# methods, band() shows mean +/- std across the 4 seed curves.

using CSV
using DataFrames
using Statistics
using CairoMakie

const CSV_PATH = "results/canonical_planner/planner/task_results.csv"
const OUT_PATH = "results/canonical_planner/figures/cumulative_cost_makie.png"

const FD_METHODS = [
    ("myopic_fd_optimal", "Myopic FD (optimal)", RGBf(0.0, 0.0, 0.0)),
    ("clairvoyant_k3_lama", "Clairvoyant FD (K=3)", RGBf(0.902, 0.624, 0.0)),
]
const DQN_METHODS = [
    ("myopic_dqn_beta1", "Myopic DQN", RGBf(0.337, 0.706, 0.914)),
    ("anticipatory_dqn_beta1_25", "Anticipatory DQN", RGBf(0.0, 0.620, 0.451)),
]

function cumulative_curve(df, target_method; target_seed=nothing)
    sub = df[df.method_id .== target_method, :]
    if target_seed !== nothing
        sub = sub[.!ismissing.(sub.checkpoint_seed) .& (sub.checkpoint_seed .== target_seed), :]
    end
    g = combine(groupby(sub, :task_index), :task_cost_pddl => mean => :avg_cost)
    sort!(g, :task_index)
    g.cum_cost = cumsum(g.avg_cost)
    return g
end

function main()
    df = CSV.read(CSV_PATH, DataFrame)

    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1],
        xlabel="Number of tasks",
        ylabel="Cumulative PDDL cost (mean over 10 sequences)",
        title="Cumulative cost vs. tasks completed (Makie.jl)")

    for (method_id, label, color) in FD_METHODS
        curve = cumulative_curve(df, method_id)
        lines!(ax, curve.task_index .+ 1, curve.cum_cost;
            color=color, linestyle=:dash, linewidth=2, label=label)
    end

    for (method_id, label, color) in DQN_METHODS
        seeds = sort(unique(skipmissing(df[df.method_id.==method_id, :checkpoint_seed])))
        curves = [cumulative_curve(df, method_id; target_seed=s).cum_cost for s in seeds]
        mat = reduce(hcat, curves)  # tasks x n_seeds
        m = vec(mean(mat, dims=2))
        s = vec(std(mat, dims=2))
        x = (0:length(m)-1) .+ 1
        band!(ax, x, m .- s, m .+ s; color=(color, 0.2))
        lines!(ax, x, m; color=color, linewidth=2, label=label)
    end

    axislegend(ax; position=:lt, framevisible=false)
    save(OUT_PATH, fig)
    println("wrote ", OUT_PATH)
end

main()
