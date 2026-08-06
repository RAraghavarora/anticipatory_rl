# Training reward vs. steps, myopic vs anticipatory (seed 0), via Makie.jl.
#
# Reads the binned mean/std already computed by plot_training_reward.py
# (results/canonical_planner/figures/training_reward_binned.csv) -- no need
# to re-parse the 1.1GB metrics.csv files in Julia.

using CSV
using DataFrames
using CairoMakie

const CSV_PATH = "results/canonical_planner/figures/training_reward_binned.csv"
const OUT_PATH = "results/canonical_planner/figures/training_reward_makie.png"

const COLORS = Dict(
    "Myopic DQN" => RGBf(0.337, 0.706, 0.914),
    "Anticipatory DQN" => RGBf(0.0, 0.620, 0.451),
)
const ORDER = ["Myopic DQN", "Anticipatory DQN"]

function main()
    df = CSV.read(CSV_PATH, DataFrame)

    fig = Figure(size=(900, 550), fontsize=15)
    ax = Axis(fig[1, 1],
        xlabel="Training step",
        ylabel="Task return",
        title="Training reward vs. steps (seed 0)",
        subtitle="binned mean ± std",
        xtickformat=xs -> [string(round(Int, x ÷ 1000), "k") for x in xs])

    for label in ORDER
        g = sort(df[df.label.==label, :], :env_step)
        color = COLORS[label]
        band!(ax, g.env_step, g.mean .- g.std, g.mean .+ g.std; color=(color, 0.25))
        lines!(ax, g.env_step, g.mean; color=color, linewidth=2.5, label=label)
    end

    axislegend(ax; position=:rb, framevisible=false)
    save(OUT_PATH, fig)
    println("wrote ", OUT_PATH)
end

main()
