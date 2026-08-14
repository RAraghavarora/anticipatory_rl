# Sequence-by-sequence cost delta, v5, via Makie.jl (CairoMakie backend) --
# Julia counterpart to scripts/plotting/plot_v5_seq_comparison.R. Same data,
# same design: one diverging horizontal bar per sequence (Anticipatory RL
# guided minus K=3 Clairvoyant Oracle), not a 2-point-per-sequence
# dumbbell/slopegraph -- see the R script's header for why those read
# poorly (crossing lines, competing encodings). Positive/amber = K=3 costs
# less; negative/teal = we cost less -- same two hues used for these methods
# in every other figure in this archive (plot_canonical_cost_makie.jl's
# Anticipatory DQN / Clairvoyant FD colors).
#
# Data: results/v5/figures/seq_comparison.csv, built by
# scripts/restaurant/build_v5_seq_comparison.py.

using CSV
using DataFrames
using CairoMakie

const CSV_PATH = "results/v5/figures/seq_comparison.csv"
const OUT_PATH = "results/v5/figures/seq_comparison_makie.png"

const TEAL = RGBf(0.0, 0.620, 0.451)   # Anticipatory RL costs less
const AMBER = RGBf(0.902, 0.624, 0.0)  # K=3 Oracle costs less

function main()
    df = CSV.read(CSV_PATH, DataFrame)

    ours = df[df.label.=="Anticipatory RL (guided)", [:sequence_id, :mean_cost]]
    rename!(ours, :mean_cost => :ours_cost)
    oracle = df[df.label.=="K=3 Clairvoyant Oracle", [:sequence_id, :mean_cost, :cap_rate]]
    rename!(oracle, :mean_cost => :oracle_cost)

    wide = innerjoin(ours, oracle; on=:sequence_id)
    wide.seq_num = replace.(wide.sequence_id, "iid-eval-seq-" => "")
    sort!(wide, :seq_num)
    wide.delta = wide.ours_cost .- wide.oracle_cost
    wide.high_cap = wide.cap_rate .>= 0.4
    wide.seq_label = [hc ? "seq $n †" : "seq $n" for (n, hc) in zip(wide.seq_num, wide.high_cap)]
    wide.color = [d > 0 ? AMBER : TEAL for d in wide.delta]

    n = nrow(wide)
    ys = n:-1:1  # seq 00 at top, seq 09 at bottom

    fig = Figure(size=(850, 650))
    ax = Axis(fig[1, 1],
        xlabel="Anticipatory RL (guided) − K=3 Oracle, total cost per 50-task sequence",
        yticks=(ys, wide.seq_label),
        title="v5 sequence-by-sequence cost delta")

    barplot!(ax, ys, wide.delta; direction=:x, color=wide.color, width=0.6)
    vlines!(ax, [0.0]; color=:gray40, linewidth=1)
    # Makie doesn't auto-expand axis limits to fit text!() labels the way
    # ggplot's expansion() does -- pad explicitly or the biggest bars' value
    # labels get clipped at the panel edge.
    xlims!(ax, minimum(wide.delta) - 2500, maximum(wide.delta) + 2500)

    for (y, d) in zip(ys, wide.delta)
        align = d > 0 ? (:left, :center) : (:right, :center)
        offset = d > 0 ? 300 : -300
        text!(ax, d + offset, y; text=(d > 0 ? "+" : "") * string(round(Int, d)),
            align=align, fontsize=13, font=:bold)
    end

    Label(fig[2, 1], "† K=3 hit its 600s satisficing cap on >=40% of this sequence's windows.";
        fontsize=11, color=:gray40, halign=:left, tellwidth=false)

    save(OUT_PATH, fig)
    println("wrote ", OUT_PATH)
end

main()
