#!/usr/bin/env julia
# scripts/plot_restaurant_multiseed_aog.jl

using Pkg
Pkg.activate(; temp=true)
Pkg.add(["DataFrames", "CSV", "CairoMakie", "AlgebraOfGraphics", "StatsBase"])

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using Statistics

function main()
    println("Loading metrics from aggregate.csv...")
    data_path = "runs/compare_restaurant_report_multiseed/aggregate.csv"
    if !isfile(data_path)
        error("Could not find aggregate.csv at $data_path")
    end
    
    df = CSV.read(data_path, DataFrame)
    
    # Reshape from wide to tidy format
    cols_to_stack = propertynames(df)[3:end] # Exclude seed and path
    long_df = stack(df, cols_to_stack, variable_name=:RawMetric, value_name=:Value)
    
    # Parse the method and metric name
    long_df.Method = map(x -> split(String(x), ".")[1] == "anticipatory" ? "Anticipatory" : "Myopic", long_df.RawMetric)
    long_df.MetricName = map(x -> split(String(x), ".")[2], long_df.RawMetric)
    
    # Keep only the requested metrics
    filter!(row -> row.MetricName in ["avg_task_steps", "avg_task_return", "reward_per_step"], long_df)
    
    # Human readable facet names
    metric_labels = Dict(
        "avg_task_steps" => "Avg steps / task",
        "avg_task_return" => "Avg task return",
        "reward_per_step" => "Reward / step"
    )
    long_df.Facet = map(x -> metric_labels[x], long_df.MetricName)
    
    # Group and calculate summary stats (Mean and Std Error/Deviation)
    agg_df = combine(groupby(long_df, [:Method, :Facet]), 
                     :Value => mean => :Mean, 
                     :Value => std => :Std)
                     
    # Professional Okabe-Ito colorblind friendly palette
    # Orange and Sky Blue
    colors = ["#E69F00", "#56B4E9"]
    
    println("Generating plot with AlgebraOfGraphics...")
    
    # Layer 1: Scatter plot for individual seed values
    pts = data(long_df) * 
          mapping(:Method, :Value, color=:Method, col=:Facet) * 
          visual(Scatter, alpha=0.5, markersize=8, strokewidth=0)
          
    # Layer 2: Error bars for standard deviation
    errs = data(agg_df) * 
           mapping(:Method, :Mean, :Std, color=:Method, col=:Facet) * 
           visual(Errorbars, linewidth=2.5, whiskerwidth=10)
           
    # Layer 3: Solid mean points overlay
    means = data(agg_df) * 
            mapping(:Method, :Mean, color=:Method, col=:Facet) * 
            visual(Scatter, marker=:rect, markersize=14)
            
    # Combine the layers
    plt = errs + means + pts
    
    # Theming for AI Academic Paper (e.g. NeurIPS style)
    custom_theme = Theme(
        fontsize = 12, # legible font defaults for single-column inserts in double-column formatting
        Axis = (
            xticklabelrotation = pi/6,
            xgridvisible = false,
            ygridvisible = true,
            ygridstyle = :dash,
            topspinevisible = false,
            rightspinevisible = false,
            leftspinecolor = :gray30,
            bottomspinecolor = :gray30,
        ),
        Palette = (color = colors,)
    )
    
    with_theme(custom_theme) do
        # 3.25 inches is typical single column width. Let's set responsive axis sizing
        # linkyaxes = :none ensures each facet has its own specific y-scale
        fig = draw(plt; 
                   axis = (; width=150, height=140), 
                   facet = (; linkyaxes = :none, linkxaxes = :none))
                   
        # Title spanning over all facets
        Label(fig.figure[0, :], "Restaurant Multi-seed Evaluation", fontsize=16, font=:bold)
        
        out_pdf = "runs/compare_restaurant_report_multiseed/restaurant_aog_plot.pdf"
        out_png = "runs/compare_restaurant_report_multiseed/restaurant_aog_plot.png"
        
        save(out_pdf, fig)
        save(out_png, fig, px_per_unit=3)
        
        println("Success! Publication-ready plots saved to: \n  $out_pdf\n  $out_png")
    end
end

main()
