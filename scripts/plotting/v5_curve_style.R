# Shared style for the three v5 training-curve figures (F2 jar mechanism,
# F3 divergence, F4 convergence). Sourced by each plot script.
#
# Colour rule: gamma is an ordered quantity, so the anticipatory runs get a
# single-hue sequential purple ramp (light = short horizon, dark = long).
# Myopic and no-demo are different agents, not different horizons, so they get
# categorical hues instead. The categorical triple (darkest purple, vermillion,
# green) passes the six-check palette validator in light mode: worst adjacent
# CVD dE 11.0 (deutan), normal-vision dE 25.8, all >= 3:1 contrast.
#
# Colour follows the entity: a run keeps its colour across all three figures.

library(tidyverse)

BINNED_CSV <- "results/v5/figures/data/training_curves_binned.csv"
BEST_CSV   <- "results/v5/figures/data/best_checkpoints.csv"

# Sequential ramp over gamma, light -> dark as the credit horizon lengthens.
# Every step clears 3:1 contrast on white; the lighter steps of a conventional
# purple ramp do not, which made gamma = 0.90/0.95 unreadable.
GAMMA_COLORS <- c(
  "0.90" = "#9385C2",
  "0.95" = "#7E68B4",
  "0.96" = "#6A51A3",
  "0.97" = "#56408B",
  "0.98" = "#43326F",
  "0.99" = "#2F2352"
)
MYOPIC_COLOR  <- "#D55E00"  # vermillion

SERIES_COLORS <- c(
  setNames(GAMMA_COLORS, paste0("γ = ", names(GAMMA_COLORS))),
  "Myopic" = MYOPIC_COLOR
)

# Run name -> series label. Anticipatory runs collapse to their gamma; the
# myopic baseline gets its own label regardless of seed.
series_label <- function(run) {
  case_when(
    str_detect(run, "myopic") ~ "Myopic",
    TRUE ~ paste0("γ = ", sprintf("%.2f", parse_gamma(run)))
  )
}

# g0.97 -> 0.97, g099 -> 0.99 (the 0.99 runs are named without the decimal).
parse_gamma <- function(run) {
  raw <- str_match(run, "[_-]g0?\\.?(\\d+)[_-]")[, 2]
  suppressWarnings(as.numeric(paste0("0.", str_sub(raw, 1, 2))))
}

seed_label <- function(run) str_match(run, "[_-]s(\\d+)$")[, 2]

load_metric <- function(metric_name, value_col = "mean") {
  read_csv(BINNED_CSV, show_col_types = FALSE) %>%
    filter(metric == metric_name, !str_detect(run, "nodemo")) %>%
    mutate(
      value  = .data[[value_col]],
      series = series_label(run),
      seed   = seed_label(run)
    )
}

# Centred rolling mean with shrinking windows at the edges, so the smoothed line
# spans the full x-range instead of stopping short. A plain moving average is
# used rather than loess: it cannot invent structure, and it preserves the step
# at which a run collapses, which loess would round off.
roll_mean <- function(x, k) {
  half <- k %/% 2
  vapply(seq_along(x), function(i) {
    mean(x[max(1, i - half):min(length(x), i + half)])
  }, numeric(1))
}

base_theme <- theme_minimal(base_size = 12.5) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray92", linewidth = 0.3),
    axis.title = element_text(size = 10.5, color = "gray30"),
    axis.text = element_text(color = "gray40"),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.key.width = unit(1.6, "lines"),
    plot.margin = margin(6, 12, 4, 8)
  )

step_axis <- scale_x_continuous(
  name = "Training step",
  labels = function(x) paste0(x / 1000, "k"),
  expand = expansion(mult = c(0.01, 0.02))
)

save_fig <- function(plot, path, width = 7.2, height = 4.0) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggsave(path, plot, width = width, height = height, device = cairo_pdf)
  cat("wrote", path, "\n")
}
