# F3 -- Divergence.
#
# Max absolute selected Q vs training step, log y. Direct evidence for the
# stability claim in the horizon/calibration appendix: past gamma ~= 0.97 the
# value function does not merely decalibrate, some seeds diverge outright.
#
# One solid line per gamma -- the best seed, ranked by the greedy evaluation
# score its saved checkpoint achieved -- so the stable trend is read without
# seed clutter. The best seed at every gamma is a non-diverging one (at gamma =
# 0.98 and 0.99 the run that explodes is the WORST seed), so showing best seeds
# alone would delete the phenomenon this figure exists to show. The diverging
# runs are therefore kept, dashed and labelled, and named as the outliers they
# are.

library(ggrepel)
source("scripts/plotting/v5_curve_style.R")

DIVERGENCE_THRESHOLD <- 1e4

df <- load_metric("q_selected_abs_max", "max") %>%
  mutate(series = factor(series, levels = names(SERIES_COLORS)))

peaks <- df %>%
  group_by(run, series, seed) %>%
  summarise(peak = max(value), .groups = "drop")

# Diverged runs: kept regardless of rank.
diverged <- peaks %>% filter(peak > DIVERGENCE_THRESHOLD)

# Best seed per gamma, by the greedy-evaluation score of its saved checkpoint.
best_seed <- read_csv(BEST_CSV, show_col_types = FALSE) %>%
  inner_join(peaks, by = "run") %>%
  anti_join(diverged, by = "run") %>%
  group_by(series) %>%
  slice_max(best_value, n = 1, with_ties = FALSE) %>%
  ungroup()

plot_df <- df %>%
  filter(run %in% c(best_seed$run, diverged$run)) %>%
  mutate(kind = if_else(run %in% diverged$run, "diverged", "best"))

ends <- plot_df %>%
  filter(kind == "diverged") %>%
  group_by(run, series, seed) %>%
  slice_max(step, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(label = sprintf("%s  seed %s", series, seed))

p <- ggplot(plot_df, aes(step, value, color = series, group = run)) +
  geom_line(aes(linetype = kind), linewidth = 0.7) +
  geom_text_repel(
    data = ends, aes(label = label),
    hjust = 1, nudge_y = 0.35, size = 3.2, fontface = "bold",
    segment.color = "gray70", segment.size = 0.25, min.segment.length = 0,
    show.legend = FALSE
  ) +
  scale_color_manual(values = SERIES_COLORS, drop = TRUE) +
  scale_linetype_manual(values = c(best = "solid", diverged = "21"), guide = "none") +
  scale_y_log10(
    name = expression("max |" * Q[selected] * "|   (log scale)"),
    labels = scales::label_log()
  ) +
  scale_x_continuous(
    name = "Training step",
    labels = function(x) paste0(x / 1000, "k"),
    breaks = seq(0, 500000, 100000),
    expand = expansion(mult = c(0.01, 0.08))
  ) +
  annotation_logticks(sides = "l", color = "gray70", linewidth = 0.25) +
  guides(color = guide_legend(nrow = 2)) +
  base_theme

save_fig(p, "results/v5/figures/f3_divergence.pdf", height = 4.2)
