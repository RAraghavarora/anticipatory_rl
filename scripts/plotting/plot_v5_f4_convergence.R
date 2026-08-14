# F4 -- Convergence and checkpoint selection.
#
# Rolling greedy success (window 100) vs training step, every run, with a hollow
# marker at the step where restaurant_dqn_best.pt was taken. Answers two
# questions a reader will ask: is 500k steps enough, and was the reported
# checkpoint cherry-picked from a lucky spike.
#
# Faceted rather than overlaid: 13 noisy runs in one panel is unreadable, and
# the facet strip doubles as the direct label so identity is never colour-alone.
#
# Marker y is the binned success at the nearest bin to best_checkpoint_step.

source("scripts/plotting/v5_curve_style.R")

# Raw bins are 1,000 steps apart and very noisy. Each run gets a 25-bin centred
# moving average (25,000 steps) drawn on top of its own raw trace, so the trend
# is legible without hiding the variance it was computed from.
SMOOTH_BINS <- 25

df <- load_metric("success_rate_rolling__window_100", "mean") %>%
  mutate(series = factor(series, levels = names(SERIES_COLORS))) %>%
  arrange(run, step) %>%
  group_by(run) %>%
  mutate(smoothed = roll_mean(value, SMOOTH_BINS)) %>%
  ungroup()

best <- read_csv(BEST_CSV, show_col_types = FALSE) %>%
  filter(run %in% unique(df$run))

# Checkpoint markers are vertical rules, not points. The checkpoint is selected
# on persistent greedy evaluation over 100 tasks, which is a different
# measurement from the rolling training success plotted here -- the two differ
# by up to 0.50 on the noisiest run -- so a point would assert a y-value this
# curve does not have. The rule marks when the checkpoint was taken and claims
# nothing about its height.
marks <- best %>%
  distinct(run, best_step) %>%
  left_join(df %>% distinct(run, series), by = "run")

p <- ggplot(df, aes(step, value)) +
  geom_vline(
    data = marks, aes(xintercept = best_step),
    color = "gray45", linetype = "22", linewidth = 0.35
  ) +
  geom_line(aes(color = series, group = run), linewidth = 0.3, alpha = 0.28) +
  geom_line(aes(y = smoothed, color = series, group = run), linewidth = 0.85) +
  facet_wrap(~series, ncol = 4) +
  scale_color_manual(values = SERIES_COLORS, drop = TRUE) +
  scale_y_continuous(
    name = "Rolling task success (window 100)",
    labels = scales::label_percent(accuracy = 1),
    breaks = seq(0, 1, 0.25),
    limits = c(0, 1),
    expand = expansion(mult = 0.02)
  ) +
  scale_x_continuous(
    name = "Training step",
    breaks = c(0, 250000, 500000),
    labels = c("0", "250k", "500k"),
    expand = expansion(mult = 0.05)
  ) +
  base_theme +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray93", linewidth = 0.3),
    panel.background = element_rect(fill = "gray99", color = NA),
    strip.text = element_text(size = 10.5, color = "gray20", face = "bold",
                              margin = margin(b = 4)),
    panel.spacing.x = unit(1.1, "lines"),
    panel.spacing.y = unit(1.0, "lines"),
    axis.ticks = element_blank()
  )

save_fig(p, "results/v5/figures/f4_convergence.pdf", width = 9.0, height = 4.6)
