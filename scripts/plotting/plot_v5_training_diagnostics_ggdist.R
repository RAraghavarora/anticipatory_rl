# Combined training-dynamics figure, v5: success rate, task return, selected
# Q-value, TD target -- myopic vs anticipatory DQN, one shared x-axis
# (training step). v5 counterpart to plot_training_diagnostics_seeds_ggdist.R
# (2-room, left untouched).
#
# Auto-success rate dropped (not needed here) -- in the 2-room version it
# was also the one panel built from an on-policy rolling-window metric
# rather than raw per-task data, so this version is now uniformly "rebuilt
# from raw per-task/per-step data" across all four remaining panels.
#
# Both methods have 5 seeds now (0, 4, 8, 16, 42). stat_lineribbon
# (.width = 1) just uses however many seeds each method has.
#
# Data: results/v5/figures/training_diagnostics_seeds_binned.csv, built by
# scripts/plotting/build_v5_training_diagnostics.py.

library(tidyverse)
library(ggdist)

df <- read_csv("results/v5/figures/training_diagnostics_seeds_binned.csv", show_col_types = FALSE) %>%
  mutate(metric = factor(metric, levels = c(
    "Success rate", "Task return", "Selected Q-value", "TD target"
  )))

out_path <- "results/v5/figures/training_diagnostics_seeds_ggdist.pdf"
colors <- c("Myopic DQN" = "#56B4E9", "Anticipatory DQN" = "#009E73")

p <- ggplot(df, aes(x = step, y = mean_value, color = label, fill = label)) +
  stat_lineribbon(.width = 1, alpha = 0.22, linewidth = 0, show.legend = FALSE) +
  stat_summary(fun = median, geom = "line", linewidth = 0.7, alpha = 1) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_color_manual(values = colors, name = NULL) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +
  labs(x = "Training step", y = NULL) +
  guides(color = guide_legend(override.aes = list(linewidth = 1.2))) +
  theme_minimal(base_size = 10.5) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    strip.text = element_text(face = "bold", size = 9.5, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40", size = 7.5),
    panel.spacing = unit(1, "lines"),
    plot.margin = margin(8, 10, 4, 6)
  )

ggsave(out_path, p, width = 12, height = 3.4, device = cairo_pdf)
cat("wrote", out_path, "\n")
