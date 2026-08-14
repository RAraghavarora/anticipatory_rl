# Combined training-dynamics figure, 2-room and 3-room (v5) stacked: success
# rate, task return, selected Q-value, TD target -- myopic vs anticipatory
# DQN, one shared x-axis (training step) per row. Auto-success rate dropped
# from both (it was the one panel in the original 2-room figure built from
# an on-policy rolling-window metric rather than raw per-task data, so
# dropping it makes every remaining panel uniformly "rebuilt from raw
# per-task/per-step data" in both domains).
#
# Unlike the cumulative-cost combined figure, this one CAN be built in one R
# session/patchwork call: that earlier crash was ggplot2's guide-building
# breaking on two DIFFERENT-length override.aes vectors (8 vs 6 methods,
# one per method). Here every panel has exactly the same 2 series (Myopic
# DQN, Anticipatory DQN) and the only override.aes is a scalar
# (linewidth = 1.2), so there's no length mismatch for ggplot2 to choke on.
# One shared legend (extracted from the 2-room panel) is used for both rows
# since the two panels' legends would be identical anyway.
#
# 2-room: results/canonical_planner/figures/training_diagnostics_seeds_binned.csv
# (built by build_seed_training_diagnostics.py, unchanged -- Auto-success
# rate is just filtered out here at plot time, not rebuilt).
# v5: results/v5/figures/training_diagnostics_seeds_binned.csv (built by
# build_v5_training_diagnostics.py, which never included Auto-success rate).
# v5's DQN arms have 5 seeds now (0, 4, 8, 16, 42); the 2-room arms have 4.

library(tidyverse)
library(ggdist)
library(patchwork)

colors <- c("Myopic DQN" = "#56B4E9", "Anticipatory DQN" = "#009E73")
metric_levels <- c("Success rate", "Task return", "Selected Q-value", "TD target")

build_panel <- function(csv_path, show_legend) {
  df <- read_csv(csv_path, show_col_types = FALSE) %>%
    filter(metric != "Auto-success rate") %>%
    mutate(metric = factor(metric, levels = metric_levels))

  ggplot(df, aes(x = step, y = mean_value, color = label, fill = label)) +
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
      legend.position = if (show_legend) "bottom" else "none",
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
}

get_legend <- function(p) {
  g <- ggplotGrob(p)
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}

legend_grob <- get_legend(build_panel("results/canonical_planner/figures/training_diagnostics_seeds_binned.csv", TRUE))

panel_2room <- build_panel("results/canonical_planner/figures/training_diagnostics_seeds_binned.csv", FALSE) +
  labs(x = NULL, y = "2-room") + theme(axis.text.x = element_blank(), axis.title.y = element_text(face = "bold", size = 10.5, color = "gray20"))
panel_v5 <- build_panel("results/v5/figures/training_diagnostics_seeds_binned.csv", FALSE) +
  theme(strip.text = element_blank()) +
  labs(y = "3-room") + theme(axis.title.y = element_text(face = "bold", size = 10.5, color = "gray20"))

combined <- panel_2room / panel_v5 / wrap_elements(legend_grob) + plot_layout(heights = c(1, 1, 0.15))

out_path <- "results/v5/figures/thesis/training_diagnostics_both.pdf"
ggsave(out_path, combined, width = 12, height = 6.4, device = cairo_pdf)
cat("wrote", out_path, "\n")
