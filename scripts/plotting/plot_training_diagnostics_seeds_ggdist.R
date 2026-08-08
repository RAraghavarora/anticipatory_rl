# Combined training-dynamics figure: success rate, auto-success rate, task
# return, selected Q-value, TD target -- myopic vs anticipatory DQN, all 4
# checkpoint seeds, one shared x-axis (training step). Same rationale as the
# IVR-paper-style 2x2 grid: these are all views of the same training run for
# the same 2 methods, so one figure is appropriate (unlike the cumulative-cost
# evaluation figure, which has a different x-axis and a different method set).
#
# Success rate / task return / selected Q-value / TD target are rebuilt from
# raw per-task or per-step data. Auto-success rate is the one exception: no
# raw per-task auto-success metric was logged (only the training script's own
# on-policy 100-task rolling window), so that panel alone reflects on-policy
# exploration noise the other panels don't -- noted here and in the caption,
# not hidden.
#
# stat_lineribbon(.width = 1) gives the min/max band across the 4 seeds per
# bin -- a literal statement of observed spread, not a fabricated
# distributional claim at n=4.

library(tidyverse)
library(ggdist)

df <- read_csv("results/canonical_planner/figures/training_diagnostics_seeds_binned.csv", show_col_types = FALSE) %>%
  mutate(metric = factor(metric, levels = c(
    "Success rate", "Auto-success rate", "Task return", "Selected Q-value", "TD target"
  )))

out_path <- "results/canonical_planner/figures/training_diagnostics_seeds_ggdist.pdf"
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

ggsave(out_path, p, width = 15, height = 3.4, device = cairo_pdf)
cat("wrote", out_path, "\n")
