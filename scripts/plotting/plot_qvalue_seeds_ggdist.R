# Q-value diagnostic: mean selected Q vs. mean TD target, over training steps,
# aggregated across all 4 checkpoint seeds {0,4,8,16}, myopic vs anticipatory.
#
# Appendix figure motivated by this project's documented history of Q-value
# instability (self-loop bug, multi-task Q-explosion) -- following the
# DQN/Double-DQN precedent of plotting Q-value (and Q-vs-target) over training
# as direct evidence of stability, not a generic diagnostic add-on.
#
# Same seed-aggregation approach as the reward/success curves: each seed's
# own binned mean curve is one "draw" per bin; ggdist::stat_lineribbon(.width=1)
# gives the min/max band across the 4 seeds -- a literal statement of observed
# spread, not a fabricated distributional claim at n=4.

library(tidyverse)
library(ggdist)

df <- read_csv("results/canonical_planner/figures/qvalue_seeds_binned.csv", show_col_types = FALSE) %>%
  mutate(metric = recode(metric, q_selected_mean = "Selected Q-value", target_mean = "TD target")) %>%
  mutate(metric = factor(metric, levels = c("Selected Q-value", "TD target")))

out_path <- "results/canonical_planner/figures/qvalue_seeds_ggdist.pdf"
colors <- c("Myopic DQN" = "#56B4E9", "Anticipatory DQN" = "#009E73")

p <- ggplot(df, aes(x = step, y = mean_value, color = label, fill = label)) +
  stat_lineribbon(.width = 1, alpha = 0.18, linewidth = 0.7) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(values = colors, name = NULL) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +
  labs(x = "Training step", y = "Value") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    strip.text = element_text(face = "bold", size = 10, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40"),
    panel.spacing = unit(1.2, "lines"),
    plot.margin = margin(8, 10, 4, 6)
  )

ggsave(out_path, p, width = 7.5, height = 4, device = cairo_pdf)
cat("wrote", out_path, "\n")
