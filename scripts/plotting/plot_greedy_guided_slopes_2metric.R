# Greedy-vs-guided evaluation, 2 metrics only (Auto% dropped), as a single
# row of 2 paired-slope panels -- not a 2x2 grid, and not the bivariate
# steps-vs-cost scatter (rejected: turning two independent outcome metrics
# into x/y coordinates asked the reader to do more interpretive work than
# the claim needs, and abandoned the paired-slope visual language already
# in use for no real gain).
#
# Mark language: bold mean line + point per (method, restaurant), a small
# min/max error bar on each mean point, and a repelled value label. The
# first version of this script also drew the 4 raw per-seed lines behind
# the mean -- dropped on request: at only 4 seeds they mostly restated the
# error bar's own range without adding much, and the combination read as
# clutter ("ghost lines"). The error bar itself is min/max (not a computed
# SD), keeping the same n=4-honesty stance as the rest of this repo's
# _ggdist.R scripts -- just moved from a full spaghetti display to a single
# summary range per point, which is the smaller, still-honest version of
# the same idea.
#
# Restaurant is folded into linetype (solid = 2-room, dashed = 3-room)
# within the same 2 metric panels rather than a facet row, since Steps/task
# and Cost/task sit on comparable scales across both restaurants.
#
# Labels use ggrepel (direction = "y") instead of manual vjust/hjust
# offsets -- with the spaghetti gone there's nothing left to dodge around,
# but two means can still sit close enough to collide on their own (e.g.
# Myopic/Anticipatory both ~1,430 at the 2-room Greedy point, or the two
# Anticipatory arms both ~900-1000 at Guided); repel resolves those
# per-facet instead of hand-tuning offsets that only fixed one collision.
#
# Data: results/v5/figures/greedy_guided_per_seed.csv, built by
# scripts/restaurant/build_greedy_guided_summary.py.

library(tidyverse)
library(ggrepel)
library(patchwork)

df <- read_csv("results/v5/figures/greedy_guided_per_seed.csv", show_col_types = FALSE) %>%
  filter(metric %in% c("Steps/task", "Cost/task")) %>%
  mutate(
    metric = factor(metric, levels = c("Steps/task", "Cost/task")),
    deployment = factor(deployment, levels = c("Greedy", "Guided")),
    method = factor(method, levels = c("Myopic RL", "Anticipatory RL")),
    restaurant = factor(restaurant, levels = c("2-room", "3-room"))
  )

out_path <- "results/v5/figures/thesis/greedy_guided_slopes_2metric.pdf"
colors <- c("Myopic RL" = "#56B4E9", "Anticipatory RL" = "#009E73")
linetypes <- c("2-room" = "solid", "3-room" = "22")
method_offset <- c("Myopic RL" = -0.08, "Anticipatory RL" = 0.08)

df <- df %>% mutate(x_pos = as.numeric(deployment) + method_offset[as.character(method)])

means <- df %>%
  group_by(restaurant, metric, method, deployment, x_pos) %>%
  # named mean_value (not value) during aggregation: dplyr::summarise()
  # evaluates arguments sequentially, so reassigning "value" to mean(value)
  # first would make the later min(value)/max(value) read back that scalar
  # instead of the original per-seed column, collapsing ymin/ymax to it.
  summarise(mean_value = mean(value), ymin = min(value), ymax = max(value), .groups = "drop") %>%
  rename(value = mean_value)

label_fmt <- function(metric, value) {
  if_else(metric == "Cost/task", scales::label_comma(accuracy = 1)(value), scales::label_number(accuracy = 0.01)(value))
}
means <- means %>% mutate(value_label = label_fmt(metric, value))

p <- ggplot() +
  geom_errorbar(
    data = means, aes(x = x_pos, ymin = ymin, ymax = ymax, color = method),
    width = 0.05, linewidth = 0.6, alpha = 0.55
  ) +
  geom_line(
    data = means, aes(x = x_pos, y = value, group = interaction(method, restaurant),
                       color = method, linetype = restaurant),
    linewidth = 1.3
  ) +
  geom_point(
    data = means, aes(x = x_pos, y = value, color = method),
    size = 2.6
  ) +
  geom_text_repel(
    data = means, aes(x = x_pos, y = value, label = value_label, color = method),
    size = 2.9, fontface = "bold", show.legend = FALSE, seed = 42,
    direction = "y", segment.size = 0.3, min.segment.length = 0, box.padding = 0.25,
    nudge_x = ifelse(means$deployment == "Greedy", -0.3, 0.3),
    hjust = ifelse(means$deployment == "Greedy", 1, 0)
  ) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  scale_color_manual(values = colors, name = NULL) +
  scale_linetype_manual(values = linetypes, name = NULL) +
  scale_x_continuous(
    breaks = c(1, 2), labels = c("Greedy", "Guided"),
    limits = c(1 - 0.55, 2 + 0.55), expand = expansion(mult = c(0.1, 0.1))
  ) +
  labs(x = NULL, y = NULL) +
  guides(
    color = guide_legend(override.aes = list(linetype = "solid", linewidth = 1.3, size = 2.6)),
    linetype = guide_legend(override.aes = list(color = "gray30", linewidth = 1))
  ) +
  theme_minimal(base_size = 12.5) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 11, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.text = element_text(color = "gray40"),
    panel.spacing = unit(1.6, "lines"),
    plot.margin = margin(8, 16, 4, 6)
  )

ggsave(out_path, p, width = 8.5, height = 4.8, device = cairo_pdf)
cat("wrote", out_path, "\n")
