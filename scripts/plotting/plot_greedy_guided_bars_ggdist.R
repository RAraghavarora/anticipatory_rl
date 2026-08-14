# Greedy-vs-guided evaluation grid, ggdist version: 2 restaurants (stacked
# rows) x 2 metrics (Auto% dropped, same as the slope figures). Same bar
# layout as plot_greedy_guided_grid.R, but the uncertainty mark is swapped
# from a plain geom_errorbar(mean +/- sd) to ggdist::stat_pointinterval
# (.width = 1, the default median_qi point_interval) computed directly on
# the 4 raw per-seed values -- median + literal min/max, not a computed SD
# whisker. Same reasoning as every other _ggdist.R script in this repo:
# a symmetric SD interval at n=4 implies distributional knowledge the data
# doesn't support; ggdist's point_interval machinery is built to summarize
# a raw sample honestly instead.
#
# Data: results/v5/figures/greedy_guided_per_seed.csv (raw, one row per
# seed), built by scripts/restaurant/build_greedy_guided_summary.py -- NOT
# the pre-aggregated greedy_guided_grid.csv, since stat_pointinterval needs
# the raw sample to compute its own interval.

library(tidyverse)
library(ggdist)
library(patchwork)

df <- read_csv("results/v5/figures/greedy_guided_per_seed.csv", show_col_types = FALSE) %>%
  filter(metric %in% c("Steps/task", "Cost/task")) %>%
  mutate(
    metric = factor(metric, levels = c("Steps/task", "Cost/task")),
    deployment = factor(deployment, levels = c("Greedy", "Guided")),
    method = factor(method, levels = c("Myopic RL", "Anticipatory RL"))
  )

out_path <- "results/v5/figures/greedy_guided_bars_ggdist.pdf"
colors <- c("Myopic RL" = "#56B4E9", "Anticipatory RL" = "#009E73")

label_fmt <- function(metric, value) {
  if_else(metric == "Cost/task", scales::label_comma(accuracy = 1)(value), scales::label_number(accuracy = 0.01)(value))
}
bar_means <- df %>%
  group_by(restaurant, deployment, method, metric) %>%
  summarise(mean_value = mean(value), .groups = "drop") %>%
  mutate(bar_label = label_fmt(metric, mean_value))

base_theme <- theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10.5, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.text.x = element_text(color = "gray30"),
    axis.text.y = element_text(color = "gray40", size = 8),
    panel.spacing = unit(1.4, "lines"),
    plot.margin = margin(6, 12, 4, 6)
  )

row_panel <- function(data, means_data, show_strip, show_xaxis, show_legend, tag) {
  ggplot(mapping = aes(x = deployment, group = method)) +
    geom_col(
      data = means_data, aes(y = mean_value, fill = method, alpha = deployment),
      position = position_dodge(width = 0.75), width = 0.65, color = NA
    ) +
    stat_pointinterval(
      data = data, aes(y = value, color = method),
      .width = 1, position = position_dodge(width = 0.75),
      point_size = 1.6, interval_size_range = c(0.5, 0.5), color = "gray20"
    ) +
    geom_text(
      data = means_data, aes(y = mean_value, label = bar_label, color = method),
      position = position_dodge(width = 0.75), vjust = -1.6, size = 2.9,
      fontface = "bold", show.legend = FALSE
    ) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = colors, name = NULL, drop = FALSE) +
    scale_color_manual(values = colors, guide = "none", drop = FALSE) +
    scale_alpha_manual(values = c(Greedy = 0.55, Guided = 1.0), guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
    labs(x = NULL, y = tag) +
    guides(fill = guide_legend(override.aes = list(alpha = 1))) +
    base_theme +
    theme(
      strip.text = if (show_strip) element_text(face = "bold", size = 10.5, color = "gray20") else element_blank(),
      axis.text.x = if (show_xaxis) element_text(color = "gray30") else element_blank(),
      axis.title.y = element_text(face = "bold", size = 10.5, color = "gray20", angle = 90),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = 10)
    )
}

# Built the same way as plot_greedy_guided_grid.R: the 3-room panel has no
# Myopic RL data, so its own legend would render that key blank. Extract the
# legend from the 2-room panel (which has real bars for both methods) and
# place it as its own row instead of trusting patchwork's guide collection.
get_legend <- function(p) {
  g <- ggplotGrob(p)
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}
legend_grob <- get_legend(row_panel(
  df %>% filter(restaurant == "2-room"), bar_means %>% filter(restaurant == "2-room"),
  TRUE, FALSE, TRUE, "2-room"
))

p_2room <- row_panel(df %>% filter(restaurant == "2-room"), bar_means %>% filter(restaurant == "2-room"),
                      show_strip = TRUE, show_xaxis = FALSE, show_legend = FALSE, tag = "2-room")
p_3room <- row_panel(df %>% filter(restaurant == "3-room"), bar_means %>% filter(restaurant == "3-room"),
                      show_strip = FALSE, show_xaxis = TRUE, show_legend = FALSE, tag = "3-room")

combined <- p_2room / p_3room / wrap_elements(legend_grob) + plot_layout(heights = c(1, 1, 0.12))

ggsave(out_path, combined, width = 7.5, height = 5.6, device = cairo_pdf)
cat("wrote", out_path, "\n")
