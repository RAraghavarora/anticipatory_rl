# Greedy-vs-guided evaluation as paired per-seed slopes, not bars with an SD
# whisker. Same panel layout as plot_greedy_guided_grid.R (2 restaurants
# stacked x 3 metrics), but each of the 4 checkpoint seeds is evaluated both
# Greedy and Guided, so this draws the actual paired change per seed: one
# thin line per (method, seed) from its Greedy value to its Guided value,
# plus one bold mean line on top.
#
# Why not the bar+SD version: this repo's other seed-level figures
# (plot_cumulative_cost_panels.R, plot_qvalue_seeds_ggdist.R) deliberately
# use stat_lineribbon(.width = 1) -- literal min/max across 4 seeds -- and
# say why in their headers: at n=4, a computed spread (like SD) implies
# distributional knowledge the data doesn't support. A bar+SD errorbar
# makes exactly that unsupported claim. Showing the 4 raw seed lines has no
# such problem and additionally shows whether every seed improves (not just
# the mean) -- the more relevant question when arguing guided beats greedy.
#
# Data: results/v5/figures/greedy_guided_per_seed.csv, built by
# scripts/restaurant/build_greedy_guided_summary.py.

library(tidyverse)
library(patchwork)

df <- read_csv("results/v5/figures/greedy_guided_per_seed.csv", show_col_types = FALSE) %>%
  filter(metric != "Success") %>%
  mutate(
    metric = factor(metric, levels = c("Auto%", "Steps/task", "Cost/task")),
    deployment = factor(deployment, levels = c("Greedy", "Guided")),
    method = factor(method, levels = c("Myopic RL", "Anticipatory RL"))
  )

out_path <- "results/v5/figures/greedy_guided_slopes.pdf"
colors <- c("Myopic RL" = "#56B4E9", "Anticipatory RL" = "#009E73")
method_offset <- c("Myopic RL" = -0.08, "Anticipatory RL" = 0.08)

df <- df %>%
  mutate(x_pos = as.numeric(deployment) + method_offset[as.character(method)])

means <- df %>%
  group_by(restaurant, metric, method, deployment, x_pos) %>%
  summarise(value = mean(value), .groups = "drop")

label_fmt <- function(metric, value) {
  case_when(
    metric == "Cost/task" ~ scales::label_comma(accuracy = 1)(value),
    metric == "Auto%" ~ paste0(scales::label_number(accuracy = 0.1)(value), "%"),
    TRUE ~ scales::label_number(accuracy = 0.01)(value)
  )
}
means <- means %>% mutate(value_label = label_fmt(metric, value))

base_theme <- theme_minimal(base_size = 12) +
  theme(
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.text.y = element_text(color = "gray40", size = 8),
    panel.spacing = unit(1.1, "lines"),
    plot.margin = margin(6, 16, 4, 6)
  )

row_panel <- function(data, means_data, show_strip, show_xaxis, show_legend, tag) {
  ggplot() +
    geom_line(
      data = data, aes(x = x_pos, y = value, group = interaction(method, seed), color = method),
      linewidth = 0.4, alpha = 0.35
    ) +
    geom_point(
      data = data, aes(x = x_pos, y = value, color = method),
      size = 1.3, alpha = 0.45
    ) +
    geom_line(
      data = means_data, aes(x = x_pos, y = value, group = method, color = method),
      linewidth = 1.3
    ) +
    geom_point(
      data = means_data, aes(x = x_pos, y = value, color = method),
      size = 2.6
    ) +
    geom_text(
      data = means_data, aes(x = x_pos, y = value, label = value_label, color = method),
      size = 2.9, fontface = "bold", show.legend = FALSE,
      hjust = ifelse(means_data$deployment == "Greedy", 1.25, -0.25)
    ) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_color_manual(values = colors, name = NULL, drop = FALSE) +
    scale_x_continuous(
      breaks = c(1, 2), labels = c("Greedy", "Guided"),
      limits = c(1 - 0.45, 2 + 0.45), expand = expansion(mult = c(0.08, 0.08))
    ) +
    labs(x = NULL, y = tag) +
    guides(color = guide_legend(override.aes = list(alpha = 1, linewidth = 1.3, size = 2.6))) +
    base_theme +
    theme(
      strip.text = if (show_strip) element_text(face = "bold", size = 10.5, color = "gray20") else element_blank(),
      axis.text.x = if (show_xaxis) element_text(color = "gray30") else element_blank(),
      axis.title.y = element_text(face = "bold", size = 10.5, color = "gray20", angle = 90),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = 10)
    )
}

get_legend <- function(p) {
  g <- ggplotGrob(p)
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}
legend_grob <- get_legend(row_panel(
  df %>% filter(restaurant == "2-room"), means %>% filter(restaurant == "2-room"),
  TRUE, FALSE, TRUE, "2-room"
))

p_2room <- row_panel(df %>% filter(restaurant == "2-room"), means %>% filter(restaurant == "2-room"),
                      show_strip = TRUE, show_xaxis = FALSE, show_legend = FALSE, tag = "2-room")
p_3room <- row_panel(df %>% filter(restaurant == "3-room"), means %>% filter(restaurant == "3-room"),
                      show_strip = FALSE, show_xaxis = TRUE, show_legend = FALSE, tag = "3-room")

combined <- p_2room / p_3room / wrap_elements(legend_grob) + plot_layout(heights = c(1, 1, 0.12))

ggsave(out_path, combined, width = 9.5, height = 5.6, device = cairo_pdf)
cat("wrote", out_path, "\n")
