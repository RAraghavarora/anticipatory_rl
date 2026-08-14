# Greedy-vs-guided evaluation grid: 2 restaurants (stacked rows) x 3 metrics
# (columns), reproducing tab:greedy-guided as small multiples instead of a
# 6-metric, 8-row table. Data: results/v5/figures/greedy_guided_grid.csv,
# built by scripts/restaurant/build_greedy_guided_summary.py from the same
# source files the table was built from.
#
# Two facet_wrap(~metric) rows stacked with patchwork, not one
# facet_grid(restaurant ~ metric): facet_grid's free_y only frees the scale
# per row, not per cell, so Auto%/Steps would still share Cost/task's 0-1500
# axis and render as invisible slivers. facet_wrap frees every panel
# independently, which is what three different-unit metrics need.
#
# Success is dropped (trivially 100% by construction for every guided bar,
# ~95-98% and barely varying for greedy). 3-room myopic (both deployments)
# has no data yet and is omitted rather than zero-filled; the missing bars
# are the fact, not a rendering gap.
#
# Color = credit horizon (2 fixed hues, matching every other figure in this
# repo: Myopic RL blue, Anticipatory RL green). Deployment (Greedy/Guided) is
# the x position, not a third color, and is additionally distinguished by
# fill alpha (greedy = tint, guided = full saturation) so a grayscale print
# still separates the two.

library(tidyverse)
library(patchwork)

df <- read_csv("results/v5/figures/greedy_guided_grid.csv", show_col_types = FALSE) %>%
  filter(metric != "Success") %>%
  mutate(
    metric = factor(metric, levels = c("Auto%", "Steps/task", "Cost/task")),
    deployment = factor(deployment, levels = c("Greedy", "Guided")),
    method = factor(method, levels = c("Myopic RL", "Anticipatory RL"))
  )

out_path <- "results/v5/figures/greedy_guided_grid.pdf"
colors <- c("Myopic RL" = "#56B4E9", "Anticipatory RL" = "#009E73")

label_fmt <- function(metric, value) {
  case_when(
    metric == "Cost/task" ~ scales::label_comma(accuracy = 1)(value),
    metric == "Auto%" ~ paste0(scales::label_number(accuracy = 0.1)(value), "%"),
    TRUE ~ scales::label_number(accuracy = 0.01)(value)
  )
}
df <- df %>% mutate(bar_label = label_fmt(metric, mean))

base_theme <- theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10.5, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.text.x = element_text(color = "gray30"),
    axis.text.y = element_text(color = "gray40", size = 8),
    panel.spacing = unit(1.1, "lines"),
    plot.margin = margin(6, 12, 4, 6)
  )

row_panel <- function(data, show_strip, show_xaxis, show_legend, tag) {
  ggplot(data, aes(x = deployment, y = mean, fill = method, alpha = deployment, group = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65, color = NA) +
    geom_errorbar(
      aes(ymin = mean - sd, ymax = mean + sd),
      position = position_dodge(width = 0.75), width = 0.18, linewidth = 0.5,
      color = "gray25", alpha = 1
    ) +
    geom_text(
      aes(label = bar_label, y = mean + sd),
      position = position_dodge(width = 0.75), vjust = -0.6, size = 2.9,
      color = "gray20", alpha = 1, fontface = "bold"
    ) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = colors, name = NULL, drop = FALSE) +
    scale_alpha_manual(values = c(Greedy = 0.55, Guided = 1.0), guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
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

# The 3-room panel only has Anticipatory RL data, so a legend built from it
# renders "Myopic RL" as a blank swatch (no row exists to supply the paired
# alpha aesthetic). Building the shared legend from the 2-room panel (which
# has real data for both methods) and placing it as its own row avoids that,
# and avoids patchwork guides = "collect", which doesn't merge two legends
# with different key counts anyway.
get_legend <- function(p) {
  g <- ggplotGrob(p)
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}
legend_grob <- get_legend(row_panel(df %>% filter(restaurant == "2-room"), TRUE, FALSE, TRUE, "2-room"))

p_2room <- row_panel(df %>% filter(restaurant == "2-room"), show_strip = TRUE, show_xaxis = FALSE, show_legend = FALSE, tag = "2-room")
p_3room <- row_panel(df %>% filter(restaurant == "3-room"), show_strip = FALSE, show_xaxis = TRUE, show_legend = FALSE, tag = "3-room")

combined <- p_2room / p_3room / wrap_elements(legend_grob) + plot_layout(heights = c(1, 1, 0.12))

ggsave(out_path, combined, width = 9.5, height = 5.6, device = cairo_pdf)
cat("wrote", out_path, "\n")
