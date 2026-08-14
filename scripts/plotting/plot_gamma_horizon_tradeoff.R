# Discount factor sweep, v5: jar-delivery rate and greedy success rate vs.
# effective horizon gamma/(1-gamma). Two stacked panels sharing one x-axis,
# not a dual-y-axis single panel -- jar% and success rate are different
# units, and a shared x-axis with separate y-scales is the correct form for
# that (a dual-axis chart is the #1 chart-design mistake for exactly this
# situation).
#
# Data is the 6-row gamma sweep table given directly (no source file to
# derive it from -- hardcoded here rather than building a throwaway CSV for
# 6 numbers). gamma=0.96's jar% (14.5) dips below its neighbors (21.3 at
# 0.95, 32.9 at 0.97) -- a real non-monotonicity in the swept data, not a
# typo -- though it's no longer visually flagged (no distinct color or
# annotation, removed on request); the gamma label on that point is the
# only remaining way to spot it.
#
# Each point is labeled with its gamma value directly (the x-axis alone,
# gamma/(1-gamma), doesn't read back to gamma without doing the arithmetic
# in your head).

library(tidyverse)
library(patchwork)

df <- tribble(
  ~gamma, ~jar_pct, ~success,
  0.90, 4.6, 0.783,
  0.95, 21.3, 0.842,
  0.96, 14.5, 0.826,
  0.97, 32.9, 0.815,
  0.98, 65.2, 0.750,
  0.99, 73.0, 0.661
) %>%
  mutate(x = gamma / (1 - gamma))

out_path <- "results/v5/figures/gamma_horizon_tradeoff.pdf"
jar_color <- "#8856A7"
success_color <- "#009E73"

base_theme <- theme_minimal(base_size = 12.5) +
  theme(
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(size = 10.5, color = "gray30"),
    axis.text = element_text(color = "gray40"),
    plot.margin = margin(6, 12, 4, 8)
  )

panel_jar <- ggplot(df, aes(x = x, y = jar_pct)) +
  geom_line(color = jar_color, linewidth = 0.9) +
  geom_point(color = jar_color, size = 2.8) +
  geom_text(aes(label = sprintf("%.2f", gamma)), nudge_y = 4, size = 3.1, color = "gray30", fontface = "bold") +
  scale_y_continuous(limits = c(0, 85), expand = expansion(mult = c(0.02, 0.1))) +
  labs(x = NULL, y = "Water drawn from jar (%)") +
  base_theme +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

panel_success <- ggplot(df, aes(x = x, y = success)) +
  geom_line(color = success_color, linewidth = 0.9) +
  geom_point(color = success_color, size = 2.8) +
  geom_text(aes(label = sprintf("%.2f", gamma)), nudge_y = 0.025, size = 3.1, color = "gray30", fontface = "bold") +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0.6, 0.9)) +
  labs(x = expression(gamma / (1 - gamma) ~ "  (effective horizon)"), y = "Task success rate") +
  base_theme

combined <- panel_jar / panel_success + plot_layout(heights = c(1, 1))

ggsave(out_path, combined, width = 6.5, height = 6, device = cairo_pdf)
cat("wrote", out_path, "\n")
