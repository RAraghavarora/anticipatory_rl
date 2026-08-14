# Advantage over Myopic Oracle (%), all v5 methods -- a plain horizontal
# bar leaderboard, not the diamond/circle/errorbar lollipop mix this
# replaces (that version reused panel_c's mark language from
# plot_cumulative_cost_panels.R, but mixing two point shapes, 9 mostly-
# distinct hues including a couple of washed-out pastel tints, and
# staggered per-row label positions didn't read cleanly here).
#
# Simpler recipe, matching the well-received seq_comparison.R delta chart:
# one bar per method, ordered by advantage, direct end labels, thin
# error bars only where real seed variance exists. Color families group
# related methods (Myopic=black, Oracle=amber, GNN=pink/vermillion,
# RL=teal); a method's weaker/ablated variant reuses its flagship's hue at
# lower alpha (same convention as plot_greedy_guided_grid.R's greedy/guided
# tint) instead of a separate pastel hex -- fewer colors to tell apart.
#
# Advantage = (myopic_total - method_total) / myopic_total * 100 over each
# method's 10-sequence x 50-task canonical run -- positive means cheaper
# than pure myopic (K=1) planning. Myopic Oracle itself is the zero-line,
# plotted as its own anchor bar (zero-length), not omitted.
#
# Data: results/v5/figures/advantage_over_myopic.csv, built by
# scripts/restaurant/build_v5_advantage.py (recomputed from the current
# 4-seed GNN sweep in run_summary.csv, not RESULTS.md's still-stale
# single-seed-42 headline numbers).

library(tidyverse)

methods <- tribble(
  ~label, ~color, ~alpha,
  "Myopic Oracle", "#000000", 1.0,
  "K=2 Optimal", "#0072B2", 1.0,
  "One-task GNN", "#CC79A7", 1.0,
  "One-task GNN (augmented)", "#D55E00", 1.0,
  "Anticipatory RL (guided)", "#009E73", 1.0,
  "K=3 Clairvoyant Oracle", "#E69F00", 1.0,
  "K=4 Clairvoyant Oracle", "#E69F00", 1.0
)

df <- read_csv("results/v5/figures/advantage_over_myopic.csv", show_col_types = FALSE) %>%
  inner_join(methods, by = "label") %>%
  mutate(
    has_range = n > 1,
    label = factor(label, levels = rev(label[order(mean_adv)])),
    bar_label = sprintf("%+.1f%%", mean_adv)
  )
level_order <- levels(df$label)

out_path <- "results/v5/figures/advantage_over_myopic.pdf"

p <- ggplot(df, aes(x = mean_adv, y = label)) +
  geom_col(aes(fill = label, alpha = label), width = 0.65) +
  geom_errorbarh(
    data = df %>% filter(has_range),
    aes(xmin = min_adv, xmax = max_adv), height = 0.25, linewidth = 0.6, color = "gray30"
  ) +
  geom_text(
    aes(label = bar_label, x = pmax(mean_adv, if_else(has_range, max_adv, mean_adv))),
    hjust = -0.15, size = 3.6, fontface = "bold", color = "gray20"
  ) +
  scale_fill_manual(values = setNames(df$color, df$label), guide = "none") +
  scale_alpha_manual(values = setNames(df$alpha, df$label), guide = "none") +
  scale_y_discrete(limits = level_order) +
  scale_x_continuous(labels = scales::label_percent(scale = 1), expand = expansion(mult = c(0, 0.16))) +
  labs(
    x = "Advantage over Myopic Oracle: (myopic cost − method cost) / myopic cost",
    y = NULL,
    caption = paste(
      "Error bars: seed min-max (4 seeds) where a sweep exists; single-run methods (K=2/K=3/K=4) have none.",
      "K=3/K=4 are satisficing UPPER BOUNDS (window-budget cap), so their advantage is optimistic, not exact.",
      sep = "\n"
    )
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.3),
    axis.text.y = element_text(face = "bold", size = 11, color = "gray20"),
    axis.title.x = element_text(color = "gray30", size = 10.5),
    axis.text.x = element_text(color = "gray40"),
    plot.margin = margin(10, 20, 8, 8),
    plot.caption = element_text(color = "gray40", size = 8, hjust = 0, lineheight = 1.3)
  )

ggsave(out_path, p, width = 9, height = 6, device = cairo_pdf)
cat("wrote", out_path, "\n")
