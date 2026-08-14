# Per-task-type cost breakdown, 2-room and 3-room (v5) stacked as two rows
# of the same small-multiples layout -- not a new chart type, an extension
# of plot_task_type_breakdown.R (which is left untouched and still produces
# its own standalone figure).
#
# Why stacked rows and not a bigger grid or a new encoding: the two configs
# share the exact same 6 task_type values (verified before writing this),
# so the columns line up directly -- task_type stays the facet dimension,
# same as the original, and restaurant becomes a second stacked row via
# patchwork, exactly like plot_greedy_guided_grid.R and
# plot_greedy_guided_bars_ggdist.R earlier in this session. This is not the
# "2x2 grid" that was rejected for the slope figure -- that objection was
# about a cramped 2x2 from only 2 metrics; 6 task-type columns is squarely
# the many-small-panels convention the original figure already uses, just
# with a second row.
#
# 3-room only has Anticipatory RL (guided) and One-task GNN (augmented) --
# Myopic RL (guided) has no v5 run yet, so that row shows two bars per panel
# instead of three. Omitted, not zero-filled or faked to match the top row.
#
# Aggregation matches everywhere in this figure: mean cost per (seed,
# task_type) first, then mean +/- SEM across the 4 checkpoint seeds. The GNN
# used to be single-seed in 2-room (results/canonical_planner/gnn/*_seed0_
# seq_cost.csv, now under superseded_seed42/) but was re-run across the same
# 4-seed sweep as every other arm -- see results/canonical_planner/gnn/
# README.md -- so seed_task_costs.csv now gives it a real error bar in both
# rows, on equal footing with the DQN arms.

library(tidyverse)
library(patchwork)

out_path <- "results/v5/figures/thesis/task_type_breakdown_v5.pdf"

methods <- tribble(
  ~method_id, ~label, ~color,
  "myopic_dqn_beta1_25", "Myopic RL (guided)", "#56B4E9",  # 2-room, beta=1.25 (matched)
  "anticipatory_dqn_beta1_25", "Anticipatory RL (guided)", "#009E73",
  "gnn_counterfactual", "One-task GNN (augmented)", "#D55E00"
)
colors <- setNames(methods$color, methods$label)

# --- 2-room: identical computation to plot_task_type_breakdown.R, plus the
# 4-seed GNN augmented arm (results/canonical_planner/gnn/seed_task_costs.csv).
# Myopic RL (guided) beta=1.25 per-task rows come from the supplement CSV
# (task_results.csv still carries only the old beta=1.00 arm). ---
df_2room <- bind_rows(
  read_csv("results/canonical_planner/planner/task_results.csv", show_col_types = FALSE),
  read_csv("results/canonical_planner/planner/myopic_b125_per_task.csv", show_col_types = FALSE)
) %>%
  semi_join(methods, by = "method_id")

per_seed_2room <- df_2room %>%
  group_by(method_id, checkpoint_seed, task_type) %>%
  summarise(mean_cost = mean(task_cost_pddl), .groups = "drop")

summary_2room_dqn <- per_seed_2room %>%
  group_by(method_id, task_type) %>%
  summarise(sem = sd(mean_cost) / sqrt(n()), mean_cost = mean(mean_cost), .groups = "drop")

summary_2room_gnn <- read_csv("results/canonical_planner/gnn/seed_task_costs.csv", show_col_types = FALSE) %>%
  filter(method_id == "gnn_counterfactual") %>%
  group_by(method_id, seed, task_type) %>%
  summarise(mean_cost = mean(pddl_cost), .groups = "drop") %>%
  group_by(method_id, task_type) %>%
  summarise(sem = sd(mean_cost) / sqrt(n()), mean_cost = mean(mean_cost), .groups = "drop")

summary_2room <- bind_rows(summary_2room_dqn, summary_2room_gnn) %>%
  inner_join(methods, by = "method_id") %>%
  mutate(label = factor(label, levels = methods$label))

task_type_levels <- sort(unique(summary_2room$task_type))

# --- 3-room (v5): pre-aggregated by scripts/restaurant/build_v5_task_type_breakdown.py ---
summary_3room <- read_csv("results/v5/figures/task_type_breakdown.csv", show_col_types = FALSE) %>%
  mutate(label = factor(label, levels = methods$label), task_type = factor(task_type, levels = task_type_levels))

summary_2room <- summary_2room %>% mutate(task_type = factor(task_type, levels = task_type_levels))

base_theme <- theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.title.y = element_text(face = "bold", color = "gray20", size = 10),
    axis.text.y = element_text(color = "gray40", size = 7.5),
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(6, 12, 4, 10)
  )

row_panel <- function(data, show_strip, tag) {
  ggplot(data, aes(x = label, y = mean_cost, fill = label)) +
    geom_col(width = 0.65, alpha = 0.85) +
    geom_errorbar(
      data = data %>% filter(!is.na(sem)),
      aes(ymin = mean_cost - sem, ymax = mean_cost + sem), width = 0.3, linewidth = 0.6, color = "black"
    ) +
    facet_wrap(~task_type, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = colors, name = NULL, drop = FALSE) +
    scale_y_continuous(labels = scales::label_comma()) +
    labs(x = NULL, y = tag) +
    base_theme +
    theme(
      strip.text = if (show_strip) element_text(face = "bold", size = 9, color = "gray20") else element_blank(),
      legend.position = "none"
    )
}

get_legend <- function(p) {
  g <- ggplotGrob(p + theme(legend.position = "bottom"))
  g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
}
legend_grob <- get_legend(row_panel(summary_2room, show_strip = TRUE, tag = "2-room"))

p_2room <- row_panel(summary_2room, show_strip = TRUE, tag = "2-room · mean PDDL cost/task")
p_3room <- row_panel(summary_3room, show_strip = FALSE, tag = "3-room · mean PDDL cost/task")

combined <- p_2room / p_3room / wrap_elements(legend_grob) + plot_layout(heights = c(1, 1, 0.1))

ggsave(out_path, combined, width = 15, height = 6.4, device = cairo_pdf)
cat("wrote", out_path, "\n")
