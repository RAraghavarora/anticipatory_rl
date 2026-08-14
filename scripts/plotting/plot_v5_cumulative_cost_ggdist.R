# Cumulative PDDL cost vs. number of tasks, v5 (3-room), via ggdist --
# standalone Julia-sibling of plot_canonical_cost_ggdist.R's 2-room figure,
# same design: one dense panel, color = method, linetype = method family
# (oracle dashed, GNN dotted, guided solid, greedy dot-dash), shape =
# individual method (sparse markers every 5 tasks), stat_lineribbon
# (.width = 1) for the 4-seed RL arms' min/max spread.
#
# Deliberately its own standalone script/process, not merged into one R
# session with the 2-room script: building two guide_legend() calls with
# different-length override.aes vectors in the same session previously
# crashed inside ggplot2's guide-building (confirmed as a real ggplot2
# state-leak bug, not a data or logic error, by verifying panels render
# cleanly in isolation). Now that v5 has all 8 of 2-room's methods too, that
# specific length mismatch no longer applies, but this stays standalone
# regardless -- no reason to retest the ggplot2 bug boundary. The two PDFs
# get composited into one figure afterward at the shell level (image stack
# + reflatten), not inside R.
#
# Myopic RL is a partial seed sweep -- guided has seeds {0, 8}, greedy has
# {0, 8, 16} (seed 4 pending for both). Data:
# results/v5/figures/cumulative_cost_per_task.csv, built by
# scripts/restaurant/build_v5_cumulative_cost.py.

library(tidyverse)
library(ggdist)

MARKER_EVERY <- 5
out_path <- "results/v5/figures/cumulative_cost_ggdist_v5.pdf"

methods <- tribble(
  ~id, ~label, ~color, ~family, ~shape,
  "myopic_fd_optimal", "Myopic Oracle", "#000000", "oracle", 16,
  "clairvoyant_k3_lama", "Clairvoyant Oracle", "#E69F00", "oracle", 17,
  "gnn_faithful", "One-task GNN", "#CC79A7", "gnn", 15,
  "gnn_counterfactual", "One-task GNN (augmented)", "#D55E00", "gnn", 18,
  "myopic_dqn_guided", "Myopic RL (guided)", "#56B4E9", "guided", 3,
  "anticipatory_dqn_guided", "Anticipatory RL (guided)", "#009E73", "guided", 4,
  "myopic_dqn_greedy", "Myopic RL (greedy)", "#A6D8F0", "greedy", 8,
  "anticipatory_dqn_greedy", "Anticipatory RL (greedy)", "#7FCFB4", "greedy", 1
) %>%
  mutate(
    linetype = recode(family, oracle = "dashed", gnn = "dotted", guided = "solid", greedy = "dotdash"),
    label = factor(label, levels = label)
  )

df <- read_csv("results/v5/figures/cumulative_cost_per_task.csv", show_col_types = FALSE)

cumulative_curve <- function(target_method, per_seed = FALSE) {
  sub_method <- df[df$method_id == target_method, ]
  one_curve <- function(sub) {
    sub %>%
      group_by(task_index) %>%
      summarise(avg_cost = mean(cost), .groups = "drop") %>%
      arrange(task_index) %>%
      mutate(cum_cost = cumsum(avg_cost), task = task_index + 1)
  }
  if (!per_seed) return(one_curve(sub_method))
  seeds <- sort(unique(na.omit(sub_method$seed)))
  map_dfr(seeds, function(s) one_curve(sub_method[sub_method$seed == s, ]) %>% mutate(seed = s))
}

deterministic_ids <- methods$id[methods$family %in% c("oracle", "gnn")]
rl_ids <- methods$id[methods$family %in% c("guided", "greedy")]

line_curves <- map_dfr(deterministic_ids, function(id) {
  m <- methods[methods$id == id, ]
  cumulative_curve(id) %>% mutate(label = m$label)
})
rl_curves <- map_dfr(rl_ids, function(id) {
  m <- methods[methods$id == id, ]
  cumulative_curve(id, per_seed = TRUE) %>% mutate(label = m$label)
})

linetypes <- setNames(methods$linetype, methods$label)
colors <- setNames(methods$color, methods$label)
shapes <- setNames(methods$shape, methods$label)

marker_curves <- bind_rows(
  line_curves,
  rl_curves %>% group_by(label, task) %>% summarise(cum_cost = median(cum_cost), .groups = "drop")
) %>% filter(task %% MARKER_EVERY == 0 | task == 1)

p <- ggplot() +
  stat_lineribbon(
    data = rl_curves, aes(x = task, y = cum_cost, color = label, fill = label, linetype = label),
    .width = 1, alpha = 0.18, linewidth = 0.7
  ) +
  geom_line(
    data = line_curves, aes(x = task, y = cum_cost, color = label, linetype = label),
    linewidth = 0.7
  ) +
  geom_point(
    data = marker_curves, aes(x = task, y = cum_cost, color = label, shape = label),
    size = 1.8, stroke = 0.6
  ) +
  scale_color_manual(values = colors, breaks = levels(methods$label), name = NULL) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_linetype_manual(values = linetypes, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(x = "Number of tasks", y = "Cumulative PDDL cost", subtitle = "3-room") +
  guides(color = guide_legend(
    ncol = 4,
    override.aes = list(
      linetype = linetypes[levels(methods$label)],
      shape = shapes[levels(methods$label)],
      alpha = 1, fill = NA
    )
  )) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    legend.key.width = unit(1.8, "lines"),
    legend.spacing.x = unit(0.4, "lines"),
    plot.subtitle = element_text(face = "bold", size = 11, color = "gray20"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40"),
    plot.margin = margin(8, 10, 4, 6)
  )

ggsave(out_path, p, width = 8.5, height = 5, device = cairo_pdf)
cat("wrote", out_path, "\n")
