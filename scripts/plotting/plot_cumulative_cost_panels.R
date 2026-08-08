# Cumulative PDDL cost vs. number of tasks, split into a 3-panel layout:
# (a) guided/deterministic methods, (b) greedy RL methods, (c) leaderboard
# of final (task-50) cumulative cost across all 8 methods.
#
# Same data sources, cumulative-cost computation, and base-R subsetting
# convention as plot_canonical_cost_ggdist.R (read that file's header first).
# The one deliberate departure: seed-variance ribbons here use
# ggdist::stat_lineribbon(.width = 1) (a literal min/max band across the 4
# checkpoint seeds), not the nested .5/.8/.95 quantile bands from the other
# script -- with only 4 seeds, nested quantiles imply tail knowledge the data
# doesn't support. This was locked in across a prior design discussion.

library(tidyverse)
library(ggdist)
library(ggrepel)
library(patchwork)

guided_csv <- "results/canonical_planner/planner/task_results.csv"
greedy_csv <- "results/canonical_planner/greedy_rl/task_results.csv"
out_path <- "results/canonical_planner/figures/cumulative_cost_panels.pdf"

fd_methods <- list(
  list(id = "myopic_fd_optimal", label = "Myopic Oracle", color = "#000000"),
  list(id = "clairvoyant_k3_lama", label = "Clairvoyant Oracle", color = "#E69F00")
)
guided_methods <- list(
  list(id = "myopic_dqn_beta1", label = "Myopic RL (guided)", color = "#56B4E9"),
  list(id = "anticipatory_dqn_beta1_25", label = "Anticipatory RL (guided)", color = "#009E73")
)
greedy_methods <- list(
  list(id = "myopic_dqn_greedy", label = "Myopic RL (greedy)", color = "#A6D8F0"),
  list(id = "anticipatory_dqn_greedy", label = "Anticipatory RL (greedy)", color = "#7FCFB4")
)
gnn_methods <- list(
  list(id = "gnn_faithful", label = "One-task GNN", color = "#CC79A7",
       path = "results/canonical_planner/gnn/faithful_seed0_seq_cost.csv"),
  list(id = "gnn_counterfactual", label = "One-task GNN (augmented)", color = "#D55E00",
       path = "results/canonical_planner/gnn/counterfactual_seed0_seq_cost.csv")
)

df_guided <- read_csv(guided_csv, show_col_types = FALSE)
df_greedy <- read_csv(greedy_csv, show_col_types = FALSE)

# Base-R subsetting (not dplyr::filter NSE) so target_method/target_seed/
# target_variant can never be shadowed by data's own same-named columns.
cumulative_curve <- function(data, target_method, target_seed = NULL, target_variant = NULL) {
  sub <- data[data$method_id == target_method, ]
  if (!is.null(target_seed)) sub <- sub[sub$checkpoint_seed == target_seed, ]
  if (!is.null(target_variant) && "checkpoint_variant" %in% names(sub)) {
    sub <- sub[sub$checkpoint_variant == target_variant, ]
  }
  sub %>%
    group_by(task_index) %>%
    summarise(avg_cost = mean(task_cost_pddl), .groups = "drop") %>%
    arrange(task_index) %>%
    mutate(cum_cost = cumsum(avg_cost), task = task_index + 1)
}

fd_curves <- map_dfr(fd_methods, function(m) {
  cumulative_curve(df_guided, m$id) %>% mutate(label = m$label)
})

gnn_curves <- map_dfr(gnn_methods, function(m) {
  d <- read_csv(m$path, show_col_types = FALSE) %>%
    rename(task_cost_pddl = pddl_cost) %>%
    mutate(method_id = m$id)
  cumulative_curve(d, m$id) %>% mutate(label = m$label)
})

guided_curves <- map_dfr(guided_methods, function(m) {
  seeds <- sort(unique(na.omit(df_guided$checkpoint_seed[df_guided$method_id == m$id])))
  map_dfr(seeds, function(s) {
    cumulative_curve(df_guided, m$id, target_seed = s) %>% mutate(label = m$label, seed = s)
  })
})

# Seed 16 uses the `best` checkpoint for myopic-greedy only (its `final`
# diverged during training); every other (method, seed) combo uses `final`.
greedy_curves <- map_dfr(greedy_methods, function(m) {
  seeds <- sort(unique(na.omit(df_greedy$checkpoint_seed[df_greedy$method_id == m$id])))
  map_dfr(seeds, function(s) {
    variant <- if (m$id == "myopic_dqn_greedy" && s == 16) "best" else "final"
    cumulative_curve(df_greedy, m$id, target_seed = s, target_variant = variant) %>%
      mutate(label = m$label, seed = s)
  })
})

fd_colors <- setNames(map_chr(fd_methods, "color"), map_chr(fd_methods, "label"))
gnn_colors <- setNames(map_chr(gnn_methods, "color"), map_chr(gnn_methods, "label"))
guided_colors <- setNames(map_chr(guided_methods, "color"), map_chr(guided_methods, "label"))
greedy_colors <- setNames(map_chr(greedy_methods, "color"), map_chr(greedy_methods, "label"))
all_colors <- c(fd_colors, gnn_colors, guided_colors, greedy_colors)

panel_a_deterministic <- bind_rows(fd_curves, gnn_curves)
panel_a_curves <- guided_curves
panel_a_colors <- c(fd_colors, gnn_colors, guided_colors)
panel_b_curves <- greedy_curves
panel_b_colors <- greedy_colors

x_range <- range(c(panel_a_deterministic$task, panel_a_curves$task, panel_b_curves$task))

det_end_labels <- function(det_curves) {
  det_curves %>%
    group_by(label) %>%
    filter(task == max(task)) %>%
    transmute(label, task, y = cum_cost)
}
seed_end_labels <- function(seed_curves) {
  seed_curves %>%
    group_by(label, task) %>%
    summarise(y = median(cum_cost), .groups = "drop") %>%
    group_by(label) %>%
    filter(task == max(task))
}

panel_a_labels <- bind_rows(det_end_labels(panel_a_deterministic), seed_end_labels(panel_a_curves))
panel_b_labels <- seed_end_labels(panel_b_curves)

base_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.4),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40")
  )

panel_a <- ggplot() +
  stat_lineribbon(
    data = panel_a_curves, aes(x = task, y = cum_cost, fill = label, color = label),
    .width = 1, alpha = 0.28, linewidth = 1.1, lineend = "round"
  ) +
  geom_line(
    data = fd_curves, aes(x = task, y = cum_cost, color = label),
    linetype = "dashed", linewidth = 1, lineend = "round"
  ) +
  geom_line(
    data = gnn_curves, aes(x = task, y = cum_cost, color = label),
    linetype = "dotted", linewidth = 1, lineend = "round"
  ) +
  geom_text_repel(
    data = panel_a_labels, aes(x = task, y = y, label = label, color = label),
    hjust = 0, nudge_x = 1.5, direction = "y", segment.size = 0.3,
    size = 3.3, fontface = "bold", show.legend = FALSE,
    max.overlaps = Inf, box.padding = 0.3
  ) +
  scale_color_manual(values = all_colors, guide = "none") +
  scale_fill_manual(values = panel_a_colors, guide = "none") +
  scale_x_continuous(breaks = scales::breaks_pretty(), expand = expansion(mult = c(0.01, 0.30))) +
  scale_y_continuous(labels = scales::label_comma()) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Cumulative PDDL cost", tag = "a") +
  base_theme +
  theme(plot.margin = margin(10, 130, 5, 10), axis.text.x = element_blank())

panel_b <- ggplot() +
  stat_lineribbon(
    data = panel_b_curves, aes(x = task, y = cum_cost, fill = label, color = label),
    .width = 1, alpha = 0.28, linewidth = 1.1, lineend = "round"
  ) +
  geom_text_repel(
    data = panel_b_labels, aes(x = task, y = y, label = label, color = label),
    hjust = 0, nudge_x = 1.5, direction = "y", segment.size = 0.3,
    size = 3.3, fontface = "bold", show.legend = FALSE,
    max.overlaps = Inf, box.padding = 0.3
  ) +
  scale_color_manual(values = all_colors, guide = "none") +
  scale_fill_manual(values = panel_b_colors, guide = "none") +
  scale_x_continuous(breaks = scales::breaks_pretty(), expand = expansion(mult = c(0.01, 0.30))) +
  scale_y_continuous(labels = scales::label_comma()) +
  coord_cartesian(clip = "off") +
  labs(x = "Number of tasks", y = "Cumulative PDDL cost", tag = "b") +
  base_theme +
  theme(plot.margin = margin(5, 130, 10, 10))

# --- Panel c: leaderboard of final (task-50) cumulative cost, all 8 methods.
final_task <- max(x_range)

det_final <- panel_a_deterministic %>%
  group_by(label) %>%
  filter(task == final_task) %>%
  summarise(value = cum_cost, .groups = "drop") %>%
  mutate(kind = "deterministic")

rl_final <- bind_rows(panel_a_curves, panel_b_curves) %>%
  group_by(label) %>%
  filter(task == final_task) %>%
  summarise(
    value = median(cum_cost),
    lo = min(cum_cost),
    hi = max(cum_cost),
    .groups = "drop"
  ) %>%
  mutate(kind = "rl")

leaderboard <- bind_rows(det_final, rl_final) %>%
  mutate(
    label = fct_reorder(label, value, .desc = TRUE),
    label_x = if_else(kind == "rl", hi, value)
  )

# Direct colored row labels (not axis text) -- avoids relying on axis-tick
# ordering to line up a separately-built color vector with factor levels.
panel_c <- ggplot(leaderboard, aes(x = value, y = label, color = label)) +
  geom_errorbarh(
    data = leaderboard %>% filter(kind == "rl"),
    aes(xmin = lo, xmax = hi), height = 0.28, linewidth = 0.9
  ) +
  geom_point(
    data = leaderboard %>% filter(kind == "rl"),
    shape = 16, size = 1.8
  ) +
  geom_point(
    data = leaderboard %>% filter(kind == "deterministic"),
    shape = 23, size = 3.2, fill = "white", stroke = 1.1
  ) +
  geom_text(
    aes(x = label_x, label = label), hjust = 0, nudge_x = diff(range(leaderboard$value)) * 0.03,
    size = 3.5, fontface = "bold", show.legend = FALSE
  ) +
  scale_color_manual(values = all_colors, guide = "none") +
  scale_x_continuous(
    labels = scales::label_comma(),
    expand = expansion(mult = c(0.02, 0.42))
  ) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Cumulative PDDL cost at task 50", y = NULL, tag = "c",
    caption = "diamonds: single deterministic run; circles: median across 4 seeds, whiskers = min-max"
  ) +
  base_theme +
  theme(
    plot.margin = margin(10, 15, 10, 5),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.caption = element_text(color = "gray30", size = 8.5, hjust = 0)
  )

left_col <- panel_a / panel_b + plot_layout(heights = c(2, 1))
combined <- left_col | panel_c
combined <- combined + plot_layout(widths = c(1.6, 1))

ggsave(out_path, combined, width = 15, height = 7.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
