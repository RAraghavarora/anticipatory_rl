# Cumulative PDDL cost vs. number of tasks, all 8 methods, one compact panel.
#
# Design (see thesis-figure discussion): a single dense panel, not small
# multiples -- several methods are genuinely near-coincident (e.g. Myopic
# Oracle / Myopic RL guided / One-task GNN all track within ~1%), and that
# overlap is itself the finding, not a rendering problem to engineer around.
# Distinguishing encoding: color = method, linetype = method family (oracle
# dashed, GNN dotted, guided RL solid, greedy RL dot-dash), shape = individual
# method (sparse markers every 5 tasks) -- so even fully-overlapping lines
# stay identifiable by marker shape peeking through. No end-of-line text
# labels; one compact multi-column legend instead. No in-figure title/
# subtitle/caption -- that text belongs in the external LaTeX caption.
#
# Guided methods (myopic_dqn_beta1, anticipatory_dqn_beta1_25) come from
# results/canonical_planner/planner/task_results.csv, pooling all 4
# checkpoint seeds. Greedy methods (direct policy rollout, no FD guidance)
# come from results/canonical_planner/greedy_rl/task_results.csv; every
# (method, seed) combo uses its `final` checkpoint (seed 16's checkpoint was
# retrained after a training-instability bug was fixed -- see manifest.json
# "corrections" -- so the old best-checkpoint workaround no longer applies).
# For the 4 RL methods, stat_lineribbon(.width = 1) gives the min/max band
# across the 4 per-seed curves -- a literal statement of observed spread,
# not a fabricated distributional claim the way a nested-quantile or
# parametric band would be at n=4. GNN variants are single-seed (seed 0,
# matching this dataset's eval_seed convention -- not the seed-42 data in
# the separate wt-gnn repo), so like the FD oracles they get a plain line.

library(tidyverse)
library(ggdist)

guided_csv <- "results/canonical_planner/planner/task_results.csv"
greedy_csv <- "results/canonical_planner/greedy_rl/task_results.csv"
out_path <- "results/canonical_planner/figures/cumulative_cost_ggdist.pdf"
MARKER_EVERY <- 5

methods <- tribble(
  ~id, ~label, ~color, ~family,
  "myopic_fd_optimal", "Myopic Oracle", "#000000", "oracle",
  "clairvoyant_k3_lama", "Clairvoyant Oracle", "#E69F00", "oracle",
  "gnn_faithful", "One-task GNN", "#CC79A7", "gnn",
  "gnn_counterfactual", "One-task GNN (augmented)", "#D55E00", "gnn",
  "myopic_dqn_beta1", "Myopic RL (guided)", "#56B4E9", "guided",
  "anticipatory_dqn_beta1_25", "Anticipatory RL (guided)", "#009E73", "guided",
  "myopic_dqn_greedy", "Myopic RL (greedy)", "#A6D8F0", "greedy",
  "anticipatory_dqn_greedy", "Anticipatory RL (greedy)", "#7FCFB4", "greedy"
) %>%
  mutate(label = factor(label, levels = label))

family_linetype <- c(oracle = "dashed", gnn = "dotted", guided = "solid", greedy = "dotdash")
linetypes <- setNames(family_linetype[methods$family], methods$label)
colors <- setNames(methods$color, methods$label)
shapes <- setNames(c(16, 17, 15, 18, 3, 4, 8, 1), methods$label)

df_guided <- read_csv(guided_csv, show_col_types = FALSE)
df_greedy <- read_csv(greedy_csv, show_col_types = FALSE)

# Base-R subsetting (not dplyr::filter NSE) so target_method/target_seed
# can never be shadowed by the data's own same-named columns.
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

deterministic_ids <- c("myopic_fd_optimal", "clairvoyant_k3_lama")
gnn_specs <- list(
  gnn_faithful = "results/canonical_planner/gnn/faithful_seed0_seq_cost.csv",
  gnn_counterfactual = "results/canonical_planner/gnn/counterfactual_seed0_seq_cost.csv"
)
rl_ids <- c("myopic_dqn_beta1", "anticipatory_dqn_beta1_25", "myopic_dqn_greedy", "anticipatory_dqn_greedy")

deterministic_curves <- map_dfr(deterministic_ids, function(id) {
  m <- methods[methods$id == id, ]
  cumulative_curve(df_guided, id) %>% mutate(label = m$label)
})

gnn_curves <- imap_dfr(gnn_specs, function(path, id) {
  m <- methods[methods$id == id, ]
  d <- read_csv(path, show_col_types = FALSE) %>%
    rename(task_cost_pddl = pddl_cost) %>%
    mutate(method_id = id)
  cumulative_curve(d, id) %>% mutate(label = m$label)
})

rl_curves <- map_dfr(rl_ids, function(id) {
  m <- methods[methods$id == id, ]
  data <- if (id %in% c("myopic_dqn_greedy", "anticipatory_dqn_greedy")) df_greedy else df_guided
  seeds <- sort(unique(na.omit(data$checkpoint_seed[data$method_id == id])))
  map_dfr(seeds, function(s) {
    cumulative_curve(data, id, target_seed = s, target_variant = "final") %>%
      mutate(label = m$label, seed = s)
  })
})

line_curves <- bind_rows(deterministic_curves, gnn_curves)
marker_curves <- bind_rows(line_curves, rl_curves %>% group_by(label, task) %>% summarise(cum_cost = median(cum_cost), .groups = "drop")) %>%
  filter(task %% MARKER_EVERY == 0 | task == 1)

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
  labs(x = "Number of tasks", y = "Cumulative PDDL cost") +
  guides(color = guide_legend(
    ncol = 4,
    override.aes = list(
      linetype = linetypes[levels(methods$label)],
      shape = shapes[levels(methods$label)],
      alpha = 1,
      fill = NA
    )
  )) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    legend.key.width = unit(1.8, "lines"),
    legend.spacing.x = unit(0.4, "lines"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40"),
    plot.margin = margin(8, 10, 4, 6)
  )

ggsave(out_path, p, width = 8.5, height = 5, device = cairo_pdf)
cat("wrote", out_path, "\n")
