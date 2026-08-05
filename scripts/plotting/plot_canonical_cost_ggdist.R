# Cumulative PDDL cost vs. number of tasks, via ggdist::stat_lineribbon.
#
# Same computation as scripts/restaurant/plot_canonical_cost.py: at each
# task_index, cost is averaged across the 10 canonical sequences, then
# cumsum'd over task_index, per (method, checkpoint_seed). For each of the 4
# DQN methods, the per-seed cumulative curves are the "draws" that
# stat_lineribbon turns into a median line + nested quantile ribbons (.5/.8/.95)
# -- a more honest uncertainty picture at n=4 than a parametric mean+-std band.
#
# Guided methods (myopic_dqn_beta1, anticipatory_dqn_beta1_25) come from
# results/canonical_planner/task_results.csv and pool all 4 checkpoint seeds
# as-is. Greedy methods (direct policy rollout, no FD guidance) come from
# results/canonical_planner/greedy_direct_task_results.csv; myopic-greedy uses
# seed 16's `best` checkpoint variant instead of `final` -- the final
# checkpoint diverged late in training (see train_summary.json /
# metrics.csv for v3_myopic_g0.97_peb_s16) and pooling that outlier in would
# blow out the whole plot's axis. All other seeds/methods use `final` (the
# only variant they have).

library(tidyverse)
library(ggdist)
library(ggrepel)

guided_csv <- "results/canonical_planner/task_results.csv"
greedy_csv <- "results/canonical_planner/greedy_direct_task_results.csv"
out_path <- "results/canonical_planner/cumulative_cost_ggdist.pdf"

fd_methods <- list(
  list(id = "myopic_fd_optimal", label = "Myopic FD (optimal)", color = "#000000"),
  list(id = "clairvoyant_k3_lama", label = "Clairvoyant FD (K=3)", color = "#E69F00")
)
guided_methods <- list(
  list(id = "myopic_dqn_beta1", label = "Myopic DQN, guided", color = "#56B4E9"),
  list(id = "anticipatory_dqn_beta1_25", label = "Anticipatory DQN, guided", color = "#009E73")
)
greedy_methods <- list(
  list(id = "myopic_dqn_greedy", label = "Myopic DQN, greedy", color = "#A6D8F0"),
  list(id = "anticipatory_dqn_greedy", label = "Anticipatory DQN, greedy", color = "#7FCFB4")
)

df_guided <- read_csv(guided_csv, show_col_types = FALSE)
df_greedy <- read_csv(greedy_csv, show_col_types = FALSE)

# Base-R subsetting (not dplyr::filter NSE) so target_method/target_seed/
# target_variant can never be shadowed by data's own same-named columns.
# target_variant is ignored when data has no checkpoint_variant column
# (the guided CSV doesn't; every guided seed has exactly one checkpoint).
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

guided_curves <- map_dfr(guided_methods, function(m) {
  seeds <- sort(unique(na.omit(df_guided$checkpoint_seed[df_guided$method_id == m$id])))
  map_dfr(seeds, function(s) {
    cumulative_curve(df_guided, m$id, target_seed = s) %>% mutate(label = m$label, seed = s)
  })
})

# Seed 16 uses the `best` checkpoint for myopic-greedy only; every other
# (method, seed) combo uses `final` (myopic's other seeds have no `best`
# variant at all, and anticipatory's final checkpoints never diverged).
greedy_curves <- map_dfr(greedy_methods, function(m) {
  seeds <- sort(unique(na.omit(df_greedy$checkpoint_seed[df_greedy$method_id == m$id])))
  map_dfr(seeds, function(s) {
    variant <- if (m$id == "myopic_dqn_greedy" && s == 16) "best" else "final"
    cumulative_curve(df_greedy, m$id, target_seed = s, target_variant = variant) %>%
      mutate(label = m$label, seed = s)
  })
})

dqn_curves <- bind_rows(guided_curves, greedy_curves)

fd_colors <- setNames(map_chr(fd_methods, "color"), map_chr(fd_methods, "label"))
dqn_colors <- setNames(
  c(map_chr(guided_methods, "color"), map_chr(greedy_methods, "color")),
  c(map_chr(guided_methods, "label"), map_chr(greedy_methods, "label"))
)
all_colors <- c(fd_colors, dqn_colors)

# End-of-line labels replace the legend box.
last_labels <- bind_rows(
  fd_curves %>%
    group_by(label) %>%
    filter(task == max(task)) %>%
    transmute(label, task, y = cum_cost),
  dqn_curves %>%
    group_by(label, task) %>%
    summarise(y = median(cum_cost), .groups = "drop") %>%
    group_by(label) %>%
    filter(task == max(task))
)

p <- ggplot() +
  stat_lineribbon(
    data = dqn_curves, aes(x = task, y = cum_cost, fill = label, color = label),
    .width = c(.5, .8, .95), alpha = 0.32, linewidth = 1.1, lineend = "round"
  ) +
  geom_line(
    data = fd_curves, aes(x = task, y = cum_cost, color = label),
    linetype = "dashed", linewidth = 1, lineend = "round"
  ) +
  geom_text_repel(
    data = last_labels, aes(x = task, y = y, label = label, color = label),
    hjust = 0, nudge_x = 1.5, direction = "y", segment.size = 0.3,
    size = 3.6, fontface = "bold", show.legend = FALSE,
    max.overlaps = Inf, box.padding = 0.3
  ) +
  scale_color_manual(values = all_colors, guide = "none") +
  scale_fill_manual(values = dqn_colors, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.28))) +
  scale_y_continuous(labels = scales::label_comma()) +
  coord_cartesian(clip = "off") +
  labs(x = "Number of tasks", y = "Cumulative PDDL cost") +
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(15, 115, 10, 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.4),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40")
  )

ggsave(out_path, p, width = 10, height = 6.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
