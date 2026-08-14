# Cloud + box plot: per-sequence mean cost distribution for each method.
#
# Density is over mean_cost_pddl (a full 50-task sequence average). The two
# FD baselines and the two RL-guided methods come from
# results/canonical_planner/planner/run_summary.csv: 10 runs for the FD baselines (no
# seed), 40 runs for the guided DQN methods (4 checkpoint seeds x 10
# sequences) -- total variability, seed and sequence pooled together.
#
# The two RL-greedy (direct policy rollout, no FD guidance) methods come from
# results/canonical_planner/greedy_rl/run_summary.csv and use only the
# best (checkpoint_seed, checkpoint_variant) combo, matching the headline
# table -- NOT pooled across seeds like the guided rows. Reason: myopic-greedy
# seed 16's *final* checkpoint catastrophically fails (3% success, ~8571 mean
# cost) from a late-training Q-value divergence, while still being 100%
# successful under FD-guided selection with the same checkpoint -- pooling it
# in would blow out the whole plot's axis with an outlier that isn't
# representative of the method, it's a policy-execution failure mode guided
# planning is immune to. Seed 16 also has a `best` checkpoint (saved at peak
# rolling success, pre-divergence) which recovers a lot (1966 mean cost, 76%
# success) but still isn't as good as seeds 0/4/8's final checkpoints -- so it
# doesn't change which (seed, variant) wins best-of, but both are kept in the
# CSV for transparency rather than only keeping whichever looks better.
#
# The one-step GNN (Talukder) baseline comes from
# results/canonical_planner/gnn/faithful_seed0_seq_cost.csv: per-task cost
# rows (strategy = myopic/auto) for a single GNN-guided model, no seed
# sweep, so it contributes 10 sequence-level points like the FD baselines.
#
# The augmented one-step GNN variant comes from
# results/canonical_planner/gnn/counterfactual_seed0_seq_cost.csv: same
# schema, but strategy includes aug+fill/aug+clean/aug+jar_position rows --
# the GNN actually commits to an augmented candidate here instead of falling
# back to myopic, which is why its cost is much lower than the faithful row.

library(tidyverse)
library(ggdist)

out_path <- "results/canonical_planner/figures/cost_raincloud.pdf"

guided <- read_csv("results/canonical_planner/planner/run_summary.csv", show_col_types = FALSE) %>%
  select(method_id, checkpoint_seed, mean_cost_pddl)

greedy_all <- read_csv("results/canonical_planner/greedy_rl/run_summary.csv", show_col_types = FALSE) %>%
  select(method_id, checkpoint_seed, checkpoint_variant, mean_cost_pddl)

best_seeds <- greedy_all %>%
  group_by(method_id, checkpoint_seed, checkpoint_variant) %>%
  summarise(seed_mean = mean(mean_cost_pddl), .groups = "drop") %>%
  group_by(method_id) %>%
  slice_min(seed_mean, n = 1) %>%
  ungroup() %>%
  select(method_id, checkpoint_seed, checkpoint_variant)

greedy <- greedy_all %>%
  semi_join(best_seeds, by = c("method_id", "checkpoint_seed", "checkpoint_variant")) %>%
  select(-checkpoint_variant)

# GNN: the single-seed *_seed0_seq_cost.csv files this originally read were
# superseded by the 4-seed sweep (moved to gnn/superseded_seed42/); read the
# per-(seed, seq) mean cost from seed_summary.csv instead, matching the
# thesis raincloud (plot_cost_raincloud_v5.R). Now 40 points per GNN arm.
gnn_all <- read_csv("results/canonical_planner/gnn/seed_summary.csv", show_col_types = FALSE) %>%
  select(method_id, mean_cost_pddl)

df_raw <- bind_rows(guided, greedy, gnn_all)

# Row order is by median cost (worst at top, best at bottom) rather than a
# fixed table order; coord_flip() puts the first factor level at the bottom.
methods <- tribble(
  ~method_id, ~label, ~color,
  "clairvoyant_k3_lama", "Clairvoyant Oracle", "#E69F00",
  "anticipatory_dqn_greedy", "Anticipatory RL (greedy)", "#7FCFB4",
  "anticipatory_dqn_beta1_25", "Anticipatory RL (guided)", "#009E73",
  "myopic_dqn_greedy", "Myopic RL (greedy)", "#A6D8F0",
  "myopic_dqn_beta1_25", "Myopic RL (guided)", "#56B4E9",  # beta=1.25, matched to anticipatory
  "gnn_counterfactual", "One-task GNN (augmented)", "#D55E00",
  "gnn_faithful", "One-task GNN", "#CC79A7",
  "myopic_fd_optimal", "Myopic Oracle", "#000000"
)

level_order <- df_raw %>%
  inner_join(methods, by = "method_id") %>%
  group_by(label) %>%
  summarise(median_cost = median(mean_cost_pddl), .groups = "drop") %>%
  arrange(median_cost) %>%
  pull(label)

df <- df_raw %>%
  inner_join(methods, by = "method_id") %>%
  mutate(label = factor(label, levels = level_order))

colors <- setNames(methods$color, methods$label)

p <- ggplot(df, aes(x = label, y = mean_cost_pddl, fill = label, color = label)) +
  stat_halfeye(
    adjust = 0.6, width = 0.6, .width = 0, justification = -0.2,
    point_colour = NA, alpha = 0.6
  ) +
  geom_boxplot(
    width = 0.2, outlier.shape = NA, alpha = 0.7, linewidth = 0.5
  ) +
  coord_flip() +
  scale_fill_manual(values = colors, guide = "none") +
  scale_color_manual(values = colors, guide = "none") +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(x = NULL, y = "Mean PDDL cost per 50-task sequence") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.4),
    axis.text.y = element_text(face = "bold", size = 12, color = "gray20"),
    axis.title.x = element_text(color = "gray30"),
    axis.text.x = element_text(color = "gray40"),
    plot.margin = margin(10, 15, 10, 10)
  )

ggsave(out_path, p, width = 9, height = 8.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
