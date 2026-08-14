# Cloud + box plot, 2-room and 3-room (v5) side by side -- an extension of
# plot_canonical_cost_raincloud.R (left untouched, still produces its own
# standalone 2-room figure).
#
# Side by side (patchwork "|"), not stacked rows: each row of a raincloud is
# tall (half-eye + box), and 2-room already has 8 methods -- stacking a
# second panel under it would make the combined figure very tall for little
# benefit. Two columns keep the height to whichever panel has more rows.
# No shared legend needed: as in the original, color = method and the
# y-axis category label already names the method, so there's nothing a
# legend would add.
#
# Each panel orders its own rows by its own median cost -- same as the
# original script's rationale (row order is a within-panel fact, not a fixed
# cross-domain ranking). The two panels are NOT meant to be read as one
# combined leaderboard: 2-room and 3-room happen to sit at comparable
# absolute cost scales, but the methods available differ -- v5's Myopic RL
# rows are a single checkpoint (seed 8 only, no 4-seed sweep yet, see
# build_v5_cost_raincloud.py), and v5-only arms like K2/K4 clairvoyant or
# the no-demos ablation aren't duplicated here.
#
# Data: results/v5/figures/cost_raincloud.csv, built by
# scripts/restaurant/build_v5_cost_raincloud.py. Clairvoyant Oracle there is
# a satisficing UPPER BOUND (168/500 windows hit the 600s cap) -- reported
# because it's the arm used everywhere else in this archive, not because
# its spread is a tight distributional claim.

library(tidyverse)
library(ggdist)
library(patchwork)

# Some method_id values map to the same (label, color) in two rows below --
# 2-room and v5 spell a couple of these differently ("anticipatory_dqn_beta1_25"
# vs "anticipatory_dqn_guided" for the same role; "myopic_dqn_beta1" vs plain
# "myopic_dqn"). Each panel's data only ever contains one spelling, so the
# join stays 1:1 within a panel; this just lets one lookup table serve both
# domains' own naming. myopic_dqn_greedy happens to use the identical id in
# both domains already, no alias needed.
methods <- tribble(
  ~method_id, ~label, ~color,
  "clairvoyant_k3_lama", "Clairvoyant Oracle", "#E69F00",
  "anticipatory_dqn_greedy", "Anticipatory RL (greedy)", "#7FCFB4",
  "anticipatory_dqn_beta1_25", "Anticipatory RL (guided)", "#009E73",
  "anticipatory_dqn_guided", "Anticipatory RL (guided)", "#009E73",
  "myopic_dqn_greedy", "Myopic RL (greedy)", "#A6D8F0",
  "myopic_dqn_beta1_25", "Myopic RL (guided)", "#56B4E9",  # 2-room, beta=1.25 (matched)
  "myopic_dqn", "Myopic RL (guided)", "#56B4E9",           # v5 (3-room)
  "gnn_counterfactual", "One-task GNN (augmented)", "#D55E00",
  "gnn_faithful", "One-task GNN", "#CC79A7",
  "myopic_fd_optimal", "Myopic Oracle", "#000000"
)
colors <- setNames(methods$color, methods$label)

# --- 2-room: identical computation to plot_canonical_cost_raincloud.R ---
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

greedy_2room <- greedy_all %>%
  semi_join(best_seeds, by = c("method_id", "checkpoint_seed", "checkpoint_variant")) %>%
  select(-checkpoint_variant)

gnn_2room <- read_csv("results/canonical_planner/gnn/seed_summary.csv", show_col_types = FALSE) %>%
  select(method_id, checkpoint_seed = seed, mean_cost_pddl)

df_2room <- bind_rows(guided, greedy_2room, gnn_2room)

# --- 3-room (v5): pre-aggregated by scripts/restaurant/build_v5_cost_raincloud.py ---
df_3room <- read_csv("results/v5/figures/cost_raincloud.csv", show_col_types = FALSE) %>%
  select(method_id, mean_cost_pddl)

raincloud_panel <- function(df_raw, tag) {
  df <- df_raw %>%
    inner_join(methods, by = "method_id")

  level_order <- df %>%
    group_by(label) %>%
    summarise(median_cost = median(mean_cost_pddl), .groups = "drop") %>%
    arrange(median_cost) %>%
    pull(label)

  df <- df %>% mutate(label = factor(label, levels = level_order))

  ggplot(df, aes(x = label, y = mean_cost_pddl, fill = label, color = label)) +
    stat_halfeye(
      adjust = 0.6, width = 0.6, .width = 0, justification = -0.2,
      point_colour = NA, alpha = 0.6
    ) +
    geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.7, linewidth = 0.5) +
    coord_flip() +
    scale_fill_manual(values = colors, guide = "none") +
    scale_color_manual(values = colors, guide = "none") +
    scale_y_continuous(labels = scales::label_comma()) +
    labs(x = NULL, y = paste0(tag, " · mean PDDL cost / 50-task sequence")) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.4),
      axis.text.y = element_text(face = "bold", size = 10.5, color = "gray20"),
      axis.title.x = element_text(color = "gray30", size = 10),
      axis.text.x = element_text(color = "gray40"),
      plot.margin = margin(10, 12, 10, 8)
    )
}

p_2room <- raincloud_panel(df_2room, "2-room")
p_3room <- raincloud_panel(df_3room, "3-room")

combined <- p_2room | p_3room

out_path <- "results/v5/figures/thesis/cost_raincloud_v5.pdf"
ggsave(out_path, combined, width = 17, height = 8.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
