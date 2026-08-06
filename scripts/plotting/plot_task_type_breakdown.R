# Per-task-type cost breakdown, myopic vs anticipatory guided planner,
# small multiples (one tiny panel per task_type) -- the RL-papers convention
# of many small panels that each show the same simple comparison at a glance,
# rather than one dense combined chart.
#
# Aggregation matches the headline-table methodology: mean cost per
# (method, seed, task_type) first (averaging over that task type's
# occurrences across the 10 canonical sequences), THEN mean +/- SEM across
# the 4 checkpoint seeds -- so error bars reflect between-seed variance, not
# within-task-type noise.

library(tidyverse)

out_path <- "results/canonical_planner/figures/task_type_breakdown.pdf"

methods <- tribble(
  ~method_id, ~label, ~color,
  "myopic_dqn_beta1", "Myopic RL (guided)", "#56B4E9",
  "anticipatory_dqn_beta1_25", "Anticipatory RL (guided)", "#009E73"
)

df <- read_csv("results/canonical_planner/planner/task_results.csv", show_col_types = FALSE) %>%
  semi_join(methods, by = "method_id")

per_seed <- df %>%
  group_by(method_id, checkpoint_seed, task_type) %>%
  summarise(mean_cost = mean(task_cost_pddl), .groups = "drop")

summary_df <- per_seed %>%
  group_by(method_id, task_type) %>%
  summarise(
    sem = sd(mean_cost) / sqrt(n()),
    mean_cost = mean(mean_cost),
    .groups = "drop"
  ) %>%
  inner_join(methods, by = "method_id") %>%
  mutate(label = factor(label, levels = methods$label))

colors <- setNames(methods$color, methods$label)

p <- ggplot(summary_df, aes(x = label, y = mean_cost, fill = label)) +
  geom_col(width = 0.65, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_cost - sem, ymax = mean_cost + sem), width = 0.3, linewidth = 0.6, color = "black") +
  facet_wrap(~task_type, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = colors, name = NULL) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(x = NULL, y = "Mean PDDL cost per task (± SEM across seeds)") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.text = element_text(face = "bold", size = 9, color = "gray20"),
    legend.position = "bottom",
    axis.title.y = element_text(color = "gray30", size = 9),
    axis.text.y = element_text(color = "gray40", size = 7.5),
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(10, 12, 10, 10)
  )

ggsave(out_path, p, width = 15, height = 3.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
