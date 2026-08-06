# Two-panel learning curve (success rate | task return), aggregated across
# all 4 checkpoint seeds {0,4,8,16}, myopic vs anticipatory.
#
# Same ggdist::stat_lineribbon approach as the single-metric versions --
# each seed's binned mean curve is one "draw" at each x, turned into a
# median + nested quantile ribbons per panel (genuine between-seed variance).

library(tidyverse)
library(ggdist)
library(ggrepel)

out_path <- "results/canonical_planner/figures/training_seeds_combined_ggdist.png"

success <- read_csv("results/canonical_planner/figures/training_success_seeds_binned.csv", show_col_types = FALSE) %>%
  mutate(mean_value = mean_value * 100, metric = "Success rate (%)")
returns <- read_csv("results/canonical_planner/figures/training_reward_seeds_binned.csv", show_col_types = FALSE) %>%
  mutate(metric = "Task return")

df <- bind_rows(success, returns) %>%
  mutate(metric = factor(metric, levels = c("Success rate (%)", "Task return")))

colors <- c("Myopic DQN" = "#56B4E9", "Anticipatory DQN" = "#009E73")

last_labels <- df %>%
  group_by(metric, label, env_step) %>%
  summarise(y = median(mean_value), .groups = "drop") %>%
  group_by(metric, label) %>%
  filter(env_step == max(env_step))

p <- ggplot(df, aes(x = env_step, y = mean_value, fill = label, color = label)) +
  stat_lineribbon(.width = c(.5, .8, .95), alpha = 0.3, linewidth = 1.1) +
  geom_text_repel(
    data = last_labels, aes(x = env_step, y = y, label = label, color = label),
    hjust = 0, nudge_x = 1.5, direction = "y", segment.size = 0.3,
    size = 3.6, fontface = "bold", show.legend = FALSE
  ) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(values = colors, guide = "none") +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k"),
                      expand = expansion(mult = c(0.01, 0.28))) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Training step",
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(10, 10, 10, 10),
    panel.spacing.x = unit(2, "lines"),
    strip.text = element_text(face = "bold", size = 13, color = "gray20"),
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.4),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40")
  )

ggsave(out_path, p, width = 13, height = 5.5, dpi = 200)
cat("wrote", out_path, "\n")
