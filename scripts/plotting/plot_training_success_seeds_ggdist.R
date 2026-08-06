# Learning curve (success rate vs. training step) aggregated across all 4
# checkpoint seeds {0,4,8,16}, myopic vs anticipatory.
#
# Each seed's own binned mean-success-rate curve is one "draw" at each x
# position; ggdist::stat_lineribbon turns the 4 draws per bin into a median
# line + nested quantile ribbons (.5/.8/.95) -- genuine between-seed
# variance, the RL-literature-standard band, not within-run task noise.

library(tidyverse)
library(ggdist)
library(ggrepel)

csv_path <- "results/canonical_planner/figures/training_success_seeds_binned.csv"
out_path <- "results/canonical_planner/figures/training_success_seeds_ggdist.png"

df <- read_csv(csv_path, show_col_types = FALSE)

colors <- c("Myopic DQN" = "#56B4E9", "Anticipatory DQN" = "#009E73")

last_labels <- df %>%
  group_by(label, env_step) %>%
  summarise(y = median(mean_value), .groups = "drop") %>%
  group_by(label) %>%
  filter(env_step == max(env_step))

p <- ggplot(df, aes(x = env_step, y = mean_value, fill = label, color = label)) +
  stat_lineribbon(.width = c(.5, .8, .95), alpha = 0.3, linewidth = 1.1) +
  geom_text_repel(
    data = last_labels, aes(x = env_step, y = y, label = label, color = label),
    hjust = 0, nudge_x = 1.5, direction = "y", segment.size = 0.3,
    size = 4, fontface = "bold", show.legend = FALSE
  ) +
  scale_color_manual(values = colors, guide = "none") +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k"),
                      expand = expansion(mult = c(0.01, 0.22))) +
  scale_y_continuous(labels = scales::label_percent()) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Training step",
    y = "Success rate",
    title = "Success-rate learning curves aggregated across seeds",
    subtitle = "Myopic vs anticipatory DQN · bands are 50/80/95% seed-quantile ribbons (n=4 seeds)",
    caption = "seeds {0,4,8,16}; band reflects between-seed variance, not within-run task noise"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40", size = 11, margin = margin(b = 12)),
    plot.caption = element_text(color = "gray55", size = 8, face = "italic"),
    plot.title.position = "plot",
    plot.margin = margin(10, 95, 10, 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.4),
    axis.title = element_text(color = "gray30"),
    axis.text = element_text(color = "gray40")
  )

ggsave(out_path, p, width = 9, height = 5.5, dpi = 200)
cat("wrote", out_path, "\n")
