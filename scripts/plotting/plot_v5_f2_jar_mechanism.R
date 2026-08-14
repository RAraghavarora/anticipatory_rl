# F2 -- Jar mechanism over training.
#
# Fraction of actions that are refill_water, vs training step. Jar use scales
# with the credit horizon -- gamma = 0.98 climbs highest, 0.90 never picks the
# jar up at all -- while the myopic run decays away from the demonstrated
# starting behaviour.
#
# Series: gamma = 0.90/0.95/0.97/0.98 (seed 0), myopic.
#
# Lines are labelled at their right-hand end rather than through a legend. The
# gamma ramp is one hue by design (gamma is ordered), so adjacent steps are only
# ~6 dE apart -- too close to identify by colour alone. The direct label carries
# identity and the ramp is left to carry order.

library(ggrepel)
source("scripts/plotting/v5_curve_style.R")

RUNS <- c(
  "v5_ant_g0.90_s0", "v5_ant_g0.95_s0", "v5_ant_g0.97_s0", "v5_ant_g0.98_s0",
  "v5-myopic-g097-s8"
)

df <- load_metric("action_type_fraction__action_type_refill_water", "mean") %>%
  filter(run %in% RUNS) %>%
  mutate(series = factor(series, levels = names(SERIES_COLORS)))

ends <- df %>% group_by(series) %>% slice_max(step, n = 1, with_ties = FALSE) %>% ungroup()

p <- ggplot(df, aes(step, value, color = series)) +
  geom_line(linewidth = 0.8) +
  geom_text_repel(
    data = ends, aes(label = series),
    hjust = 0, direction = "y", nudge_x = 12000, segment.size = 0.25,
    segment.color = "gray70", size = 3.4, fontface = "bold",
    min.segment.length = 0, box.padding = 0.18, show.legend = FALSE
  ) +
  scale_color_manual(values = SERIES_COLORS, drop = TRUE) +
  scale_y_continuous(
    name = "% of actions that are refill_water",
    labels = scales::label_percent(accuracy = 0.5),
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  scale_x_continuous(
    name = "Training step",
    labels = function(x) paste0(x / 1000, "k"),
    breaks = seq(0, 500000, 100000),
    limits = c(0, 610000),
    expand = expansion(mult = c(0.01, 0))
  ) +
  base_theme +
  theme(legend.position = "none")

save_fig(p, "results/v5/figures/f2_jar_mechanism.pdf")
