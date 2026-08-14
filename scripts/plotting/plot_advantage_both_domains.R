# Advantage over Myopic Oracle (%), 2-room vs 3-room, regrouped: one row
# per agent/method, two bars side by side (2-room, 3-room) -- not the
# flat 16-row ranking this replaces.
#
# The sorted single-list version buried the actual headline: it optimizes
# for "what's the best method overall" (a cross-domain leaderboard), but
# the finding that matters here is "which methods' advantage TRANSFERS
# across domains and which don't" -- a per-method, paired comparison. That
# needs method identity fixed as the row and domain as the two bars in it,
# so a tall-then-tiny pair (GNN augmented: +29.5% -> +3.1%, "collapses") is
# visually distinct at a glance from a tall-then-tall pair (Anticipatory RL
# guided: +31.8% -> +26.1%, "holds").
#
# Greedy rows (Myopic RL / Anticipatory RL, both domains) are dropped --
# 3-room's greedy arms are still early/partial seed sweeps with outlier
# issues of their own (see the raincloud figure's dropped point), and this
# chart's question is about guided/oracle/GNN methods anyway.
#
# Sorted by 2-room advantage descending (not by cross-domain delta anymore)
# -- once greedy is out of the picture, this ordering puts Anticipatory RL
# (guided) and One-task GNN (augmented) directly adjacent, which is exactly
# the pair whose 3-room values matter most to compare (26.1% and 3.1%
# respectively -- both still visible via the bars themselves, just not the
# sort key). K=4 Optimal (v5-only, no 2-room value, so no sort key) is
# dropped rather than appended below with an arbitrary position.
#
# Domain gets its own 2-color categorical encoding now (Okabe-Ito blue /
# vermillion) instead of the previous version's alpha-tint-of-method-hue --
# that was flagged as too subtle (the text already said the domain; alpha
# barely added a second cue). Color now carries the one distinction that
# actually matters for this chart's question.
#
# Data: results/v5/figures/advantage_both_domains.csv, built by
# scripts/restaurant/build_advantage_both_domains.py (unchanged -- this is
# a re-plot of the same numbers, not a new computation).

library(tidyverse)

raw <- read_csv("results/v5/figures/advantage_both_domains.csv", show_col_types = FALSE) %>%
  filter(!label %in% c("Myopic RL (greedy)", "Anticipatory RL (greedy)", "K=4 Clairvoyant Oracle"))

wide <- raw %>%
  select(domain, label, mean_adv) %>%
  pivot_wider(names_from = domain, values_from = mean_adv)

has_2room <- wide %>%
  filter(!is.na(`2-room`)) %>%
  arrange(desc(`2-room`)) %>%
  pull(label)

no_2room <- wide %>%
  filter(is.na(`2-room`)) %>%
  arrange(label) %>%
  pull(label)

method_order <- rev(c(has_2room, no_2room))  # rev(): first level plots at bottom

df <- raw %>%
  mutate(
    has_range = n > 1,
    label = factor(label, levels = method_order),
    domain = factor(domain, levels = c("2-room", "3-room")),
    bar_label = sprintf("%+.1f%%", mean_adv)
  )

colors <- c("2-room" = "#0072B2", "3-room" = "#D55E00")
out_path <- "results/v5/figures/thesis/advantage_both_domains.pdf"

p <- ggplot(df, aes(x = mean_adv, y = label, fill = domain, group = domain)) +
  geom_vline(xintercept = 0, color = "gray70", linewidth = 0.4) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_errorbarh(
    data = df %>% filter(has_range),
    aes(xmin = min_adv, xmax = max_adv),
    position = position_dodge(width = 0.75), height = 0.2, linewidth = 0.5, color = "gray30"
  ) +
  geom_text(
    aes(label = bar_label,
        x = if_else(mean_adv >= 0, pmax(mean_adv, if_else(has_range, max_adv, mean_adv)),
                    pmin(mean_adv, if_else(has_range, min_adv, mean_adv))),
        hjust = if_else(mean_adv >= 0, -0.15, 1.15), color = domain),
    position = position_dodge(width = 0.75), size = 3.1, fontface = "bold", show.legend = FALSE
  ) +
  scale_fill_manual(values = colors, name = NULL) +
  scale_color_manual(values = colors, guide = "none") +
  # scale_y_discrete(limits=...) pinned explicitly: the errorbar layer above
  # passes a filtered subset (has_range only), and ggplot2's discrete
  # y-scale training does not reliably preserve the full factor's levels()
  # order when layers see different subsets of it -- this exact bug hit the
  # previous version of this chart (confirmed via ggplot_build() there).
  scale_y_discrete(limits = method_order) +
  scale_x_continuous(labels = scales::label_percent(scale = 1), expand = expansion(mult = c(0.14, 0.14))) +
  labs(x = "Advantage over Myopic Oracle: (myopic cost − method cost) / myopic cost", y = NULL) +
  theme_minimal(base_size = 12.5) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.3),
    axis.text.y = element_text(face = "bold", size = 10.5, color = "gray20"),
    axis.title.x = element_text(color = "gray30", size = 10.5),
    axis.text.x = element_text(color = "gray40"),
    plot.margin = margin(10, 20, 8, 8)
  )

ggsave(out_path, p, width = 9, height = 6.8, device = cairo_pdf)
cat("wrote", out_path, "\n")
