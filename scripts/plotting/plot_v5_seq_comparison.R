# Sequence-by-sequence cost delta: Anticipatory RL (guided) minus a
# clairvoyant oracle (K=3 by default, `Rscript ... 4` for K=4), v5, one
# diverging bar per canonical sequence -- a delta chart, not the
# 2-point-per-sequence dumbbell or slopegraph this replaces.
#
# Both earlier versions made the reader compute the gap themselves (read
# two points/positions, subtract mentally, x10) and the slopegraph's
# crossing lines got tangled regardless of how much else was decluttered.
# "Sequence-by-sequence comparison" is fundamentally about the gap, and the
# absolute costs already live in the raincloud figure and RESULTS.md's
# table -- so this plots the gap directly: one mark per sequence, no lines
# to cross.
#
# Diverging color, not categorical: positive (right, amber) = the oracle
# costs less than us; negative (left, teal) = we cost less. Same hues as
# every other figure's Anticipatory RL / Clairvoyant Oracle colors, so
# "which color wins" reads the same way here as everywhere else in this
# archive, not a new legend to learn.
#
# Rows stay in sequence-ID order (00-09), not sorted by delta size --
# sequence identity carries meaning (00/02 are the jar-delivered control
# sequences).
#
# A dagger flags where the oracle hit its 600s satisficing cap on >=40% of
# windows -- those amber bars are a weaker upper bound than their
# neighbors, not a genuine loss of the same size. K=4 hits this cap far
# more (266/500 windows overall vs K=3's 168/500), so expect most of a K=4
# run's bars to carry the dagger -- that's a real property of K=4, not a
# plotting artifact.
#
# Data: results/v5/figures/seq_comparison.csv (K=3) or
# seq_comparison_k4.csv (K=4), built by
# scripts/restaurant/build_v5_seq_comparison.py --k {3,4}.

library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
K <- if (length(args) >= 1) as.integer(args[1]) else 3
stopifnot(K %in% c(3, 4))

oracle_label <- paste0("K=", K, " Clairvoyant Oracle")
in_csv <- if (K == 3) "results/v5/figures/seq_comparison.csv" else sprintf("results/v5/figures/seq_comparison_k%d.csv", K)
out_path <- if (K == 3) "results/v5/figures/seq_comparison.pdf" else sprintf("results/v5/figures/seq_comparison_k%d.pdf", K)

raw <- read_csv(in_csv, show_col_types = FALSE)

# cap_rate only exists on the oracle rows -- carrying it through
# pivot_wider alongside mean_cost (without declaring it an id_col) would
# make pivot_wider treat (sequence_id, cap_rate) as the row key, silently
# doubling every row instead of collapsing to one row per sequence. Pivot
# on mean_cost alone, then join cap_rate back in from the oracle-only subset.
wide <- raw %>%
  select(sequence_id, label, mean_cost) %>%
  pivot_wider(names_from = label, values_from = mean_cost)

cap <- raw %>%
  filter(label == oracle_label) %>%
  select(sequence_id, cap_rate)

df <- wide %>%
  left_join(cap, by = "sequence_id") %>%
  mutate(
    seq_num = str_replace(sequence_id, "iid-eval-seq-", ""),
    high_cap = !is.na(cap_rate) & cap_rate >= 0.4,
    seq_label = if_else(high_cap, paste0("seq ", seq_num, " †"), paste0("seq ", seq_num)),
    seq_label = factor(seq_label, levels = rev(seq_label[order(seq_num)])),
    delta = `Anticipatory RL (guided)` - .data[[oracle_label]],
    winner = if_else(delta > 0, paste0(oracle_label, " costs less"), "Anticipatory RL costs less")
  )

colors <- setNames(c("#E69F00", "#009E73"), c(paste0(oracle_label, " costs less"), "Anticipatory RL costs less"))

p <- ggplot(df, aes(x = delta, y = seq_label, fill = winner)) +
  geom_col(width = 0.6) +
  geom_vline(xintercept = 0, color = "gray40", linewidth = 0.5) +
  geom_text(
    aes(label = scales::label_comma(style_positive = "plus")(delta),
        hjust = if_else(delta > 0, -0.15, 1.15)),
    size = 3.2, fontface = "bold", color = "gray20"
  ) +
  scale_fill_manual(values = colors, name = NULL) +
  scale_x_continuous(labels = scales::label_comma(), expand = expansion(mult = 0.18)) +
  labs(
    x = paste0("Anticipatory RL (guided) − ", oracle_label, ", total cost per 50-task sequence"), y = NULL,
    caption = paste0("† ", oracle_label, " hit its 600s satisficing cap on >=40% of this sequence's windows.")
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.3),
    axis.text.y = element_text(face = "bold", size = 11, color = "gray20"),
    axis.title.x = element_text(color = "gray30", size = 10),
    axis.text.x = element_text(color = "gray40"),
    plot.margin = margin(10, 16, 8, 8),
    plot.caption = element_text(color = "gray40", size = 8.5, hjust = 0)
  )

ggsave(out_path, p, width = 8, height = 6.5, device = cairo_pdf)
cat("wrote", out_path, "\n")
