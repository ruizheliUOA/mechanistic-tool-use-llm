# Appendix figure -- what the licence adds over simpler reporting rules.
#
#   (a) every evaluated unit against eight progressively stricter rules
#   (b) how many units each rule accepts
#
# The point is the last column: a rule based on aggregate gain alone accepts all
# nine units; the full frozen conjunction accepts one. R4 rises because it drops
# the specificity requirement, so the sequence is not monotone.
#
# Data: figures_final/data/licence_vs_rules.csv, exported from
# research_exploration/sakiko_standard_stress_test/LICENCE_VS_SIMPLE_RULES.csv.
# Run from the repository root:
#   Rscript figures_final/R/licence_vs_rules.R

source("figures_final/R/theme_sakiko.R")

d <- fig_data("licence_vs_rules")
d$unit <- ordered_by(d, "unit", "unit_order")
d$rule <- ordered_by(d, "rule", "rule_order")
d$decision <- factor(d$decision, levels = c("accept", "reject"))

DEC_FILL <- c(accept = GREEN_F, reject = ROSE_F)
DEC_INK  <- c(accept = INK_ARRIVE, reject = INK_BROKEN)

# ---- (a) the grid -----------------------------------------------------------
pa <- ggplot(d, aes(x = rule, y = unit, fill = decision)) +
  geom_tile(aes(colour = decision), linewidth = 0.4,
            width = 0.92, height = 0.82) +
  scale_fill_manual(values = DEC_FILL,
                    labels = c(accept = "accepted by this rule",
                               reject = "rejected by this rule")) +
  scale_colour_manual(values = DEC_INK, guide = "none") +
  scale_x_discrete(position = "top") +
  labs(x = NULL, y = NULL,
       title = "(a) Every unit against eight reporting rules",
       subtitle = "eight alternative rules; R4 is not nested — it drops specificity") +
  theme_sakiko() +
  theme(panel.grid = element_blank(),
        legend.position = "bottom",
        axis.ticks = element_blank(),
        axis.text.x = element_text(size = rel(0.72), lineheight = 0.9),
        axis.text.y = element_text(size = rel(0.78)))

# ---- (b) how many survive ---------------------------------------------------
n_unit <- length(levels(d$unit))
cnt <- aggregate(list(n = d$decision == "accept"),
                 by = list(rule = d$rule, rule_order = d$rule_order), FUN = sum)
cnt <- cnt[order(cnt$rule_order), ]
cnt$rule <- factor(cnt$rule, levels = levels(d$rule))
cnt$lab <- sprintf("%d", cnt$n)

pb <- ggplot(cnt, aes(x = rule, y = n, group = 1)) +
  geom_hline(yintercept = 1, linetype = "dashed",
             colour = GREY_L, linewidth = 0.35) +
  geom_line(colour = REAL, linewidth = 0.6) +
  geom_point(size = 2.4, colour = REAL, fill = REAL_F,
             shape = 21, stroke = 0.5) +
  geom_text(aes(label = lab), vjust = -1.15, size = 2.5, colour = INK_REAL) +
  annotate("text", x = 1, y = 1, hjust = 0, vjust = -0.9,
           label = "one licensed claim", size = 2.2, colour = GREY30) +
  scale_y_continuous(limits = c(0, 10.5), breaks = c(0, 3, 6, 9)) +
  labs(x = NULL, y = "units accepted",
       title = "(b) Acceptances fall from nine to one",
       subtitle = "R4 rises because it drops the specificity requirement") +
  theme_sakiko() +
  theme(axis.text.x = element_text(size = rel(0.72), lineheight = 0.9))

save_fig(pa / pb + patchwork::plot_layout(heights = c(1.45, 1)),
         "licence_vs_rules", width = 5.5, height = 4.6)
