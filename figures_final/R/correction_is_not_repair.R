# Figure 1 (Section 3) -- correction is not repair.
#
#   (a) one intervention, every outcome resolved   Phi-3.5, historical, row-level
#       All FIVE classes of Table 1 are drawn. An earlier version omitted
#       SOURCE RETAINED (47), so the panel showed 246 of the 293 decisions the
#       Router fired on while its title claimed every outcome.
#
#       Layout: the two population labels are wrapped onto two lines and the
#       five-swatch legend sits at the left of (a) rather than under both
#       panels, which removes the bottom legend row entirely.
#   (b) target-hit with 95% bootstrap intervals    sealed, three settings
#
# Data: figures_final/data/{outcomes,forest,constants}.csv. Run from the repo root:
#   Rscript figures_final/R/fig1_thesis.R

source("figures_final/R/theme_sakiko.R")

const <- fig_data("constants")
cval <- function(n) const$value[const$name == n]

# ---- (a) where every touched decision ended up ------------------------------
out <- fig_data("outcomes")
out$population <- ordered_by(out, "population", "population_order")
out$outcome <- ordered_by(out, "outcome", "outcome_order")
OUTCOME_FILL   <- c(`source retained` = "#f7f9fb", arrive = ARRIVE_F, elsewhere = OTHER_F,
                    kept = GREY_F, broken = BROKEN_F)
OUTCOME_BORDER <- c(`source retained` = GREY, arrive = ARRIVE, elsewhere = OTHER,
                    kept = GREY_L, broken = BROKEN)
OUTCOME_INK    <- c(`source retained` = GREY30, arrive = INK_ARRIVE, elsewhere = INK_OTHER,
                    kept = GREY30, broken = INK_BROKEN)

totals <- unique(out[, c("population", "total")])

pa <- ggplot(out, aes(x = population, y = count, fill = outcome,
                      colour = outcome)) +
  geom_col(width = 0.52, linewidth = 0.45,
           position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = count, colour = outcome),
            position = position_stack(vjust = 0.5, reverse = TRUE),
            size = 2.4, fontface = "bold", show.legend = FALSE) +
  scale_colour_manual(values = OUTCOME_INK,
                      breaks = c("source retained", "arrive", "elsewhere", "kept", "broken"),
                      guide = "none") +
  geom_text(data = totals, aes(x = population, y = total, label = sprintf("of %d", total)),
            inherit.aes = FALSE, hjust = -0.18, size = 2.3, colour = GREY30) +
  scale_fill_manual(values = OUTCOME_FILL,
                    breaks = c("source retained", "arrive", "elsewhere", "kept", "broken")) +
  scale_y_continuous(limits = c(0, 248), breaks = seq(0, 200, 50)) +
  coord_flip() +
  labs(x = NULL, y = "decisions",
       title = "(a) One intervention, every outcome resolved",
       subtitle = sprintf("Phi-3.5 · historical · net gain +%d · target-hit %.2f",
                          cval("phi_net"), cval("phi_target_hit"))) +
  theme_sakiko() +
  theme(legend.position = "none")   # 圖例統一由 patchwork 收到左側

# ---- (b) what the evidence licenses -----------------------------------------
fo <- fig_data("forest")
fo$setting <- ordered_by(fo, "setting", "setting_order")
fo$label <- sprintf("%s\n%d source exits", fo$setting, fo$exits)
fo$label <- factor(fo$label, levels = fo$label[order(fo$setting_order)])

pb <- ggplot(fo, aes(x = point, y = label, fill = verdict)) +
  geom_vline(xintercept = cval("target_hit_criterion"), linetype = "dashed",
             colour = GREY30, linewidth = 0.35) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.14,
                colour = "#333333", linewidth = 0.4) +
  geom_point(shape = 21, size = 2.6, colour = "#333333", stroke = 0.4) +
  geom_text(aes(label = sprintf("%.3f", point)), vjust = -1.3, size = 2.4,
            colour = "#333333") +
  geom_text(aes(x = 1.09, label = verdict, colour = verdict), hjust = 1, size = 2.5,
            fontface = "bold", show.legend = FALSE) +
  annotate("text", x = cval("target_hit_criterion"), y = 0.62, hjust = -0.1,
           label = "criterion 0.50", size = 2.2, colour = GREY30) +
  scale_fill_manual(values = VERDICT_FILL, guide = "none") +
  scale_colour_manual(values = c(ADMIT = "#2f6b4a", DECLINE = "#4f3d70"),
                      guide = "none") +
  scale_x_continuous(limits = c(0.38, 1.10), breaks = seq(0.4, 1.0, 0.2)) +
  coord_cartesian(clip = "off") +
  labs(x = "target-hit, 95% bootstrap interval", y = NULL,
       title = "(b) What the evidence licenses",
       subtitle = "sealed · criteria frozen before evaluation") +
  theme_sakiko()

# 圖例收到最左側：橫向靠換行後的 y 標籤騰出的空間，垂直則省掉整條底列
pa_leg <- pa + theme(legend.position = "left",
                     legend.key.size = unit(8, "pt"),
                     legend.text = element_text(size = rel(0.72)),
                     legend.margin = margin(0, 2, 0, 0))

save_fig(pa_leg | pb, "correction_is_not_repair", width = 5.5, height = 1.95)
