# Figure 5 (Section 5) -- Correctable is not Licensable.
#
# One panel: the six properties and the frozen verdict for every evaluated setting,
# sealed settings above historical ones. Fill carries the status, the label carries
# the evidence behind it. Intervals are in Figure 1(b) and are not repeated here.
#
# Data: figures_final/data/ladder.csv. Run from the repo root:
#   Rscript figures_final/R/evidence_ladder.R

source("figures_final/R/theme_sakiko.R")

lad <- fig_data("ladder")
lad$property <- ordered_by(lad, "property", "property_order")
lad$setting <- ordered_by(lad, "setting", "setting_order")
lad$protocol <- factor(lad$protocol, levels = c("sealed", "historical"))
lad$status <- factor(lad$status,
                     levels = c("holds", "partial", "fails", "not evaluated"))
lad$bold <- ifelse(lad$property == "Licensable" & lad$label %in% c("ADMIT", "DECLINE"),
                   "bold", "plain")

p <- ggplot(lad, aes(x = property, y = setting, fill = status)) +
  geom_tile(aes(colour = status), linewidth = 0.4, width = 0.9, height = 0.74,
            show.legend = FALSE) +
  geom_text(aes(label = label, colour = status, fontface = bold), size = 2.3,
            show.legend = FALSE) +
  facet_grid(rows = vars(protocol), scales = "free_y", space = "free_y",
             switch = "y") +
  scale_fill_manual(values = STATUS_FILL, drop = FALSE) +
  scale_colour_manual(values = STATUS_INK, drop = FALSE, guide = "none") +
  scale_x_discrete(position = "bottom") +
  labs(x = NULL, y = NULL, title = "Where each evaluated setting stops",
       subtitle = paste("Steerable: random directions reaching the real effect (K differs by setting)  ·",
                        "Correctable: target-hit  ·  Preserving: broken of all baseline-correct (E1)")) +
  theme_sakiko() +
  theme(panel.grid = element_blank(),
        legend.position = "bottom",
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90),
        axis.ticks = element_blank(),
        plot.subtitle = element_text(size = rel(0.72)))

save_fig(p, "evidence_ladder", width = 5.5, height = 2.6)
