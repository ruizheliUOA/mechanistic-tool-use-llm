# Figure 3 (Section 4) -- correction is direction-specific.
#
#   (a) the Phi-3.5 control battery: what each arm corrects and what it costs
#   (b) every one of the 59 matched random directions in each sealed setting,
#       against the real direction
#
# Data: figures_final/data/{phi_controls,nulls,nulls_summary,constants}.csv,
# exported by figures_final/export_figure_data.py. Run from the repository root:
#   Rscript figures_final/R/fig3_correction.R

source("figures_final/R/theme_sakiko.R")
suppressPackageStartupMessages(library(ggrepel))

# ---- (a) the control battery ------------------------------------------------
ctl <- fig_data("phi_controls")
ctl$arm <- ordered_by(ctl, "arm", "arm_order")

pa <- ggplot(ctl, aes(x = arm, y = net, colour = component)) +
  geom_hline(yintercept = 0, colour = "#9a9a9a", linewidth = 0.3) +
  geom_segment(aes(xend = arm, y = 0, yend = net), linewidth = 0.5) +
  geom_point(size = 2.4) +
  geom_text(aes(label = sprintf("%+d", net)),
            hjust = ifelse(ctl$net >= 0, -0.45, 1.45), size = 2.5,
            show.legend = FALSE) +
  scale_colour_manual(values = COMPONENT_COLOURS, guide = "none") +
  scale_y_continuous(limits = c(-30, 72), breaks = seq(-20, 60, 20)) +
  coord_flip() +
  labs(x = NULL, y = "net gain on the locked test",
       title = "(a) Corrupted variants do not reproduce it",
       subtitle = "Phi-3.5 · historical · locked test") +
  theme_sakiko()

# ---- (b) the matched-random nulls -------------------------------------------
nul <- fig_data("nulls")
sm  <- fig_data("nulls_summary")
nul$setting <- ordered_by(nul, "setting", "setting_order")
sm$setting  <- factor(sm$setting, levels = levels(nul$setting))
rand <- nul[nul$kind == "random", ]
real <- nul[nul$kind == "real", ]
set.seed(20260914)                      # jitter only; the values themselves are exact

pb <- ggplot(rand, aes(x = target_gain, y = setting)) +
  geom_vline(xintercept = 0, colour = "#9a9a9a", linewidth = 0.3,
             linetype = "dashed") +
  geom_jitter(height = 0.16, width = 0, size = 0.7, colour = GREY, alpha = 0.55) +
  geom_point(data = real, aes(x = target_gain, y = setting), shape = 23,
             size = 2.6, fill = REAL, colour = "white", stroke = 0.4) +
  geom_text(data = sm, aes(x = real, y = setting, label = sprintf("%.3f", real)),
            vjust = -1.5, size = 2.5, colour = "#4f3d70", fontface = "bold") +
  geom_text(data = sm, aes(x = real, y = setting,
                           label = sprintf("%d of %d reach it", ge_real, k)),
            hjust = -0.18, vjust = 0.5, size = 2.2, colour = GREY30) +
  scale_x_continuous(limits = c(-0.125, 0.52), breaks = seq(-0.1, 0.4, 0.1)) +
  labs(x = "target gain", y = NULL,
       title = "(b) No random direction reaches it",
       subtitle = "sealed · 59 random directions · p = 0.017") +
  theme_sakiko()

save_fig(pa | pb, "direction_specificity", width = 5.5, height = 2.2)
