# Figure 4 (Section 5) -- destination correctness and preservation are separate.
#
#   (a) two corrections of the same size, landing differently   Qwen3-8B, sealed
#   (b) collateral on both denominators, against the 5% bound   every setting with
#       an exposure record, protocol labelled per row
#
# The intervals that decide the licensing outcome are in Figure 1(b) and are not
# repeated here. Data: figures_final/data/{arms,arms_target_hit,preservation,
# constants}.csv. Run from the repo root:
#   Rscript figures_final/R/fig_chain.R

source("figures_final/R/theme_sakiko.R")

const <- fig_data("constants")
cval <- function(n) const$value[const$name == n]

# ---- (a) activation against the score-space comparator ----------------------
arms <- fig_data("arms")
th <- fig_data("arms_target_hit")
arms$metric <- ordered_by(arms, "metric", "metric_order")
ARM_COLOUR <- c(activation = REAL, `score-space` = COMPARE)
arm_label <- setNames(sprintf("%s   target-hit %.3f", th$arm, th$target_hit), th$arm)

wide <- reshape(arms[, c("metric", "arm", "count")], idvar = "metric",
                timevar = "arm", direction = "wide")
names(wide) <- c("metric", "activation", "score_space")

pa <- ggplot(arms, aes(x = count, y = metric)) +
  geom_segment(data = wide, aes(x = activation, xend = score_space, y = metric,
                                yend = metric), inherit.aes = FALSE,
               colour = GREY_L, linewidth = 1.4, lineend = "round") +
  geom_point(aes(colour = arm, shape = arm), size = 2.4) +
  geom_text(aes(label = count, colour = arm),
            hjust = ifelse(arms$count >= ave(arms$count, arms$metric, FUN = mean),
                           -0.55, 1.55),
            size = 2.4, fontface = "bold", show.legend = FALSE) +
  scale_colour_manual(values = ARM_COLOUR, labels = arm_label) +
  scale_shape_manual(values = c(activation = 18, `score-space` = 16),
                     labels = arm_label) +
  scale_x_continuous(limits = c(0, 48), breaks = seq(0, 40, 10)) +
  labs(x = "count, of 72 routed channel errors", y = NULL,
       title = "(a) Similar gain, different destination",
       subtitle = "Qwen3-8B · sealed · descriptive comparison") +
  theme_sakiko() +
  theme(legend.position = "bottom")

# ---- (b) collateral on both denominators ------------------------------------
pres <- fig_data("preservation")
pres$label <- sprintf("%s\n(%s)", pres$setting, pres$protocol)
pres$label <- factor(pres$label, levels = unique(pres$label[order(pres$setting_order)]))
pres$estimand <- factor(pres$estimand, levels = c("E1", "E2"))

wide_p <- reshape(pres[, c("label", "estimand", "rate")], idvar = "label",
                  timevar = "estimand", direction = "wide")
names(wide_p) <- c("label", "E1", "E2")
ann <- pres[pres$estimand == "E2", ]
ann$text <- sprintf("E2 %d of %d%s  ·  E1 %d of %d", ann$broken, ann$n,
                    ifelse(ann$vacuous == 1, " (vacuous)", ""),
                    pres$broken[pres$estimand == "E1"], pres$n[pres$estimand == "E1"])

pb <- ggplot(pres, aes(x = rate, y = label)) +
  geom_vline(xintercept = cval("collateral_bound_pct"), linetype = "dashed",
             colour = GREY30, linewidth = 0.35) +
  geom_segment(data = wide_p, aes(x = E1, xend = E2, y = label, yend = label),
               inherit.aes = FALSE, colour = GREY_L, linewidth = 1.2,
               lineend = "round") +
  geom_point(aes(shape = estimand, fill = estimand), size = 2.2, colour = ROSE,
             stroke = 0.5) +
  geom_text(data = ann, aes(x = 76, y = label, label = text), inherit.aes = FALSE,
            hjust = 1, vjust = -1.5, size = 2.1, colour = GREY30) +
  annotate("text", x = cval("collateral_bound_pct"), y = 0.55, hjust = -0.12,
           label = "bound 5%", size = 2.2, colour = GREY30) +
  scale_shape_manual(values = c(E1 = 21, E2 = 21)) +
  scale_fill_manual(values = c(E1 = "white", E2 = ROSE)) +
  scale_x_continuous(limits = c(-2, 78), breaks = c(0, 5, 20, 40, 60)) +
  labs(x = "correct decisions broken (%)", y = NULL,
       title = "(b) Preservation, on both denominators",
       subtitle = "E2 ≥ E1 by construction") +
  theme_sakiko() +
  theme(legend.position = "bottom")

save_fig(pa | pb, "destination_and_preservation", width = 5.5, height = 2.6)
