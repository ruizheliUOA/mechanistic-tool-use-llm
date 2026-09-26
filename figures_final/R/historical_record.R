# Figure D1 (Appendix D) -- the historical record behind Sections 4 and 5.
#
#   (a) E1 collateral of every artifact-backed Phi-3.5 seed, against the 5% bound,
#       with each run's net gain
#   (b) the Qwen2.5-7B locked battery
#
# The Phi-3.5 control battery is Figure 3(a) and is not repeated here.
# Data: figures_final/data/{seeds,q25_battery,constants}.csv. From the repo root:
#   Rscript figures_final/R/figD1_historical.R

source("figures_final/R/theme_sakiko.R")

const <- fig_data("constants")
cval <- function(n) const$value[const$name == n]

# ---- (a) every realization exceeds the bound --------------------------------
sd <- fig_data("seeds")
sd$label <- factor(sprintf("seed %s", sd$seed),
                   levels = sprintf("seed %s", sd$seed[order(sd$seed_order)]))

pa <- ggplot(sd, aes(x = rate, y = label)) +
  geom_vline(xintercept = cval("collateral_bound_pct"), linetype = "dashed",
             colour = GREY30, linewidth = 0.35) +
  geom_segment(aes(x = 0, xend = rate, y = label, yend = label), colour = ROSE_F,
               linewidth = 1.3) +
  geom_point(size = 2.2, colour = ROSE) +
  geom_text(aes(label = sprintf("%d broken", broken)), hjust = -0.25, size = 2.2,
            colour = "#8f3f44") +
  geom_text(aes(x = 40, label = sprintf("net +%d", net)), hjust = 1, size = 2.2,
            colour = "#2f6b4a") +
  annotate("text", x = cval("collateral_bound_pct"), y = 5.45, hjust = -0.12,
           label = "bound 5%", size = 2.2, colour = GREY30) +
  scale_x_continuous(limits = c(0, 41), breaks = c(0, 5, 10, 20, 30, 40)) +
  coord_cartesian(clip = "off") +
  labs(x = sprintf("E1 collateral: broken of %d baseline-correct (%%)", sd$n[1]),
       y = NULL, title = "(a) Every realization exceeds the bound",
       subtitle = "Phi-3.5 · five seeds · historical") +
  theme_sakiko()

# ---- (b) the Qwen2.5-7B locked battery --------------------------------------
bat <- fig_data("q25_battery")
bat$arm <- ordered_by(bat, "arm", "arm_order")
BATTERY_COLOUR <- c(real = REAL, control = GREY)

pb <- ggplot(bat, aes(x = net, y = arm, colour = kind)) +
  geom_segment(aes(x = 0, xend = net, y = arm, yend = arm), linewidth = 0.5) +
  geom_point(aes(shape = kind), size = 2.4) +
  geom_text(aes(label = sprintf("+%g", net)), vjust = -1.2, size = 2.3,
            show.legend = FALSE) +
  geom_point(data = bat[bat$annotation != "", ], aes(x = 50, y = arm), shape = 124,
             size = 2.6, colour = GREY, show.legend = FALSE) +
  geom_text(data = bat[bat$annotation != "", ],
            aes(x = 50, label = annotation), hjust = -0.10, vjust = 1.9,
            size = 2.1, colour = GREY30, show.legend = FALSE) +
  scale_colour_manual(values = BATTERY_COLOUR, guide = "none") +
  scale_shape_manual(values = c(real = 18, control = 16), guide = "none") +
  scale_x_continuous(limits = c(0, 96), breaks = seq(0, 80, 20)) +
  labs(x = "net gain", y = NULL, title = "(b) Real above every control",
       subtitle = "Qwen2.5-7B · locked battery · aggregate records") +
  theme_sakiko()

save_fig(pa | pb, "historical_record", width = 5.5, height = 2.3)
