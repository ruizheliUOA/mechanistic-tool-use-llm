# Case-study figure -- one channel through the protocol.
#
# Qwen3-8B, cannot_answer -> tool_call, sealed protocol. Each row is one
# population, drawn to equal width so that the *composition* is comparable even
# where the absolute sizes differ by two orders of magnitude (211 against 6).
# Counts are printed inside each segment; the row label carries the total.
#
# Data: figures_final/data/casestudy_flow.csv. From the repository root:
#   Rscript figures_final/R/casestudy_flow.R

source("figures_final/R/theme_sakiko.R")

d <- fig_data("casestudy_flow")

# four semantic classes, so the legend stays readable
cls <- c(reached = "carried to the next stage", exit = "carried to the next stage",
         exposed = "carried to the next stage",
         unreached = "dropped out", retained = "dropped out",
         unexposed = "dropped out",
         gold = "arrived / preserved", kept = "arrived / preserved",
         other = "wrong destination / broken", broken = "wrong destination / broken")
d$class <- factor(cls[d$role],
                  levels = c("carried to the next stage", "dropped out",
                             "arrived / preserved", "wrong destination / broken"))

CLS <- c("carried to the next stage" = REAL_F, "dropped out" = GREY_XL,
         "arrived / preserved" = ARRIVE_F, "wrong destination / broken" = OTHER_F)
CLS_INK <- c("carried to the next stage" = INK_REAL, "dropped out" = GREY30,
             "arrived / preserved" = INK_ARRIVE, "wrong destination / broken" = INK_OTHER)

# equal-width rows: share within stage
tot <- stats::aggregate(count ~ population + stage, d, sum)
names(tot)[3] <- "stage_total"
d <- merge(d, tot, by = c("population", "stage"))
d$share <- d$count / d$stage_total
d <- d[order(d$population, d$stage_order, d$seg_order), ]
d$stage <- factor(d$stage, levels = rev(unique(d$stage[order(d$population, d$stage_order)])))

panel <- function(pop, title, sub) {
  x <- d[d$population == pop, ]
  x$stage <- droplevels(x$stage)
  ggplot(x, aes(x = share, y = stage, fill = class)) +
    geom_col(aes(colour = class), width = 0.58, linewidth = 0.35) +
    geom_text(aes(label = count, colour = class),
              position = position_stack(vjust = 0.5), size = 2.5,
              fontface = "bold", show.legend = FALSE) +
    scale_fill_manual(values = CLS, drop = FALSE) +
    scale_colour_manual(values = CLS_INK, drop = FALSE, guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.01))) +
    labs(x = NULL, y = NULL, title = title, subtitle = sub) +
    theme_sakiko() +
    theme(panel.grid = element_blank(),
          axis.text.x = element_blank(), axis.ticks = element_blank(),
          axis.text.y = element_text(size = rel(0.82), colour = INK),
          legend.position = "bottom")
}

pa <- panel("errors",  "(a) The error population, stage by stage",
            "rows drawn to equal width; the label carries the total")
pb <- panel("correct", "(b) The correct population",
            "the preservation bound that follows rests on 6 decisions")

save_fig(pa / pb + patchwork::plot_layout(heights = c(1.5, 1.15), guides = "collect") &
           theme(legend.position = "bottom"),
         "casestudy_flow", width = 5.5, height = 3.6)
