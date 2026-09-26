# Shared theme and palette for the SAKIKO figures.
#
# Tone follows the author's overview figure: a macaron palette — pale fills carrying
# the area, a deeper mark of the same hue carrying the point, line or border — on a
# white panel with a thin light frame and navy headings. Hues are color.amfe.space
# palette 106; the pale fills are tints of those hues, the marks are the hues
# themselves. Nothing is drawn at full saturation over a large area.
#
# Each role keeps one meaning in every figure:
#   blue    the real direction, the intervention, the activation arm
#   green   arrival at the required action, a property that holds, ADMIT
#   cyan    a comparison arm: score-space, ungated
#   peach   moved, but not to the required action
#   rose    a correct decision broken, a property that fails
#   yellow  a formal DECLINE: insufficient evidence, not a failure colour
#
# Every figure plots data exported by figures_final/export_figure_data.py, where the
# evidence index and the frozen verdict artifacts are read and asserted. Nothing here
# computes a number.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

# ---- palette 106 hues (marks) and their macaron tints (fills) ----------------
BLUE   <- "#7b95c6"; BLUE_F   <- "#ccd7ea"   # intervention
CYAN   <- "#49c2d9"; CYAN_F   <- "#c6e9f2"   # comparison arm
GREEN  <- "#67a583"; GREEN_F  <- "#d3e6d5"   # arrival, holds, ADMIT
LIME   <- "#a2c986"; LIME_F   <- "#e2eed6"
YELLOW <- "#e8cf6a"; YELLOW_F <- "#fbf1c7"   # DECLINE
PEACH  <- "#f59c7c"; PEACH_F  <- "#fde0d3"   # wrong destination
ROSE   <- "#d4646a"; ROSE_F   <- "#f6d8da"   # broken, fails
GREY   <- "#9aa3ad"; GREY_F   <- "#e8ecf1"   # controls, unchanged
GREY_L <- "#c3cad2"; GREY_XL  <- "#eef1f5"

NAVY   <- "#1f3864"    # headings, as in the overview figure
INK    <- "#22303f"; GREY30 <- "#5b6672"

# ---- roles ------------------------------------------------------------------
REAL    <- BLUE;   REAL_F    <- BLUE_F
ARRIVE  <- GREEN;  ARRIVE_F  <- GREEN_F
COMPARE <- CYAN;   COMPARE_F <- CYAN_F
OTHER   <- PEACH;  OTHER_F   <- PEACH_F
BROKEN  <- ROSE;   BROKEN_F  <- ROSE_F
DECLINE <- YELLOW; DECLINE_F <- YELLOW_F
NULLGREY <- GREY;  UNCHANGED <- GREY_L; UNEXPOSED <- GREY_XL

# darker inks, for small text on a pale fill
INK_REAL   <- "#41598f"
INK_ARRIVE <- "#357a57"
INK_COMPARE<- "#1f7f96"
INK_OTHER  <- "#b9603c"
INK_BROKEN <- "#a7434a"
INK_DECLINE<- "#8a7526"

# what each control arm corrupts -> its colour, cool to warm across the battery
COMPONENT_COLOURS <- c(real = REAL, direction = CYAN, sign = GREEN,
                       site = PEACH, channel = ROSE)
STATUS_FILL <- c(holds = GREEN_F, partial = BLUE_F, fails = ROSE_F,
                 `not evaluated` = "#f5f7f9")
STATUS_INK  <- c(holds = INK_ARRIVE, partial = INK_REAL, fails = INK_BROKEN,
                 `not evaluated` = "#9aa3ad")
VERDICT_FILL <- c(ADMIT = GREEN_F, DECLINE = YELLOW_F, `not sealed` = "#f5f7f9")
VERDICT_INK  <- c(ADMIT = INK_ARRIVE, DECLINE = INK_DECLINE)

SERIF <- "Times New Roman"

theme_sakiko <- function(base_size = 8.5, base_family = SERIF) {
  theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.background  = element_rect(fill = "white", colour = NA),
      panel.border      = element_rect(colour = "#d8dee8", linewidth = 0.4),
      panel.grid.major  = element_line(colour = "#eef1f5", linewidth = 0.3),
      panel.grid.minor  = element_line(colour = "#f6f8fa", linewidth = 0.25),
      axis.ticks        = element_line(colour = "#c3cad2", linewidth = 0.3),
      axis.text         = element_text(colour = GREY30, size = rel(0.85)),
      axis.title        = element_text(colour = INK),
      plot.title        = element_text(colour = NAVY, size = rel(1.05), hjust = 0.5,
                                       margin = margin(b = 3)),
      plot.subtitle     = element_text(colour = GREY30, size = rel(0.8), hjust = 0.5,
                                       margin = margin(b = 4)),
      plot.caption      = element_text(colour = GREY30, size = rel(0.78), hjust = 1,
                                       face = "italic"),
      strip.background  = element_rect(fill = "#eef1f5", colour = "#d8dee8",
                                       linewidth = 0.4),
      strip.text        = element_text(colour = NAVY, size = rel(0.85),
                                       margin = margin(2.5, 2.5, 2.5, 2.5)),
      legend.background = element_blank(),
      legend.key        = element_blank(),
      legend.title      = element_blank(),
      legend.text       = element_text(colour = GREY30, size = rel(0.8)),
      legend.margin     = margin(0, 0, 0, 0),
      legend.box.spacing = unit(3, "pt"),
      plot.margin       = margin(3, 4, 3, 3)
    )
}

# Read an exported table by name, e.g. fig_data("nulls").
fig_data <- function(name) {
  utils::read.csv(file.path("figures_final", "data", paste0(name, ".csv")),
                  check.names = FALSE, stringsAsFactors = FALSE)
}

# Order a column by its exported *_order companion, so the R scripts never
# re-decide an ordering the exporter already fixed.
ordered_by <- function(df, col, order_col) {
  # levels must be unique: a table may carry many rows per category
  factor(df[[col]], levels = unique(df[[col]][order(df[[order_col]])]))
}

save_fig <- function(plot, name, width, height) {
  dir.create("figures", showWarnings = FALSE)
  for (dev in list(list(ext = "pdf", d = grDevices::cairo_pdf),
                   list(ext = "png", d = NULL))) {
    args <- list(filename = file.path("figures", paste0(name, ".", dev$ext)),
                 plot = plot, width = width, height = height, units = "in")
    if (!is.null(dev$d)) args$device <- dev$d else args$dpi <- 300
    do.call(ggsave, args)
  }
  cat(sprintf("  wrote figures/%s.pdf / .png\n", name))
}
