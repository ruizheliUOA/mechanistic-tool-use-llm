# Case-study figure -- one transition, three verdicts.
#
# Three input variants of one When2Call question. Each panel shows the four
# action modes; the curved arrow on the right of each panel is the observed
# move, tool_call -> request_for_info, and it is IDENTICAL in all three panels
# because all three are drawn by one call with the same relative offsets. Only
# the heavy outline -- the required action -- moves. The three bands below say
# what each readout can distinguish, by how they are segmented: the transition
# band is one piece (separates nothing), the accuracy band leaves B empty
# (separates two of three), the destination band is three pieces.
#
# Drawn in grid rather than ggplot: this is a diagram, not a plot, and grid has
# roundrectGrob and exact inch coordinates. Font and device match the other
# figures -- Times New Roman through cairo_pdf, which embeds.
#
# No in-figure legend: the caption states what the arrow and the heavy outline
# mean, and the page budget is tight.
#
# Colour appears ONLY in the destination band, in Okabe-Ito hues, as a light
# tint with same-hue dark text; every band also states its class in words, so
# nothing depends on colour. Everything else is greyscale.
#
# COUNTS ARE OUTCOME-CLASS TOTALS, from final_freeze/GLOBAL_INTEGRITY_RECHECK.md
# (GOLD=107, OTHER=46, BROKEN=52 of 93 exposed). They are not counts of this
# particular transition, which the artifacts do not resolve.
#
# Run from the repository root:
#   Rscript figures_final/R/casestudy_transition.R

suppressPackageStartupMessages(library(grid))

W_IN <- 5.5; H_IN <- 1.91
SERIF <- "Times New Roman"; MONO <- "Courier New"

# ---- greyscale roles --------------------------------------------------------
BOX_EDGE  <- "#C3CAD2"   # an option not required here
REQ_EDGE  <- "#22303F"   # the required action
LAB_GREY  <- "#5B6672"
LAB_INK   <- "#22303F"
ARROW_COL <- "#5B6672"
BAND_FILL <- "#EEF1F5"
DASH_EDGE <- "#9AA3AD"

# ---- Okabe-Ito, light tint + same-hue dark text; only in the destination band
DEST <- list(
  list(fill = "#D7EFE7", ink = "#00664A", text = "gold arrival · 107"),
  list(fill = "#FAEBD1", ink = "#8A6000", text = "other wrong · 46"),
  list(fill = "#F9DFD2", ink = "#8F3F00", text = "broken · 52 of 93"))

MODES <- c("tool_call", "request_for_info", "cannot_answer", "direct_answer")
# required action per panel: A -> request_for_info, B -> cannot_answer, C -> tool_call
REQ   <- c(2L, 3L, 1L)
TITLE <- c("parameter dropped", "related, unanswerable", "unchanged")

# ---- layout, measured from the top ------------------------------------------
PX   <- c(0.66, 2.32, 3.98); PW <- 1.44
BX_W <- 1.14; BX_H <- 0.20; BX_GAP <- 0.04
T_TITLE <- 0.135
T_BOX0  <- 0.22
T_BAND  <- c(1.345, 1.555, 1.765); BAND_H <- 0.17
LAB_X   <- 0.60

yy <- function(t) unit(H_IN - t, "in")          # top-down -> device
# measure a string as it will actually be drawn
w_of <- function(s, fam, size, face = "plain")
  convertWidth(grobWidth(textGrob(s, gp = gpar(fontfamily = fam, fontsize = size,
                                               fontface = face))), "in", valueOnly = TRUE)
# lay a run of (string, family, size, face) pieces out left to right, centred on cx
run <- function(pieces, cx, ty, col) {
  ws <- vapply(pieces, function(q) w_of(q[[1]], q[[2]], as.numeric(q[[3]]), q[[4]]), numeric(1))
  x <- cx - sum(ws) / 2
  for (k in seq_along(pieces)) {
    q <- pieces[[k]]
    grid.text(q[[1]], x = xx(x), y = yy(ty), hjust = 0,
              gp = gpar(fontfamily = q[[2]], fontsize = as.numeric(q[[3]]),
                        col = col, fontface = q[[4]]))
    x <- x + ws[k]
  }
}
xx <- function(x) unit(x, "in")
box_ct <- function(i) T_BOX0 + (i - 1) * (BX_H + BX_GAP) + BX_H / 2

# ---- one panel; the arrow is produced here and nowhere else -----------------
draw_panel <- function(p) {
  px <- PX[p]
  grid.text(LETTERS[p], x = xx(px + 0.01), y = yy(T_TITLE), hjust = 0,
            gp = gpar(fontfamily = SERIF, fontsize = 8.2, col = LAB_INK, fontface = "bold"))
  grid.text(TITLE[p], x = xx(px + 0.01 + w_of(LETTERS[p], SERIF, 8.2, "bold") + 0.045),
            y = yy(T_TITLE), hjust = 0,
            gp = gpar(fontfamily = SERIF, fontsize = 7.8, col = LAB_GREY))
  for (i in seq_along(MODES)) {
    req <- (i == REQ[p])
    grid.roundrect(x = xx(px + 0.01 + BX_W / 2), y = yy(box_ct(i)),
                   width = xx(BX_W), height = xx(BX_H), r = unit(0.028, "in"),
                   gp = gpar(col = if (req) REQ_EDGE else BOX_EDGE,
                             lwd = if (req) 1.5 else 0.6, fill = NA))
    grid.text(MODES[i], x = xx(px + 0.01 + BX_W / 2), y = yy(box_ct(i)),
              gp = gpar(fontfamily = MONO, fontsize = 7.3,
                        col = if (req) LAB_INK else LAB_GREY,
                        fontface = if (req) "bold" else "plain"))
  }
  # the observed move: same relative offsets in every panel
  ex <- px + 0.01 + BX_W
  grid.xspline(x = xx(c(ex, ex + 0.155, ex)),
               y = yy(c(box_ct(1), (box_ct(1) + box_ct(2)) / 2, box_ct(2))),
               shape = 1, open = TRUE,
               arrow = arrow(length = unit(0.042, "in"), type = "closed",
                             angle = 22, ends = "last"),
               gp = gpar(col = ARROW_COL, lwd = 0.9, fill = ARROW_COL))
}

draw_all <- function() {
  grid.newpage()
  for (p in 1:3) draw_panel(p)

  band_lab <- function(j, s)
    grid.text(s, x = xx(LAB_X), y = yy(T_BAND[j]), hjust = 1,
              gp = gpar(fontfamily = SERIF, fontsize = 7.6, col = LAB_GREY))
  seg <- function(x0, x1, j, fill, edge = NA, lty = "solid", lwd = 0.6)
    grid.roundrect(x = xx((x0 + x1) / 2), y = yy(T_BAND[j]),
                   width = xx(x1 - x0), height = xx(BAND_H), r = unit(0.024, "in"),
                   gp = gpar(fill = fill, col = edge, lty = lty, lwd = lwd))
  txt <- function(x, j, s, col, fam = SERIF, size = 7.4, face = "plain")
    grid.text(s, x = xx(x), y = yy(T_BAND[j]),
              gp = gpar(fontfamily = fam, fontsize = size, col = col, fontface = face))

  # (1) transition -- one piece: it separates nothing
  band_lab(1, "transition")
  seg(PX[1], PX[3] + PW, 1, BAND_FILL)
  run(list(list("tool_call", MONO, 7.3, "plain"),
           list(" \u2192 ", SERIF, 7.8, "plain"),
           list("request_for_info", MONO, 7.3, "plain"),
           list("  in all three", SERIF, 7.4, "plain")),
      cx = (PX[1] + PX[3] + PW) / 2, ty = T_BAND[1], col = LAB_GREY)

  # (2) accuracy -- A and C filled, B an empty dashed frame: it separates two
  band_lab(2, "accuracy")
  for (p in c(1, 3)) {
    seg(PX[p], PX[p] + PW, 2, BAND_FILL)
    txt(PX[p] + PW / 2, 2, if (p == 1) "+1" else "−1", LAB_GREY)
  }
  seg(PX[2], PX[2] + PW, 2, NA, DASH_EDGE, "22", 0.7)
  txt(PX[2] + PW / 2, 2, "0 · invisible", DASH_EDGE)

  # (3) destination -- three pieces: it separates all three
  band_lab(3, "destination")
  for (p in 1:3) {
    seg(PX[p], PX[p] + PW, 3, DEST[[p]]$fill, DEST[[p]]$ink, "solid", 0.5)
    txt(PX[p] + PW / 2, 3, DEST[[p]]$text, DEST[[p]]$ink, SERIF, 7.4, "bold")
  }

}

dir.create("figures", showWarnings = FALSE)
cairo_pdf("figures/casestudy_transition.pdf", width = W_IN, height = H_IN, family = SERIF)
draw_all(); invisible(dev.off())
png("figures/casestudy_transition.png", width = W_IN, height = H_IN, units = "in",
    res = 300, type = "cairo", family = SERIF)
draw_all(); invisible(dev.off())
cat("  wrote figures/casestudy_transition.pdf / .png\n")
