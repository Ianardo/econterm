"""Chart functions. Each returns a Figure; the caller saves and closes it."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

PALETTE = ["#086D92", "#ef4123", "#00a3a6", "#f6921e"]

# Use only installed fonts so machines without Inter don't get findfont warnings.
_PREFERRED_FONTS = ["Inter 18pt", "Arial", "Liberation Sans"]
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONT_FAMILY = [name for name in _PREFERRED_FONTS if name in _INSTALLED] + ["DejaVu Sans"]

# Applied with plt.rc_context inside each plot function, so global defaults are untouched.
STYLE = {
    "figure.dpi": 300,
    "axes.axisbelow": True,
    "font.family": FONT_FAMILY,
    "text.color": "#222222",
    "font.size": 10,
    "figure.figsize": (8, 6),
    "figure.facecolor": "#f1f4f5",
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.10,
    "figure.subplot.right": 0.90,
    "axes.facecolor": "#f1f4f5",
    "axes.edgecolor": "#111111",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#4a4a4a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "lines.linewidth": 2.5,
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "none",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#ccd4d8",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "xtick.color": "#111111",
    "xtick.direction": "out",
    "xtick.major.size": 4,
    "xtick.major.width": 1.0,
    "xtick.labelsize": 9,
    "ytick.color": "#111111",
    "ytick.major.size": 0,
    "ytick.labelsize": 9,
}

RECESSION_COLOR = "#ecabab"
DOT_COLOR = "#222"

# Title block. Spacing is in points so it holds if figsize changes.
TEXT_LEFT = 0.03          # figure fraction; shared left edge for title, subtitle, source
TITLE_SIZE, SUBTITLE_SIZE, SOURCE_SIZE = 14, 10, 8
TITLE_COLOR, SUBTITLE_COLOR, SOURCE_COLOR = "#111111", "#555555", "#888888"
LINE_SPACING = 1.2
TOP_MARGIN_PT = 16        # figure edge to title
SUBTITLE_GAP_PT = 5       # title to subtitle
AXES_GAP_PT = 20          # title block to axes
SOURCE_BOTTOM_PT = 10     # figure edge to source line
TITLE_WIDTH = 60          # characters per title line
TITLE_MAX_LINES = 2


def _add_titles(fig, title, subtitle=None, source=None):
    """Add a wrapped title, optional subtitle and source line, and move the axes top to fit."""
    height_pt = fig.get_size_inches()[1] * 72

    def frac(pt):
        return pt / height_pt

    title = textwrap.fill(title, width=TITLE_WIDTH, max_lines=TITLE_MAX_LINES, placeholder=" …")
    y = 1 - frac(TOP_MARGIN_PT)
    fig.text(TEXT_LEFT, y, title, ha="left", va="top", fontsize=TITLE_SIZE,
             fontweight="bold", color=TITLE_COLOR, linespacing=LINE_SPACING)
    y -= frac(TITLE_SIZE * LINE_SPACING * (title.count("\n") + 1))

    if subtitle:
        y -= frac(SUBTITLE_GAP_PT)
        fig.text(TEXT_LEFT, y, subtitle, ha="left", va="top",
                 fontsize=SUBTITLE_SIZE, color=SUBTITLE_COLOR)
        y -= frac(SUBTITLE_SIZE * LINE_SPACING)

    # the axes start below however much space the title block used
    fig.subplots_adjust(top=y - frac(AXES_GAP_PT))

    if source:
        fig.text(TEXT_LEFT, frac(SOURCE_BOTTOM_PT), source, ha="left", va="bottom",
                 fontsize=SOURCE_SIZE, color=SOURCE_COLOR)


def recession_spans(dates, flags):
    """Convert 0/1 recession flags into (start, end) date pairs."""
    d = np.diff(flags)
    starts = list(dates[1:][d == 1])
    ends = list(dates[1:][d == -1])
    if flags[0]:
        starts.insert(0, dates[0])
    if flags[-1]:
        ends.append(dates[-1])
    return list(zip(starts, ends))


def plot_series(dates, values, *, title, y_label, subtitle=None, source=None, recessions=None):
    """Line plot of one series, optionally with shaded recession spans."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        # clip_on=False keeps the thick line from being cut in half at the axis edges
        ax.plot(dates, values, clip_on=False)
        if recessions is not None:
            for start, end in recessions:
                ax.axvspan(start, end, alpha=0.2, facecolor=RECESSION_COLOR)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylabel(y_label)
        _add_titles(fig, title, subtitle, source)
        return fig


def plot_regression(x, y, dates, result, *, title, x_label, y_label, subtitle=None, source=None):
    """Scatter with fitted line, plus residuals over time below. Assumes one regressor."""
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1])
        fig.subplots_adjust(hspace=0.3)

        ax1.scatter(x, y, c=DOT_COLOR, zorder=2)
        b = result.coefficients
        fitted_line_x = np.array([np.min(x), np.max(x)])
        ax1.plot(fitted_line_x, b[1] * fitted_line_x + b[0], color=PALETTE[1])
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)

        ax2.scatter(dates, result.residuals, s=20, facecolor=DOT_COLOR, zorder=2)
        ax2.axhline(0, dashes=[3, 3], linewidth=2, color="#bbbbc0", zorder=1)
        # symmetric limits keep zero centered; 15% margin keeps dots from clipping
        L = 1.15 * np.max(np.abs(result.residuals))
        ax2.set_ylim(-L, L)
        ax2.set_ylabel("Residual")

        _add_titles(fig, title, subtitle, source)
        return fig