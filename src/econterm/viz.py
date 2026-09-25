"""Chart functions. Each returns a Figure; the caller saves and closes it."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import textwrap

TITLE_WIDTH = 60
TITLE_MAX_LINES = 2

PALETTE = ["#086D92", "#ef4123", "#00a3a6", "#f6921e"]

# Use only fonts that are installed, so machines without Inter don't get
# findfont warnings. DejaVu Sans ships with matplotlib, so it's always available.
_PREFERRED_FONTS = ["Inter 18pt", "Arial", "Liberation Sans"]
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONT_FAMILY = [name for name in _PREFERRED_FONTS if name in _INSTALLED] + [
    "DejaVu Sans"
]

# Applied with plt.rc_context inside each plot function, so it never changes
# the defaults for other code that uses matplotlib.
STYLE = {
    # DPI
    "figure.dpi": 300,
    # gridlines and ticks drawn beneath data and recession bands
    "axes.axisbelow": True,
    # font
    "font.family": FONT_FAMILY,
    "text.color": "#222222",
    "font.size": 10,
    # figure size and background
    "figure.figsize": (8, 6),
    "figure.facecolor": "#f1f4f5",
    # margins
    "figure.subplot.top": 0.85,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.10,
    "figure.subplot.right": 0.90,
    # axes-level title
    "axes.titlesize": 14,
    "axes.titleweight": "700",
    "axes.titlecolor": "#111111",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    # axes colors
    "axes.facecolor": "#f1f4f5",
    "axes.edgecolor": "#111111",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#4a4a4a",
    # spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    # colors
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    # chart lines
    "lines.linewidth": 2.5,
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "none",
    # gridlines
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#ccd4d8",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    # ticks
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

def _wrap_title(title):
    return textwrap.fill(title, width=TITLE_WIDTH, max_lines=TITLE_MAX_LINES, placeholder=" …")

def recession_spans(dates, flags):
    """Convert 0/1 recession flags into (start, end) date pairs.

    Each end is the first non-recession date, except for a recession still
    ongoing at the last date, which ends at dates[-1].
    """
    d = np.diff(flags)
    starts = list(dates[1:][d == 1])
    ends = list(dates[1:][d == -1])
    if flags[0]:
        starts.insert(0, dates[0])
    if flags[-1]:
        ends.append(dates[-1])
    return list(zip(starts, ends))


def plot_series(dates, values, *, title, y_label, recessions=None):
    """Line plot of one series, optionally with shaded recession spans."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        # clip_on=False keeps the thick line from being cut in half at the axis edges
        ax.plot(dates, values, clip_on=False)
        if recessions is not None:
            for start, end in recessions:
                ax.axvspan(start, end, alpha=0.2, facecolor=RECESSION_COLOR)
        # pin the axis to the data; spans outside this range are clipped
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylabel(y_label)
        ax.set_title(_wrap_title(title))
        return fig


def plot_regression(x, y, dates, result, *, title, x_label, y_label):
    """Scatter with fitted line, plus residuals over time in a panel below.

    Assumes a single regressor: coefficients are [intercept, slope].
    """
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1])
        fig.subplots_adjust(hspace=0.3)

        ax1.scatter(x, y, c=DOT_COLOR, zorder=2)
        b = result.coefficients
        # a straight line only needs its two endpoints
        fitted_line_x = np.array([np.min(x), np.max(x)])
        fitted_line_y = b[1] * fitted_line_x + b[0]
        ax1.plot(fitted_line_x, fitted_line_y, color=PALETTE[1])
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(_wrap_title(title))

        ax2.scatter(dates, result.residuals, s=20, facecolor=DOT_COLOR, zorder=2)
        ax2.axhline(0, dashes=[3, 3], linewidth=2, color="#bbbbc0", zorder=1)

        # symmetric limits keep zero centered; the 15% margin keeps dots from clipping
        L = 1.15 * np.max(np.abs(result.residuals))
        ax2.set_ylim(-L, L)
        ax2.set_ylabel("Residual")
        return fig