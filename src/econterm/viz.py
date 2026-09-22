import matplotlib.pyplot as plt
import numpy as np
from econterm.models import OLSResult

PALETTE = ["#086D92",  '#ef4123', '#00a3a6', '#f6921e']

STYLE = {
    # DPI
    'figure.dpi': 300,     
    
    # z order
    'axes.axisbelow': True,
    
    # font
    'font.family': ['Inter 18pt', 'DejaVu Sans'],
    'text.color': '#222222',
    'font.size': 10,
    
    # figure size and background
    'figure.figsize': (8, 6),
    'figure.facecolor': '#f1f4f5',      
    
    # margins
    'figure.subplot.top': 0.85,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.10,
    'figure.subplot.right': 0.90,
    
    # axes-level title
    'axes.titlesize': 14,
    'axes.titleweight': '700',
    'axes.titlecolor': '#111111',
    'axes.titlelocation': 'left',       
    'axes.titlepad': 8,                
    
    # axes colors
    'axes.facecolor': '#f1f4f5',
    'axes.edgecolor': '#111111',
    'axes.linewidth': 1.0,
    'axes.labelcolor': '#4a4a4a',
    
    # spines
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': True,
    
    # colors
    'axes.prop_cycle': plt.cycler(color=PALETTE),
    
    # chart lines
    'lines.linewidth': 2.5,
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'none',          
    
    # gridlines
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#ccd4d8',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    
    # ticks
    'xtick.color': '#111111',
    'xtick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.major.width': 1.0,
    'xtick.labelsize': 9,
    'ytick.color': '#111111',
    'ytick.major.size': 0,              
    'ytick.labelsize': 9,
}

RECESSION_COLOR = "#ecabab"
DOT_COLOR = "#222"

def recession_spans(dates, flags):
    d = np.diff(flags)
    starts = list(dates[1:][d == 1])
    ends = list(dates[1:][d == -1])
    if flags[0]:
        starts.insert(0, dates[0])
    if flags[-1]:
        ends.append(dates[-1])
    return list(zip(starts, ends))

def plot_series(dates, values, *, title, y_label, recessions=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(dates, values, clip_on=False)
        if recessions is not None:
            for start, end in recessions:
                ax.axvspan(start, end, alpha=0.2, facecolor=RECESSION_COLOR)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylabel(y_label)
        ax.set_title(title)
        return fig

def plot_regression(x, y, dates, result, *, title, x_label, y_label): # -> Figure
    with plt.rc_context(STYLE):
        # 1 col, 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3,1])
        fig.subplots_adjust(hspace=0.3) 
        
        # scatter plot
        ax1.scatter(x, y, c=DOT_COLOR, zorder=2)
        b = result.coefficients
        fitted_line_x = np.array([np.min(x), np.max(x)])
        fitted_line_y = b[1] * fitted_line_x + b[0]
        ax1.plot(fitted_line_x, fitted_line_y, color=PALETTE[1])
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(title)
        
        # residual plot
        ax2.scatter(dates, result.residuals, s=20, facecolor=DOT_COLOR, zorder=2)
        ax2.axhline(0, dashes=[3, 3], linewidth=2, color='#bbbbc0', zorder=1)
        
        L = 1.15 * np.max(np.abs(result.residuals))
        ax2.set_ylim(-L, L)
        ax2.set_ylabel("Residual")
        return fig
    
if __name__ == "__main__":
    import pandas as pd
    from econterm.analysis import ols   # adjust to where ols() lives

    n = 40
    dates = pd.date_range("2000-01-01", periods=n, freq="QS").to_numpy().astype("datetime64[D]")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(len(x)), x])
    y = 2.0 + 0.5 * x + rng.normal(0, 0.3, n)   # true intercept 2.0, slope 0.5

    result = ols(y, X)   # match your ols() argument order / constant handling

    fig = plot_regression(x, y, dates, result, title="Regression test", x_label="x (test)", y_label="y (test)")
    fig.savefig("test_regression.png")
    plt.close(fig)
    print("wrote test_regression.png")