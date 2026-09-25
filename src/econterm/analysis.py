"""Data preparation and OLS regression."""

from collections import defaultdict
from datetime import date

import numpy as np
import scipy.stats

from econterm.models import OLSResult


def _transform_values(transform, dates, values):
    """Apply a transform and trim dates to match. Transforms may only drop leading values."""
    values = transform(values)
    if len(values) > len(dates):
        raise ValueError(
            f"transform returned {len(values)} values for {len(dates)} dates"
        )
    dropped = len(dates) - len(values)
    dates = dates[dropped:]
    return dates, values


def crop(dates, *arrays, start=None, end=None):
    """Keep observations between start and end, inclusive. Call after transforming."""
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= np.datetime64(start, "D")
    if end is not None:
        mask &= dates <= np.datetime64(end, "D")
    return (dates[mask], *(a[mask] for a in arrays))


def to_quarterly(dates, values):
    """Average monthly values into quarters. Incomplete quarters are dropped."""
    quarter_dates = []
    quarter_values = []

    groups = defaultdict(list)
    for d, value in zip(dates.astype(object), values):
        quarter_month = ((d.month - 1) // 3) * 3 + 1
        key = date(d.year, quarter_month, 1)
        groups[key].append(value)
    for d, vals in groups.items():
        if len(vals) == 3:
            quarter_dates.append(d)
            quarter_values.append(np.mean(vals))
    return np.array(quarter_dates, dtype="datetime64[D]"), np.array(quarter_values)


def level(values):
    return values


def diff(values):
    return values[1:] - values[:-1]


def pct_change(values):
    return (values[1:] - values[:-1]) / values[:-1]


def log_diff(values):
    return np.log(values[1:]) - np.log(values[:-1])


def log_diff_pct(values):
    return log_diff(values) * 100


def prepare(
    y_dates, y_values, y_freq, x_dates, x_values, x_freq, y_transform, x_transform
):
    """Convert frequency, transform, align dates, and drop NaNs. Returns (dates, y, x)."""
    # only handles monthly/quarterly pairs
    if y_freq != x_freq:
        if x_freq == "Quarterly":
            y_dates, y_values = to_quarterly(y_dates, y_values)
        else:
            x_dates, x_values = to_quarterly(x_dates, x_values)

    y_dates, y_values = _transform_values(y_transform, y_dates, y_values)
    x_dates, x_values = _transform_values(x_transform, x_dates, x_values)

    dates, iy, ix = np.intersect1d(y_dates, x_dates, return_indices=True)
    y = y_values[iy]
    x = x_values[ix]

    mask = ~np.isnan(y) & ~np.isnan(x)
    return dates[mask], y[mask], x[mask]


def ols(y, X):
    """OLS via the normal equations. X must include a constant column for an intercept."""
    n = len(y)
    k = X.shape[1]
    df_resid = n - k

    X_T = X.T
    b = np.linalg.solve(X_T @ X, X_T @ y)

    e = y - (X @ b)
    s_squared = (e @ e) / df_resid

    # sqrt of the diagonal of s^2 * (X'X)^-1
    std_errors = np.sqrt(np.diag(s_squared * np.linalg.solve(X_T @ X, np.eye(k))))

    condition_number = float(np.linalg.cond(X))

    tvalues = b / std_errors
    pvalues = 2 * scipy.stats.t.sf(np.abs(tvalues), df=df_resid)

    r_squared = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

    return OLSResult(
        coefficients=b,
        std_errors=std_errors,
        condition_number=condition_number,
        tvalues=tvalues,
        pvalues=pvalues,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        n=n,
        residuals=e,
    )


def format_ols(result, names):
    """Format an OLSResult as a text table."""
    width = 10 + 4 * 12
    lines = [
        f"{'':<10}{'coef':>12}{'se':>12}{'t':>12}{'p':>12}",
        "-" * width,
    ]
    for name, coef, se, t, p in zip(
        names, result.coefficients, result.std_errors, result.tvalues, result.pvalues
    ):
        lines.append(f"{name:<10}{coef:>12.4f}{se:>12.4f}{t:>12.3f}{p:>12.3f}")
    lines.append("-" * width)
    lines.append(f"{'n':<22}{result.n:>12}")
    lines.append(f"{'R²':<22}{result.r_squared:>12.4f}")
    lines.append(f"{'Adj. R²':<22}{result.adj_r_squared:>12.4f}")
    lines.append(f"{'Condition number':<22}{result.condition_number:>12.1f}")
    return "\n".join(lines)