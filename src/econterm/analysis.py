import numpy as np
import scipy.stats
import tabulate
from collections import defaultdict
from datetime import date
from econterm.models import OLSResult
import random

def to_quarterly(dates, values):
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

def diff(values):
    return values[1:] - values[:-1]

def pct_change(values):
    return (values[1:] - values[:-1]) / values[:-1]

def log_diff(values):
    return np.log(values[1:]) - np.log(values[:-1])

def prepare(y_dates, y_values, y_freq, x_dates, x_values, x_freq, y_transform, x_transform):
    # if only one is quarterly, convert the other
    if y_freq != x_freq:
        if x_freq == "Quarterly":
            y_dates, y_values = to_quarterly(y_dates, y_values)
        else:
            x_dates, x_values = to_quarterly(x_dates, x_values)
    
    y_values = y_transform(y_values)
    y_dates = y_dates[1:]
    x_values = x_transform(x_values)
    x_dates = x_dates[1:]
    
    common, iy, ix = np.intersect1d(y_dates, x_dates, return_indices=True)

    dates, iy, ix = np.intersect1d(y_dates, x_dates, return_indices=True)
    y = y_values[iy]
    x = x_values[ix]
    
    mask = ~np.isnan(y) & ~np.isnan(x)
    return dates[mask], y[mask], x[mask]

def ols(y, X):
    n = len(y)
    k = X.shape[1]  
    df_resid = n - k  # degrees of freedom for residuals

    # coefficients
    X_T = X.T
    rhs = X_T @ y
    b = np.linalg.solve((X_T @ X), rhs)
    
    # residuals & variance
    e = y - (X @ b) 
    s_squared = (e @ e) / df_resid
    
    # standard errors of coefficients
    std_errors = np.sqrt(np.diag(s_squared * np.linalg.solve(X_T @ X, np.eye(k)))) # sqrt of diagonal of s^2 * (X^T * X)^-1
    
    # condition number
    condition_number = float(np.linalg.cond(X))
    
    # t-values and p-values
    tvalues = b / std_errors
    pvalues = 2 * scipy.stats.t.sf(np.abs(tvalues), df=df_resid)
    
    # R^2
    r_squared = 1 - (e @ e) / (((y - y.mean())) @ (y - y.mean()))
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
        residuals=e
    )

def format_ols(result, names):
    width = 10 + 4 * 12
    lines = [
        f"{'':<10}{'coef':>12}{'se':>12}{'t':>12}{'p':>12}",
        "-" * width,
    ]
    for name, coef, se, t, p in zip(names, result.coefficients, result.std_errors,
                                    result.tvalues, result.pvalues):
        lines.append(f"{name:<10}{coef:>12.4f}{se:>12.4f}{t:>12.3f}{p:>12.3f}")
    lines.append("-" * width)
    lines.append(f"{'n':<22}{result.n:>12}")
    lines.append(f"{'R²':<22}{result.r_squared:>12.4f}")
    lines.append(f"{'Adj. R²':<22}{result.adj_r_squared:>12.4f}")
    lines.append(f"{'Condition number':<22}{result.condition_number:>12.1f}")
    return "\n".join(lines)