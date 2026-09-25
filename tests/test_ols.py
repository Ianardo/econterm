"""Check econterm's OLS against statsmodels to six decimal places."""

import numpy as np
import pytest
import statsmodels.api as sm

from econterm.analysis import ols


def make_data(n, k, seed):
    """Random regressors with a constant column, and y = X @ beta + noise."""
    rng = np.random.default_rng(seed)
    X = sm.add_constant(rng.normal(size=(n, k)))
    beta = rng.normal(size=k + 1)
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return y, X


@pytest.mark.parametrize(
    "n, k, seed",
    [
        (40, 1, 0),   # small sample, one regressor
        (200, 1, 1),  # larger sample
        (100, 3, 2),  # several regressors
    ],
)
def test_ols_matches_statsmodels(n, k, seed):
    y, X = make_data(n, k, seed)
    ours = ols(y, X)
    ref = sm.OLS(y, X).fit()

    checks = {
        "coefficients": (ours.coefficients, ref.params),
        "std_errors": (ours.std_errors, ref.bse),
        "tvalues": (ours.tvalues, ref.tvalues),
        "pvalues": (ours.pvalues, ref.pvalues),
        "residuals": (ours.residuals, ref.resid),
        "r_squared": (ours.r_squared, ref.rsquared),
        "adj_r_squared": (ours.adj_r_squared, ref.rsquared_adj),
        "condition_number": (ours.condition_number, ref.condition_number),
    }
    for name, (mine, theirs) in checks.items():
        np.testing.assert_allclose(mine, theirs, atol=1e-6, err_msg=name)

    assert ours.n == n