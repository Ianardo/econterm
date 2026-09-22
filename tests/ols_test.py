import numpy as np
import pytest
import statsmodels.api as sm
from econterm.analysis import ols


def make_data(seed, n, k):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k - 1))
    X = np.column_stack([np.ones(n), x])
    true_b = rng.normal(size=k)
    y = X @ true_b + rng.normal(scale=0.5, size=n)
    return y, X


@pytest.mark.parametrize("seed, n, k", [
    (0, 20, 2),
    (1, 50, 2),
    (2, 100, 2),
    (3, 30, 3),
    (4, 200, 4),
])
def test_ols_matches_statsmodels(seed, n, k):
    y, X = make_data(seed, n, k)
    ours = ols(y, X)
    ref = sm.OLS(y, X).fit()

    np.testing.assert_allclose(ours.coefficients, ref.params, rtol=1e-6)
    np.testing.assert_allclose(ours.std_errors, ref.bse, rtol=1e-6)
    np.testing.assert_allclose(ours.tvalues, ref.tvalues, rtol=1e-6)
    np.testing.assert_allclose(ours.pvalues, ref.pvalues, rtol=1e-6)
    np.testing.assert_allclose(ours.r_squared, ref.rsquared, rtol=1e-6)
    np.testing.assert_allclose(ours.adj_r_squared, ref.rsquared_adj, rtol=1e-6)