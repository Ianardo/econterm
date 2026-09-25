import numpy as np

from econterm.analysis import diff, log_diff, prepare


def test_prepare():
    y_dates = np.array(
        ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"],
        dtype="datetime64[D]",
    )
    y_values = np.array([100.0, 102.0, 101.0, 104.0])

    x_dates = np.array(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-09-01",
        ],
        dtype="datetime64[D]",
    )
    x_values = np.array([3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 8.0, 9.0, 10.0])

    dates, y, x = prepare(
        y_dates,
        y_values,
        "Quarterly",
        x_dates,
        x_values,
        "Monthly",
        y_transform=log_diff,
        x_transform=diff,
    )

    np.testing.assert_array_equal(
        dates, np.array(["2020-04-01", "2020-07-01"], dtype="datetime64[D]")
    )
    np.testing.assert_allclose(y, [np.log(102 / 100), np.log(101 / 102)])
    np.testing.assert_allclose(x, [9.0, -4.0])
