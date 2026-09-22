from econterm.storage import Repository
from econterm.api import fetch_series, fetch_series_info
from econterm.analysis import prepare, ols, format_ols, diff, pct_change, log_diff, log_diff_pct
from econterm.viz import plot_series, plot_regression
from pathlib import Path
import numpy as np
import httpx
import asyncio
import matplotlib.pyplot as plt

DB_PATH = Path(__file__).parent / "econterm.db"

VARIABLES = {
    'y': 'GDPC1', 
    'x': 'UNRATE'
}

async def main():
    with Repository(DB_PATH) as repo:
        repo.init_db()
        CONCURRENCY_LIMIT = 5
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        series_ids = ['GDPC1', 'UNRATE', 'CPIAUCSL']
        
        # # fetch metadata and observations from FRED for each of our 3 series
        # async with httpx.AsyncClient() as client:
        #     tasks_info = [fetch_series_info(series_id, client, sem) for series_id in series_ids]
        #     results_info = await asyncio.gather(*tasks_info)
        #     tasks_obs = [fetch_series(series_id, "1980-01-01", "2010-01-01", client, sem) for series_id in series_ids]
        #     results_obs = await asyncio.gather(*tasks_obs)
            
        # # save observations and series, unpacked from Observation dataclass into list of tuples
        # with repo.transaction():
        #     for series_id, info, observations in zip(series_ids, results_info, results_obs):
        #         repo.save_observations(series_id, [(obs.date, obs.value) for obs in observations])
        #         repo.save_series(series_id, info.title, info.units, info.frequency, info.last_updated)

        y_info = repo.get_series_metadata(VARIABLES['y'])
        y_obs = repo.get_observations(VARIABLES['y'])
        y_dates = np.array([obs.date for obs in y_obs], dtype="datetime64[D]")
        y_values = np.array([obs.value for obs in y_obs], dtype=float)
        x_info = repo.get_series_metadata(VARIABLES['x'])
        x_obs = repo.get_observations(VARIABLES['x'])
        x_dates = np.array([obs.date for obs in x_obs], dtype="datetime64[D]")
        x_values = np.array([obs.value for obs in x_obs], dtype=float)
        dates, y, x = prepare(y_dates, y_values, y_info.frequency, x_dates, x_values, x_info.frequency, log_diff_pct, diff)
        
        X = np.column_stack([np.ones(len(x)), x])
        result = ols(y, X)
        summary = format_ols(result, ['const', 'UNRATE'])
        
        fig = plot_regression(x, y, dates, result, title="Real GDP growth vs. change in unemployment", x_label="Δ unemployment rate (pp)", y_label="Δ log real GDP")
        fig.savefig("test_regression.png")
        
        fig = plot_series(dates, x, title="UNRATE", y_label="Hmm")
        fig.savefig("test_series.png")
        plt.close(fig)
        
        print(summary)
        
asyncio.run(main())