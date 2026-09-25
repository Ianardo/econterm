"""econterm command-line interface."""

import asyncio
from pathlib import Path

import click
import httpx
import matplotlib.pyplot as plt
import numpy as np

from econterm.analysis import (
    crop,
    diff,
    format_ols,
    level,
    log_diff,
    log_diff_pct,
    ols,
    pct_change,
    prepare,
)
from econterm.api import (
    FredApiError,
    RateLimited,
    SeriesNotFound,
    fetch_one,
    get_api_key,
)
from econterm.storage import Repository
from econterm.viz import plot_regression, plot_series, recession_spans

DB_PATH = Path.home() / ".econterm" / "econterm.db"
CONCURRENCY_LIMIT = 5

TRANSFORMS = {
    "level": level,
    "diff": diff,
    "pct_change": pct_change,
    "log_diff": log_diff,
    "log_diff_pct": log_diff_pct,
}


def _to_arrays(observations):
    """Convert repository observations to (dates, values) NumPy arrays."""
    dates = np.array([obs.date for obs in observations], dtype="datetime64[D]")
    values = np.array([obs.value for obs in observations], dtype=float)
    return dates, values


def _fmt_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M UTC")

def _fmt_range(dates):
    """'Jan 1980 – Oct 2010' from a datetime64 array."""
    first, last = dates[0].astype(object), dates[-1].astype(object)
    return f"{first:%b %Y} – {last:%b %Y}"

def _save(fig, path):
    """Save a figure, creating its folder if needed, then close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)  # pyplot keeps figures in memory until closed
    click.echo(f"Saved {path}")


@click.group()
def cli():
    """Fetch, plot, and regress FRED economic data."""


@cli.command()
@click.argument("series_ids", nargs=-1, required=True)
def fetch(series_ids):
    """Download SERIES_IDS from FRED and save them locally, e.g. GDPC1 UNRATE USREC."""

    async def fetch_all():
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        async with httpx.AsyncClient(timeout=30) as client:
            return await asyncio.gather(*(fetch_one(sid, client, sem) for sid in series_ids))

    try:
        get_api_key()  # fail before any requests if the key is missing
        click.echo("Fetching from FRED...", err=True)
        results = asyncio.run(fetch_all())
    except SeriesNotFound as e:
        raise click.ClickException(
            f"FRED has no series '{e.series_id}'. Check the ID on fred.stlouisfed.org."
        )
    except RateLimited:
        raise click.ClickException(
            "FRED rate limit reached after retries. Wait a minute and try again."
        )
    except httpx.TransportError:
        raise click.ClickException("Couldn't reach FRED. Check your internet connection.")
    except FredApiError as e:  # includes MissingApiKey
        raise click.ClickException(str(e))

    with Repository(DB_PATH) as repo, repo.transaction():
        for series_id, (info, observations) in zip(series_ids, results):
            repo.save_observations(series_id, [(obs.date, obs.value) for obs in observations])
            repo.save_series(
                series_id,
                info.title,
                info.units,
                info.frequency,
                info.last_updated,
                info.fetched_at,
            )
            click.echo(f"{series_id}: {len(observations)} observations")


@cli.command()
@click.argument("series_id")
@click.option("--start", help="Start date, YYYY-MM-DD")
@click.option("--end", help="End date, YYYY-MM-DD")
@click.option("--no-recessions", is_flag=True, help="Don't shade recessions.")
@click.option("--title", help="Chart title. Defaults to the FRED title.")
def plot(series_id, start, end, no_recessions, title):
    """Plot SERIES_ID with recession shading."""
    with Repository(DB_PATH) as repo:
        info = repo.get_series_metadata(series_id)
        obs = repo.get_observations(series_id)
        rec_obs = [] if no_recessions else repo.get_observations("USREC")

    if not obs:
        raise click.ClickException(
            f"{series_id} is not stored. Run `econterm fetch {series_id}` first."
        )

    recessions = None
    if rec_obs:
        rec_dates, rec_flags = _to_arrays(rec_obs)
        recessions = recession_spans(rec_dates, rec_flags)
    elif not no_recessions:
        click.echo("Note: USREC not stored; run `econterm fetch USREC` for recession shading.", err=True)

    dates, values = _to_arrays(obs)
    dates, values = crop(dates, values, start=start, end=end)
    if len(dates) == 0:
        raise click.ClickException(f"No {series_id} data between {start} and {end}.")

    source = f"Source: FRED, Federal Reserve Bank of St. Louis ({series_id})."
    if recessions:
        source += " Shaded areas indicate U.S. recessions."

    fig = plot_series(
        dates,
        values,
        title=title or info.title,
        subtitle=f"{info.frequency}, {_fmt_range(dates)}",
        source=source,
        y_label=info.units,
        recessions=recessions,
    )
    _save(fig, Path("charts") / "series" / f"{series_id}.png")

@cli.command()
@click.argument("y_id")
@click.argument("x_id")
@click.option("--y-transform", default="level", type=click.Choice(TRANSFORMS.keys()),
              help="Transform applied to Y_ID.")
@click.option("--x-transform", default="level", type=click.Choice(TRANSFORMS.keys()),
              help="Transform applied to X_ID.")
@click.option("--start", help="Start date, YYYY-MM-DD")
@click.option("--end", help="End date, YYYY-MM-DD")
@click.option("--no-plot", is_flag=True, help="Print the table only; don't save a plot.")
def regress(y_id, x_id, y_transform, x_transform, start, end, no_plot):
    """Regress Y_ID on X_ID, print the coefficient table, and save a plot."""
    with Repository(DB_PATH) as repo:
        y_info = repo.get_series_metadata(y_id)
        y_obs = repo.get_observations(y_id)
        x_info = repo.get_series_metadata(x_id)
        x_obs = repo.get_observations(x_id)

    for sid, obs in ((y_id, y_obs), (x_id, x_obs)):
        if not obs:
            raise click.ClickException(f"{sid} is not stored. Run `econterm fetch {sid}` first.")

    y_dates, y_values = _to_arrays(y_obs)
    x_dates, x_values = _to_arrays(x_obs)
    dates, y, x = prepare(
        y_dates,
        y_values,
        y_info.frequency,
        x_dates,
        x_values,
        x_info.frequency,
        TRANSFORMS[y_transform],
        TRANSFORMS[x_transform],
    )
    # crop after prepare so the first difference in range uses the prior observation
    dates, y, x = crop(dates, y, x, start=start, end=end)

    if len(y) < 3:
        raise click.ClickException(
            f"Only {len(y)} overlapping observations in this range; need at least 3."
        )

    X = np.column_stack([np.ones(len(x)), x])
    result = ols(y, X)

    click.echo(f"{y_id}: {y_info.title} (fetched {_fmt_time(y_info.fetched_at)})")
    click.echo(f"{x_id}: {x_info.title} (fetched {_fmt_time(x_info.fetched_at)})")
    click.echo(format_ols(result, ["const", x_id]))

    if not no_plot:
        fig = plot_regression(
            x,
            y,
            dates,
            result,
            title=f"{y_info.title} vs {x_info.title}",
            subtitle=f"{_fmt_range(dates)}, n = {result.n}, R² = {result.r_squared:.2f}",
            source=f"Source: FRED, Federal Reserve Bank of St. Louis ({y_id}, {x_id}).",
            x_label=f"{x_id} ({x_transform})",
            y_label=f"{y_id} ({y_transform})",
        )
        _save(fig, Path("charts") / "regressions" / f"{y_id}_{x_id}.png")

@cli.command(name="list")
def list_series_cmd():
    """List stored series and when each was fetched."""
    with Repository(DB_PATH) as repo:
        rows = repo.list_series()
    if not rows:
        click.echo("No series stored. Run `econterm fetch SERIES_ID` to add one.")
        return
    for row in rows:
        click.echo(f"{row.series_id} — {row.title}")
        click.echo(f"    Fetched {_fmt_time(row.fetched_at)}")


if __name__ == "__main__":
    cli()