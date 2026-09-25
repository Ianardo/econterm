"""FRED API client: series metadata and observations."""

import asyncio
import logging
import os
from datetime import UTC, datetime
from functools import cache, wraps
from random import uniform

import httpx
from dotenv import find_dotenv, load_dotenv

from econterm.models import Observation, SeriesInfo

logger = logging.getLogger(__name__)

SERIES_URL = "https://api.stlouisfed.org/fred/series"
OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredApiError(Exception):
    pass


class MissingApiKey(FredApiError):
    pass


class SeriesNotFound(FredApiError):
    def __init__(self, series_id):
        self.series_id = series_id
        super().__init__(f"Unable to locate series {series_id}.")


class RateLimited(FredApiError):
    def __init__(self, message="Rate limited by FRED API."):
        super().__init__(message)


@cache
def get_api_key():
    """Load the FRED API key from the environment or a .env file. Cached after the first call."""
    load_dotenv(find_dotenv())
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise MissingApiKey(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set it as an "
            "environment variable or in a .env file in the project folder."
        )
    return api_key


def retry(retries=3, base_delay=1):
    """Retry on rate limits and timeouts, with exponential backoff and jitter."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (RateLimited, httpx.TimeoutException) as e:
                    attempt = i + 1
                    if attempt == retries:
                        raise
                    exp = base_delay * (2 ** (attempt - 1))
                    delay = uniform(exp / 2, exp)
                    logger.warning(
                        "%s failed (attempt %d of %d): %s. Retrying in %.1fs",
                        func.__name__, attempt, retries, e, delay,
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def _raise_for_status(response, series_id):
    """Map FRED error responses to exceptions. Does nothing for 200."""
    match response.status_code:
        case 200:
            return
        case 400 | 404:
            raise SeriesNotFound(series_id)
        case 429:
            raise RateLimited()
        case _:
            raise FredApiError(f"FRED returned HTTP {response.status_code} for {series_id}")


@retry()
async def fetch_series(series_id, client, sem):
    """Fetch a series' full observation history. Missing values ('.') become None."""
    params = {
        "api_key": get_api_key(),
        "file_type": "json",
        "series_id": series_id,
        "limit": 100_000,
    }
    async with sem:
        response = await client.get(OBSERVATIONS_URL, params=params)
    _raise_for_status(response, series_id)

    return [
        Observation(obs["date"], None if obs["value"] == "." else float(obs["value"]))
        for obs in response.json()["observations"]
    ]


@retry()
async def fetch_series_info(series_id, client, sem):
    """Fetch a series' metadata, stamped with the time it was fetched."""
    params = {"api_key": get_api_key(), "file_type": "json", "series_id": series_id}
    async with sem:
        response = await client.get(SERIES_URL, params=params)
    _raise_for_status(response, series_id)

    info = response.json()["seriess"][0]
    return SeriesInfo(
        series_id=info["id"],
        title=info["title"],
        units=info["units"],
        frequency=info["frequency"],
        last_updated=info["last_updated"],
        fetched_at=datetime.now(UTC),
    )


async def fetch_one(series_id, client, sem):
    """Fetch metadata and observations for one series concurrently."""
    return await asyncio.gather(
        fetch_series_info(series_id, client, sem),
        fetch_series(series_id, client, sem),
    )