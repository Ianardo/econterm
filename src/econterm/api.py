import httpx
import asyncio
import os
from random import uniform
from time import sleep
from functools import wraps
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from econterm.models import Observation, SeriesInfo
import time

def retry(retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (RateLimited, httpx.TimeoutException) as e:
                    attempt = i + 1
                    print(f"Attempt {attempt} out of {retries} failed for {func.__name__}: {e}")
                    exp = base_delay * (2 ** (attempt - 1))
                    delay = uniform(exp / 2, exp)
                    
                    if attempt < retries:
                        print(f"Retrying in {delay:.2f} second(s)...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"{func.__name__} failed. Exiting")
                        raise e
        return wrapper
    return decorator

class FredApiError(Exception):
    pass

class SeriesNotFound(FredApiError):
    def __init__(self, series_id):
        self.series_id = series_id
        super().__init__(f"Unable to locate series {series_id}.")

class RateLimited(FredApiError):
    def __init__(self, message="Rate limited by FRED API."):
        super().__init__(message)

@retry()
async def fetch_series(series_id, start, end, client, sem):
    out = []
    load_dotenv(find_dotenv())
    API_KEY = os.getenv("FRED_API_KEY")
    URL = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": API_KEY,
        "file_type": 'json',
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "limit": 100_000
    }
    async with sem:
        print(f"Fetching: {series_id} from {start} to {end}...")
        data = await client.get(URL, params=params) 
        
        match data.status_code:
            case 200:
                observations = data.json().get('observations')
                for observation in observations:
                    out.append(
                        Observation(observation.get("date"), float(observation.get("value")))
                    )
                print(f"Successfully fetched observations for {series_id}!")
            case 400 | 404:
                raise SeriesNotFound(series_id)
            case 429:
                raise RateLimited()
        
    return out

@retry()
async def fetch_series_info(series_id, client, sem):
    load_dotenv(find_dotenv())
    API_KEY = os.getenv("FRED_API_KEY")
    URL = "https://api.stlouisfed.org/fred/series"
    params = {
        "api_key": API_KEY,
        "file_type": 'json',
        "series_id": series_id
    }
    async with sem:
        print(f"Fetching metadata: {series_id}...")
        data = await client.get(URL, params=params) 
        
        match data.status_code:
            case 200:
                info = data.json()["seriess"][0]
                print(f"Successfully fetched metadata for {series_id}!")
                return SeriesInfo(
                    series_id=info["id"],
                    title=info["title"],
                    units=info["units"],
                    frequency=info["frequency"],
                    last_updated=info["last_updated"],
                )
            case 400 | 404:
                raise SeriesNotFound(series_id)
            case 429:
                raise RateLimited()

# async def main():
#     CONCURRENCY_LIMIT = 5
#     sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
#     series = ['GDP', 'UNRATE', 'CPIAUCSL']
#     async with httpx.AsyncClient() as client:
#         # tasks = [fetch_series(series_id, "2000-01-01", "2001-01-01", client, sem) for series_id in series]
#         tasks = [fetch_series_info(series_id, client, sem) for series_id in series]
#         results = await asyncio.gather(*tasks)
#         print(results)
    
# asyncio.run(main())