"""Dataclasses passed between the api, storage, and analysis layers."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Observation:
    date: str
    value: float | None  # None where FRED has no data


@dataclass
class SeriesInfo:
    series_id: str
    title: str
    units: str
    frequency: str
    last_updated: str  # FRED's timestamp for its last revision
    fetched_at: datetime  # when econterm downloaded it (UTC)


@dataclass
class OLSResult:
    coefficients: np.ndarray
    std_errors: np.ndarray
    condition_number: float
    tvalues: np.ndarray
    pvalues: np.ndarray
    r_squared: float
    adj_r_squared: float
    n: int
    residuals: np.ndarray