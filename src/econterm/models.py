from dataclasses import dataclass
import numpy as np

@dataclass
class Observation:
    date: str
    value: float
    
@dataclass
class SeriesInfo:
    series_id: str
    title: str
    units: str
    frequency: str
    last_updated: str
    
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