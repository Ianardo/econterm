import sys
from pathlib import Path
import respx
import asyncio
import httpx

external_folder = str(Path(__file__).resolve().parent.parent / "src/econterm")
sys.path.append(external_folder)

import api
