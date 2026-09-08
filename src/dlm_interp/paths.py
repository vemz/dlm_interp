import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TAG = os.environ.get("DLM_TAG", "")
CKPT_ROOT = ROOT               
RUNS = ROOT / ("runs" + TAG)             
RESULTS = ROOT / ("results" + TAG)       
DATA = ROOT / "data"

RUNS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)