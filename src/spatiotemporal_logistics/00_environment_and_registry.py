#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
import numpy as np
import sklearn
import statsmodels
import openpyxl
from pathlib import Path

ROOT = Path.cwd()
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

versions = {
    "python": sys.version,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "statsmodels": statsmodels.__version__,
    "openpyxl": openpyxl.__version__
}

with open(REPORTS_DIR / "00_environment_registry.json", "w", encoding="utf-8") as f:
    json.dump(versions, f, indent=2)

print("[PASS 00] AMBIENTE REGISTRADO SEM MODIFICAR BIBLIOTECAS DAS SESSÕES ANTERIORES.")
