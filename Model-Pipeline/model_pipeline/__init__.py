"""Model pipeline: train, validate, check bias, select and register the
model that predicts whether a clinical trial will stop early.

The stages reuse the data pipeline's code (src/ and Data-Pipeline/scripts/),
so the repository root and Data-Pipeline/ are put on the import path here.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO, REPO / "Data-Pipeline"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
