import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_pipeline  # noqa: E402,F401  (puts the repo on the path)
