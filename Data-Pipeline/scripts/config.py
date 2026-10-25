"""Paths and parameters shared by the pipeline steps.

Every path is relative to the Data-Pipeline folder, so the steps work from
any working directory, inside the Docker image and on a laptop.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
# The ingestion, feature, bias and error-handling code lives in src/ at the
# repository root (the Docker image keeps the same layout).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BIAS_CONFIG = REPO_ROOT / "configs" / "bias_config.yaml"

PARAMS_FILE = ROOT / "params.yaml"
DVC_FILE = ROOT / "dvc.yaml"
LOCK_FILE = ROOT / "dvc.lock"
ENV_FILE = ROOT / ".env"

RAW_DIR = ROOT / "data" / "raw" / "aact_snapshot"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "structured_features.csv"
FEATURES_META = PROCESSED_DIR / "structured_features.meta.json"
TRIAL_SLICES = PROCESSED_DIR / "trial_slices.csv"

SCHEMA_DIR = ROOT / "schema"
GENERATED_DIR = SCHEMA_DIR / "generated"
GENERATED_SUITE = GENERATED_DIR / "expectation_suite.json"
BASELINE_DIR = SCHEMA_DIR / "baseline"
BASELINE_SUITE = BASELINE_DIR / "expectation_suite.json"
BASELINE_STATS = BASELINE_DIR / "statistics.json"
BASELINE_INFO = BASELINE_DIR / "frozen_from.json"

STATS_DIR = ROOT / "reports" / "statistics"
STATS_JSON = STATS_DIR / "statistics.json"
STATS_MD = STATS_DIR / "statistics.md"
VALIDATION_DIR = ROOT / "reports" / "validation"
VALIDATION_JSON = VALIDATION_DIR / "validation.json"
VALIDATION_SUMMARY = VALIDATION_DIR / "summary.json"
VALIDATION_MD = VALIDATION_DIR / "validation.md"

BIAS_DIR = ROOT / "reports" / "bias"
ANOMALY_DIR = ROOT / "reports" / "anomalies"
ANOMALY_JSON = ANOMALY_DIR / "anomalies.json"
ANOMALY_MD = ANOMALY_DIR / "anomalies.md"
ANOMALY_SUMMARY = ANOMALY_DIR / "summary.json"

LOG_DIR = ROOT / "logs"
RUNS_DIR = LOG_DIR / "runs"

# AACT tables the project uses (same list as src/ingestion/fetch_aact.py).
REQUIRED_TABLES = [
    "studies", "brief_summaries", "detailed_descriptions", "eligibilities",
    "design_outcomes", "designs", "calculated_values", "sponsors",
    "countries", "interventions", "conditions", "browse_conditions",
]


def load_params(path=PARAMS_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_file(path=ENV_FILE):
    """Read KEY=VALUE lines from .env without overriding variables already set.

    Returns the values found, so callers can merge them with os.environ.
    """
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
