"""Load one row per trial with the raw columns needed for slicing.

Three sources, chosen in the config (input.source):
  aact     - query the AACT database with trials_query.sql
  file     - read a CSV or Parquet file and rename its columns using
             input.columns, so a different file layout only needs a config
             change
  snapshot - build the same columns from an AACT flat-file snapshot
             (input.snapshot_dir), as versioned by the data pipeline
"""

import logging
import os
from pathlib import Path

import pandas as pd

from src.error_handling import (ConfigError, DataUnavailableError,
                                 DataValidationError)

logger = logging.getLogger(__name__)

AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DBNAME = "aact"
QUERY_PATH = Path(__file__).with_name("trials_query.sql")

# Columns every source must provide.
REQUIRED_COLUMNS = ["nct_id", "overall_status", "phase", "study_type",
                    "start_date", "sponsor_class", "gender", "child", "adult",
                    "older_adult", "countries", "therapeutic_areas"]

# MeSH top-level disease categories, as in browse_conditions.downcase_mesh_term
# (same list as src/features/structured_features.py).
THERAPEUTIC_AREAS = [
    "infections", "neoplasms", "musculoskeletal diseases",
    "digestive system diseases", "stomatognathic diseases",
    "respiratory tract diseases", "otorhinolaryngologic diseases",
    "nervous system diseases", "eye diseases", "urogenital diseases",
    "cardiovascular diseases", "hemic and lymphatic diseases",
    "congenital, hereditary, and neonatal diseases and abnormalities",
    "skin and connective tissue diseases",
    "nutritional and metabolic diseases", "endocrine system diseases",
    "immune system diseases", "disorders of environmental origin",
    "occupational diseases", "chemically-induced disorders",
    "wounds and injuries", "pathological conditions, signs and symptoms",
    "mental disorders",
]


def query_aact(limit=None):
    """Run trials_query.sql on the AACT database and return a DataFrame."""
    user = os.environ.get("AACT_USER")
    password = os.environ.get("AACT_PASSWORD")
    if not user or not password:
        raise ConfigError(
            "set AACT_USER and AACT_PASSWORD to read from the AACT database")

    import psycopg2  # only needed for this source

    sql = QUERY_PATH.read_text()
    params = {"areas": THERAPEUTIC_AREAS}
    if limit:
        sql += "\nLIMIT %(limit)s"
        params["limit"] = int(limit)

    logger.info("querying AACT at %s", AACT_HOST)
    try:
        conn = psycopg2.connect(host=AACT_HOST, port=AACT_PORT,
                                dbname=AACT_DBNAME, user=user,
                                password=password, connect_timeout=30)
    except psycopg2.OperationalError as e:
        raise DataUnavailableError(f"could not connect to AACT: {e}") from e
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        conn.close()


def read_file(path, columns):
    """Read a CSV or Parquet file and rename its columns to our names."""
    path = Path(path)
    if not path.exists():
        raise DataUnavailableError(f"input file not found: {path}")
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            # keep AACT's literal "NA" (phase not applicable) as a value
            df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except (ValueError, OSError, UnicodeDecodeError) as e:
        raise DataValidationError(f"could not read {path}: {e}") from e
    # Config maps our name -> file name; rename needs file name -> our name.
    return df.rename(columns={theirs: ours for ours, theirs in columns.items()})


def _read_table(export_dir, table, cols):
    return pd.read_csv(Path(export_dir) / f"{table}.txt", sep="|",
                       usecols=cols, dtype=str, keep_default_na=False,
                       na_values=[""])


def read_snapshot(export_dir):
    """Build the same columns as trials_query.sql from an AACT flat-file
    snapshot (the 12 pipe-delimited tables the data pipeline versions with
    DVC), so bias detection runs on exactly the data the pipeline uses and
    needs no database account."""
    export_dir = Path(export_dir)
    if not (export_dir / "studies.txt").exists():
        raise DataUnavailableError(
            f"AACT snapshot not found: {export_dir}")
    studies = _read_table(export_dir, "studies",
                          ["nct_id", "overall_status", "phase", "study_type",
                           "start_date"])
    sponsors = _read_table(export_dir, "sponsors",
                           ["nct_id", "agency_class", "lead_or_collaborator"])
    lead = (sponsors[sponsors.lead_or_collaborator == "lead"]
            .drop_duplicates("nct_id")
            .rename(columns={"agency_class": "sponsor_class"})
            [["nct_id", "sponsor_class"]])
    elig = _read_table(export_dir, "eligibilities",
                       ["nct_id", "gender", "child", "adult", "older_adult"]
                       ).drop_duplicates("nct_id")
    countries = _read_table(export_dir, "countries",
                            ["nct_id", "name"]).dropna()
    countries = (countries.drop_duplicates().sort_values("name")
                 .groupby("nct_id").name.agg("|".join).rename("countries"))
    areas = _read_table(export_dir, "browse_conditions",
                        ["nct_id", "downcase_mesh_term"])
    areas = areas[areas.downcase_mesh_term.isin(THERAPEUTIC_AREAS)]
    areas = (areas.drop_duplicates().sort_values("downcase_mesh_term")
             .groupby("nct_id").downcase_mesh_term.agg("|".join)
             .rename("therapeutic_areas"))
    return (studies.merge(lead, on="nct_id", how="left")
            .merge(elig, on="nct_id", how="left")
            .merge(countries, on="nct_id", how="left")
            .merge(areas, on="nct_id", how="left"))


def load_trials(input_config):
    """Load trials from the source named in the config."""
    source = input_config["source"]
    if source == "aact":
        df = query_aact(input_config.get("limit"))
    elif source == "file":
        df = read_file(input_config["file_path"],
                       input_config.get("columns", {}))
    elif source == "snapshot":
        df = read_snapshot(input_config["snapshot_dir"])
    else:
        raise ValueError(f"unknown input source: {source!r} "
                         "(use 'aact', 'file' or 'snapshot')")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"input is missing columns: {missing}")

    n_rows = len(df)
    df = df.drop_duplicates("nct_id")
    if len(df) < n_rows:
        logger.warning("dropped %d duplicate trial rows", n_rows - len(df))
    logger.info("loaded %d trials from %s", len(df), source)
    return df
