"""Fetch an AACT snapshot into <out-root>/<snapshot>/ as table files.

Two sources, both writing the same layout (one pipe-delimited <table>.txt
per table):

  --zip PATH_OR_URL   AACT monthly flat-file export (.zip). Downloading it
                      from the AACT site needs a free AACT account; pass the
                      saved file or its URL.
  --db                AACT cloud PostgreSQL database. Reads AACT_USER and
                      AACT_PASSWORD from the environment (same account).

Only the tables in REQUIRED_TABLES are kept. A manifest.json records the
source, snapshot name and file sizes, so each snapshot can be versioned
with DVC.

    python -m src.ingestion.fetch_aact --zip 20261001_export_ctgov.zip
    python -m src.ingestion.fetch_aact --db
"""

import argparse
import datetime as dt
import json
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests

from src.error_handling import (ConfigError, DataUnavailableError,
                                DataValidationError)
from src.logging_utils import run_main

logger = logging.getLogger(__name__)

REQUIRED_TABLES = [
    "studies", "brief_summaries", "detailed_descriptions", "eligibilities",
    "design_outcomes", "designs", "calculated_values", "sponsors",
    "countries", "interventions", "conditions", "browse_conditions",
]
AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DBNAME = "aact"


def download(url, dest, retries=3, backoff_seconds=10, chunk_bytes=1 << 20):
    """Stream url to dest, retrying on network errors.

    Writes to a .part file first so a failed download leaves no partial
    file behind.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_bytes):
                        f.write(chunk)
            part.replace(dest)
            logger.info("downloaded %s (%d bytes)", url,
                        dest.stat().st_size)
            return dest
        except requests.RequestException as e:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise DataUnavailableError(
                    f"download failed after {retries} attempts: {url}: {e}"
                ) from e
            logger.warning("download attempt %d/%d failed: %s", attempt,
                           retries, e)
            time.sleep(backoff_seconds * attempt)


def extract_tables(zip_path, out_dir, tables=REQUIRED_TABLES):
    """Extract <table>.txt for each table from an AACT export zip.

    The table files may sit in a subfolder inside the zip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        z = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        raise DataValidationError(
            f"cannot open AACT export {zip_path}: {e}") from e
    with z:
        members = {Path(name).name: name for name in z.namelist()
                   if not name.endswith("/")}
        missing = [t for t in tables if f"{t}.txt" not in members]
        if missing:
            raise DataValidationError(
                f"{zip_path} is missing tables: {missing}")
        for t in tables:
            with z.open(members[f"{t}.txt"]) as src, \
                    open(out_dir / f"{t}.txt", "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("extracted %s", t)


def export_from_db(out_dir, tables=REQUIRED_TABLES, user=None,
                   password=None, host=AACT_HOST, port=AACT_PORT):
    """Export each table from the AACT database in the flat-file format."""
    user = user or os.environ.get("AACT_USER")
    password = password or os.environ.get("AACT_PASSWORD")
    if not user or not password:
        raise ConfigError(
            "set AACT_USER and AACT_PASSWORD to read from the AACT database")
    try:
        import psycopg2  # only needed for this source
    except ImportError as e:
        raise ConfigError(
            "psycopg2 is not installed; run pip install -r requirements.txt"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        conn = psycopg2.connect(host=host, port=port, dbname=AACT_DBNAME,
                                user=user, password=password)
    except psycopg2.OperationalError as e:
        raise DataUnavailableError(
            f"could not connect to the AACT database at {host}: {e}") from e
    try:
        with conn, conn.cursor() as cur:
            for t in tables:  # names come from REQUIRED_TABLES only
                sql = (f"COPY (SELECT * FROM ctgov.{t}) TO STDOUT "
                       "WITH (FORMAT csv, HEADER, DELIMITER '|')")
                path = out_dir / f"{t}.txt"
                with open(path, "w", encoding="utf-8", newline="") as f:
                    cur.copy_expert(sql, f)
                logger.info("exported %s (%d bytes)", t,
                            path.stat().st_size)
    finally:
        conn.close()


def validate(out_dir, tables=REQUIRED_TABLES):
    """Check each table file has an nct_id column and at least one row.

    Returns file sizes in bytes.
    """
    out_dir = Path(out_dir)
    sizes = {}
    for t in tables:
        path = out_dir / f"{t}.txt"
        if not path.exists():
            raise DataUnavailableError(f"missing table file: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                header = f.readline().rstrip("\r\n").split("|")
                first_row = f.readline()
        except UnicodeDecodeError as e:
            raise DataValidationError(
                f"{path} is not valid UTF-8 (corrupt file?): {e}") from e
        if "nct_id" not in header:
            raise DataValidationError(f"{path} has no nct_id column")
        if not first_row.strip():
            raise DataValidationError(f"{path} has no data rows")
        sizes[t] = path.stat().st_size
    logger.info("validated %d tables in %s", len(sizes), out_dir)
    return sizes


def snapshot_name(source):
    """YYYYMMDD from a name like 20261001_export_ctgov.zip, else today."""
    match = re.match(r"(\d{8})", Path(str(source)).name)
    return match.group(1) if match else dt.date.today().strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip",
                        help="path or URL of an AACT flat-file export zip")
    source.add_argument("--db", action="store_true",
                        help="export from the AACT cloud database")
    parser.add_argument("--out-root", default="data/raw/aact")
    parser.add_argument("--snapshot",
                        help="snapshot folder name (default: date from the "
                             "zip name, or today)")
    args = parser.parse_args()

    if args.snapshot:
        snapshot = args.snapshot
    elif args.zip:
        snapshot = snapshot_name(args.zip)
    else:
        snapshot = dt.date.today().strftime("%Y%m%d")
    out_dir = Path(args.out_root) / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fetching AACT snapshot %s into %s", snapshot, out_dir)

    if args.zip:
        zip_path = args.zip
        if re.match(r"https?://", args.zip):
            name = Path(args.zip.split("?")[0]).name
            zip_path = download(args.zip, out_dir / name)
        extract_tables(zip_path, out_dir)
        if zip_path != args.zip:
            Path(zip_path).unlink()
    else:
        export_from_db(out_dir)

    sizes = validate(out_dir)
    manifest = {
        "source": "db" if args.db else args.zip,
        "snapshot": snapshot,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "tables": sizes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", out_dir / "manifest.json")


if __name__ == "__main__":
    run_main(main, "fetch_aact")
