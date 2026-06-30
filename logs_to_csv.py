"""
logs_to_csv.py
--------------
Parses Airflow task log text and exports a structured CSV.

Usage:
    python3 logs_to_csv.py <log_file_path> <output_csv_path>

Example:
    python3 logs_to_csv.py \
        "/home/mpallavi/RETAIL_AI_RCA/logs/dag_id=daily_product_data_etl_job/run_id=manual__2026-06-08T09:17:20.983676+00:00/task_id=load_product_csv_to_db_1m/attempt=1.log" \
        "/home/mpallavi/RETAIL_AI_RCA/data/etl_log_report.csv"
"""

import re
import csv
import sys
import os
from pathlib import Path
from datetime import datetime


# -- Parse log file path for metadata -----------------------------------------
def extract_metadata_from_path(log_path: str) -> dict:
    """Pull dag_id, run_id, task_id, attempt from the log file path."""
    meta = {
        "dag_id"   : "",
        "run_id"   : "",
        "task_id"  : "",
        "attempt"  : "",
    }

    dag_match   = re.search(r"dag_id=([^/]+)",  log_path)
    run_match   = re.search(r"run_id=([^/]+)",  log_path)
    task_match  = re.search(r"task_id=([^/]+)", log_path)
    att_match   = re.search(r"attempt=(\d+)",   log_path)

    if dag_match : meta["dag_id"]  = dag_match.group(1)
    if run_match : meta["run_id"]  = run_match.group(1)
    if task_match: meta["task_id"] = task_match.group(1)
    if att_match : meta["attempt"] = att_match.group(1)

    return meta


# -- Parse individual log lines ------------------------------------------------
def parse_log_lines(lines: list) -> list:
    """
    Parse each log line into structured fields.
    Expected format:
        [2026-06-08 14:47:22] `INFO` - message source=X loc=Y
    """
    # matches: [timestamp] `LEVEL` - message
    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+`([A-Z]+)`\s+-\s+(.*)"
    )

    records = []
    traceback_buffer = []
    current_record   = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            # Save previous record + its traceback
            if current_record:
                current_record["traceback"] = "\n".join(traceback_buffer).strip()
                records.append(current_record)
                traceback_buffer = []

            timestamp_str, level, message = match.groups()

            # Extract source and loc if present
            source = ""
            loc    = ""
            src_match = re.search(r"source=([\w.]+)", message)
            loc_match = re.search(r"loc=([\S]+)",     message)
            if src_match: source = src_match.group(1)
            if loc_match: loc    = loc_match.group(1)

            # Clean message — remove source= and loc= suffixes
            clean_msg = re.sub(r"\s+source=\S+", "", message)
            clean_msg = re.sub(r"\s+loc=\S+",    "", clean_msg).strip()

            current_record = {
                "timestamp"    : timestamp_str,
                "level"        : level,
                "message"      : clean_msg,
                "source"       : source,
                "loc"          : loc,
                "traceback"    : "",
            }

        else:
            # Non-matching lines are traceback / continuation lines
            if current_record:
                traceback_buffer.append(line)

    # Flush last record
    if current_record:
        current_record["traceback"] = "\n".join(traceback_buffer).strip()
        records.append(current_record)

    return records


# -- Classify each log record --------------------------------------------------
def classify(record: dict) -> str:
    """Tag each record with an event type for easy RAG filtering."""
    msg = record["message"].lower()
    tb  = record["traceback"].lower()
    lvl = record["level"]

    if "sla miss detected"    in msg: return "SLA_MISS"
    if "failure detected"     in msg: return "FAILURE_TRIGGER"
    if "autorca"              in msg: return "AUTORCA"
    if "task failed"          in msg: return "TASK_ERROR"
    if "uniqueviolation"      in tb : return "DB_DUPLICATE_KEY"
    if "not all arguments"    in tb : return "DB_TYPE_ERROR"
    if "task instance is in running" in msg: return "TASK_START"
    if "task instance in failure"    in msg: return "TASK_END_FAILURE"
    if "inserted"             in msg: return "INSERT_SUCCESS"
    if "read"                 in msg and "rows" in msg: return "DATA_READ"
    if "exported"             in msg: return "EXPORT"
    if lvl == "ERROR"               : return "ERROR"
    if lvl == "WARNING"             : return "WARNING"
    return "INFO"


# -- Main ----------------------------------------------------------------------
def convert_log_to_csv(log_path: str, output_csv: str):
    log_path = log_path.strip()

    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Reading log : {log_path}")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    meta    = extract_metadata_from_path(log_path)
    records = parse_log_lines(lines)

    if not records:
        print("No structured log lines found. Check the log file format.")
        sys.exit(1)

    # Add metadata + classification to every record
    for r in records:
        r["dag_id"]    = meta["dag_id"]
        r["run_id"]    = meta["run_id"]
        r["task_id"]   = meta["task_id"]
        r["attempt"]   = meta["attempt"]
        r["event_type"]= classify(r)

    # Column order
    fieldnames = [
        "dag_id", "run_id", "task_id", "attempt",
        "timestamp", "level", "event_type",
        "message", "source", "loc", "traceback",
    ]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Done! {len(records)} rows written to: {output_csv}")


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 logs_to_csv.py <log_file_path> <output_csv_path>")
        sys.exit(1)

    convert_log_to_csv(sys.argv[1], sys.argv[2])
