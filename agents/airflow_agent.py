"""
Airflow Agent
Reads the exact Airflow task log for a given run_id and extracts
structured error information — no rule-based RCA, only log parsing.

run(dag_id, task_id, run_id, execution_date) -> dict
"""

import os
import glob
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("AirflowAgent")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LOGS_PATH = Path(os.getenv("AIRFLOW_LOGS_PATH", str(PROJECT_ROOT / "logs")))


def parse_json_log(raw_content: str) -> str:
    """
    Airflow 2.7+ writes logs as newline-delimited JSON.
    Extract the 'event' field from each line so regex patterns
    can match against plain text instead of JSON structure.
    Falls back to raw content for plain-text log files.
    """
    lines = raw_content.strip().splitlines()
    extracted = []
    json_lines_found = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            event = obj.get("event", "")
            level = obj.get("level", "info").upper()
            extracted.append(f"[{level}] {event}")
            json_lines_found += 1
        except (json.JSONDecodeError, ValueError):
            extracted.append(line)

    if json_lines_found > 0:
        log.info("AirflowAgent: JSON log format detected — extracted %d event lines", json_lines_found)

    return "\n".join(extracted)


def get_log(dag_id: str, task_id: str, run_id: str) -> str:
    exact_pattern = str(
        LOGS_PATH / f"dag_id={dag_id}" / f"run_id={run_id}"
        / f"task_id={task_id}" / "attempt=*.log"
    )
    files = glob.glob(exact_pattern)

    if not files:
        fallback_pattern = str(
            LOGS_PATH / f"dag_id={dag_id}" / "run_id=*"
            / f"task_id={task_id}" / "attempt=*.log"
        )
        files = glob.glob(fallback_pattern)

    if not files:
        files = glob.glob(
            str(LOGS_PATH / "**" / f"{task_id}*.log"),
            recursive=True,
        )

    if not files:
        log.warning("No log file found for dag=%s task=%s run_id=%s", dag_id, task_id, run_id)
        return ""

    latest = max(files, key=os.path.getmtime)
    log.info("Reading log: %s", latest)

    with open(latest, "r", errors="replace") as f:
        raw_content = f.read()
    return parse_json_log(raw_content)


def extract_error(log_content: str) -> dict:
    result = {
        "error_type"    : "Unknown",
        "error_message" : "Unknown",
        "product_id"    : None,
        "traceback"     : "",
        "execution_time": None,
    }

    if not log_content:
        result["error_message"] = "No log content available"
        return result

    error_map = {
        "SLA MISS DETECTED"         : "SLAMiss",
        "SLA breach"                : "SLAMiss",
        "sla_missed=True"           : "SLAMiss",   
        "SLA MISSED"                : "SLAMiss",   
        "did not complete within"   : "SLAMiss",  
        "SLA threshold"             : "SLAMiss",
        "UniqueViolation"           : "UniqueViolation",
        "duplicate key value"       : "UniqueViolation",
        "unique constraint"         : "UniqueViolation",
        "OperationalError"          : "OperationalError",
        "could not connect"         : "OperationalError",
        "connection refused"        : "OperationalError",
        "FileNotFoundError"         : "FileNotFoundError",
        "No such file or directory" : "FileNotFoundError",
        "NullValueError"            : "NullValueError",
        "null value in column"      : "NullValueError",
        "DataError"                 : "DataError",
        "invalid input syntax"      : "DataError",
        "IntegrityError"            : "IntegrityError",
        "ForeignKeyViolation"       : "ForeignKeyViolation",
        "foreign key constraint"    : "ForeignKeyViolation",
        "DivisionByZero"            : "DivisionByZero",
        "division by zero"          : "DivisionByZero",
        "Intentional failure"       : "PipelineTestError",
        "Task failed with exception": "PipelineTestError",
    }

    for keyword, etype in error_map.items():
        if keyword.lower() in log_content.lower():
            result["error_type"] = etype
            break

    exception_patterns = [
        r'Exception:\s*(.+)',
        r'Caused by:\s*(.+)',
        r'Failure caused by\s*(.+)',
        r'ERROR.*?-\s*(.+)',
    ]
    for pattern in exception_patterns:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            result["error_message"] = match.group(0).strip()[:300]
            break
    if result["error_message"] == "Unknown":
        error_keywords = ["duplicate key", "violation", "failed", "exception", "traceback"]
        for line in log_content.splitlines():
            if any(k in line.lower() for k in error_keywords):
                cleaned = line.strip()
                if cleaned:
                    result["error_message"] = cleaned
                    break   

    for pattern in [
        r'Key \(product_id\)=\((\d+)\)',
        r'product_id\)=\((\d+)\)',
        r'product_id\s*[=:]\s*(\d+)',
        r'"product_id":\s*(\d+)',
    ]:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            result["product_id"] = match.group(1)
            break

    tb_start = log_content.find("Traceback")
    if tb_start != -1:
        result["traceback"] = log_content[tb_start: tb_start + 2000]

    ts_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', log_content)
    if ts_match:
        result["execution_time"] = ts_match.group(1)

    return result


def run(
    dag_id        : str = "retail_product_etl",
    task_id       : str = "load_product_csv",
    run_id        : str = "",
    execution_date: str = "",
) -> dict:
    log.info("AirflowAgent starting for dag=%s task=%s run_id=%s", dag_id, task_id, run_id)

    if run_id == "sla_miss":
        log.info("SLA miss run detected — synthesising evidence (no task log)")
        return {
            "agent"         : "airflow_agent",
            "status"        : "SUCCESS",
            "dag_id"        : dag_id,
            "task_id"       : task_id,
            "run_id"        : run_id,
            "execution_date": execution_date,
            "log_found"     : False,
            "error_type"    : "SLAMiss",
            "error_message" : (
                f"Task '{task_id}' in DAG '{dag_id}' exceeded its configured SLA "
                f"threshold. Task did not complete within the allowed time window."
            ),
            "product_id"    : None,
            "traceback"     : "",
            "execution_time": execution_date,
        }

    try:
        log_content = get_log(dag_id, task_id, run_id)
        error_info  = extract_error(log_content)

        log.info("Error Type: %s | Product ID: %s", error_info["error_type"], error_info["product_id"])

        return {
            "agent"         : "airflow_agent",
            "status"        : "SUCCESS",
            "dag_id"        : dag_id,
            "task_id"       : task_id,
            "run_id"        : run_id,
            "execution_date": execution_date,
            "log_found"     : bool(log_content),
            **error_info,
        }

    except Exception as e:
        log.error("AirflowAgent failed: %s", str(e))
        return {
            "agent"  : "airflow_agent",
            "status" : "ERROR",
            "message": str(e),
            "dag_id" : dag_id,
            "task_id": task_id,
            "run_id" : run_id,
        }


if __name__ == "__main__":
    result = run()
    print("\n--- Airflow Agent Output ---")
    for k, v in result.items():
        if k != "traceback":
            print(f"  {k:20s}: {v}")
