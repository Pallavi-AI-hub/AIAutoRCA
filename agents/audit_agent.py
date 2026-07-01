"""
Audit Agent - Reads etl_audit table, filters by dag_id + task_id
"""
import os
import logging
import psycopg2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("AuditAgent")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "DB_HOST"),
        "database": os.getenv("DB_NAME", "DB_NAME"),
        "user": os.getenv("DB_USER", "DB_USER"),
        "password": os.getenv("DB_PASSWORD", "DB_PASSWORD"),
        "connect_timeout": 10,
    }

def get_conn():
    return psycopg2.connect(**get_db_config())

def run(dag_id=None, task_id=None):
    if not dag_id or not task_id:
        log.error("AuditAgent requires both dag_id and task_id parameters")
        return {"agent": "audit_agent", "status": "ERROR", "message": "Missing dag_id or task_id"}
    log.info("AuditAgent querying for dag=%s task=%s", dag_id, task_id)
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT audit_id, dag_id, task_id, run_id, execution_time,
                   status, records_processed, error_message
            FROM etl_audit
            WHERE dag_id=%s AND task_id=%s AND status='FAILED'
            ORDER BY audit_id DESC LIMIT 1
        """, (dag_id, task_id))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            log.warning("No FAILED record found for dag=%s task=%s", dag_id, task_id)
            return {"agent": "audit_agent", "status": "NO_FAILURE_FOUND", "dag_id": dag_id, "task_id": task_id}
        result = {
            "agent": "audit_agent", "status": "SUCCESS",
            "audit_id": row[0], "dag_id": row[1], "task_id": row[2], "run_id": row[3],
            "execution_time": str(row[4]), "audit_status": row[5],
            "records_processed": row[6], "error_message": row[7],
        }
        log.info("Found FAILED record audit_id=%s: %s", result["audit_id"], result["error_message"])
        return result
    except psycopg2.OperationalError as e:
        log.error("DB connection failed: %s", str(e))
        return {"agent": "audit_agent", "status": "ERROR", "message": f"DB connection failed: {str(e)}"}
    except Exception as e:
        log.error("AuditAgent failed: %s", str(e))
        return {"agent": "audit_agent", "status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    result = run()
    print("\n--- Audit Agent Output ---")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
