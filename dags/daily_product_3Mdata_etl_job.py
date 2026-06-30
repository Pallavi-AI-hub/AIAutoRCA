from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
import threading
import time as _time
import sys
from pathlib import Path
 
logger = logging.getLogger(__name__)
 
PROJECT_ROOT = Path("/home/mpallavi/RETAIL_AI_RCA")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))
 
from workflows.rca_workflow import run_rca_workflow
 
PRODUCT_FILE  = "/home/mpallavi/RETAIL_AI_RCA/data/retail_product_data_3M.csv"
EXPORT_FILE   = "/home/mpallavi/RETAIL_AI_RCA/data/product_master_3M_export.csv"
DB_CONFIG     = {"host": "localhost", "database": "retail_db", "user": "airflow_user", "password": "airflow123"}
SLA_SECONDS   = 90
_rca_launched = False
 
def get_conn():
    return psycopg2.connect(**DB_CONFIG)
 
def _launch_rca(dag_id, task_id, run_id="", execution_date="", performance_data=None):
    def _run():
        try:
            run_rca_workflow(
                dag_id          =dag_id,
                task_id         =task_id,
                run_id          =run_id,
                execution_date  =execution_date,
                performance_data=performance_data or {},
            )
        except Exception as e:
            logger.error("AutoRCA thread failed: %s", e, exc_info=True)
    thread = threading.Thread(target=_run, daemon=True, name="AutoRCA")
    thread.start()
    thread.join(timeout=300)
    if thread.is_alive():
        logger.warning("AutoRCA thread still running after 300s — releasing worker")
 
def on_failure_callback(context: dict):
    global _rca_launched
    if _rca_launched:
        logger.warning("RCA already launched — skipping duplicate | dag=%s task=%s",
                       context["dag"].dag_id, context["task_instance"].task_id)
        return
    _rca_launched  = True
    dag_id         = context["dag"].dag_id
    task_id        = context["task_instance"].task_id
    run_id         = context.get("run_id", "")
    execution_date = str(context.get("execution_date", ""))
    logger.error("FAILURE DETECTED — launching AutoRCA | dag=%s task=%s", dag_id, task_id)
    _launch_rca(dag_id, task_id, run_id, execution_date)
 
def load_product_csv():
    global _rca_launched
    _rca_launched = False
 
    DAG_ID        = "Daily_product_3Mdata_ETL_job"
    TASK_ID       = "load_3m_data"
    start         = _time.time()
    sla_triggered = False
    inserted      = 0
    chunk_size    = 10000
    total_rows    = 3000000
 
    logger.info("Starting 3M load — SLA threshold: %ds", SLA_SECONDS)
 
    conn = get_conn()
    cur  = conn.cursor()
 
    try:
        for chunk in pd.read_csv(PRODUCT_FILE, chunksize=chunk_size):
            elapsed      = _time.time() - start
            rows_per_sec = inserted / elapsed if elapsed > 0 else 0
            remaining    = total_rows - inserted
            eta_seconds  = (remaining / rows_per_sec) if rows_per_sec > 0 else 0
            pct_complete = round((inserted / total_rows) * 100, 1)
 
            if not sla_triggered and elapsed > SLA_SECONDS:
                sla_triggered = True
                _rca_launched = True
 
                performance_data = {
                    "dag_id"          : DAG_ID,
                    "task_id"         : TASK_ID,
                    "total_rows"      : total_rows,
                    "rows_inserted"   : inserted,
                    "pct_complete"    : pct_complete,
                    "elapsed_seconds" : round(elapsed, 1),
                    "rows_per_second" : round(rows_per_sec, 1),
                    "eta_seconds"     : round(eta_seconds, 1),
                    "sla_threshold"   : SLA_SECONDS,
                    "sla_breached_at" : round(elapsed, 1),
                    "chunk_size"      : chunk_size,
                    "insert_method"   : "execute_values",
                    "data_volume"     : "3M",
                }
 
                logger.warning(
                    "*** SLA MISS *** elapsed=%.1fs > threshold=%ds | "
                    "rows=%d/%d (%.1f%%) | rows/sec=%.1f | ETA=%.1fs | launching AutoRCA",
                    elapsed, SLA_SECONDS, inserted, total_rows,
                    pct_complete, rows_per_sec, eta_seconds
                )
                _launch_rca(DAG_ID, TASK_ID,
                            run_id="sla_miss",
                            execution_date=str(datetime.now()),
                            performance_data=performance_data)
 
            rows = [tuple(row) for _, row in chunk.iterrows()]
            execute_values(
                cur,
                """
                INSERT INTO product_details_master_3M (
                    product_id, product_name, category, brand, collection, color, size,
                    gender, supplier_id, supplier_name, cost_price, selling_price,
                    discount_pct, inventory_qty, reorder_level, rating, review_count,
                    status, created_date
                ) VALUES %s
                ON CONFLICT (product_id) DO NOTHING
                """,
                rows,
                page_size=chunk_size
            )
            conn.commit()
            inserted += len(rows)
            logger.info(
                "Progress: %d / %d rows (%.1f%%) | elapsed=%.1fs | rows/sec=%.1f | ETA=%.1fs | sla_missed=%s",
                inserted, total_rows, pct_complete,
                _time.time() - start, rows_per_sec, eta_seconds, sla_triggered
            )
 
        logger.info("3M load complete — %d rows | total elapsed=%.1fs",
                    inserted, _time.time() - start)
 
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
 
def export_product_csv():
    logger.info("Exporting to %s", EXPORT_FILE)
    conn = get_conn()
    df   = pd.read_sql("SELECT * FROM product_details_master_3M", conn)
    conn.close()
    df.to_csv(EXPORT_FILE, index=False)
    logger.info("Exported %d rows", len(df))
 
with DAG(
    dag_id="Daily_product_3Mdata_ETL_job",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "on_failure_callback": on_failure_callback},
) as dag:
    load_task   = PythonOperator(task_id="load_3m_data", python_callable=load_product_csv, on_failure_callback=on_failure_callback)
    export_task = PythonOperator(task_id="export_data",  python_callable=export_product_csv)
    load_task >> export_task
