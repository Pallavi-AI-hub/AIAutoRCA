"""
Postgres Agent - Investigates specific product_id in product_master
"""
import os
import logging
import psycopg2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("PostgresAgent")

def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "retail_db"),
        "user": os.getenv("DB_USER", "airflow_user"),
        "password": os.getenv("DB_PASSWORD", "airflow123"),
        "connect_timeout": 10,
    }

def get_conn():
    return psycopg2.connect(**get_db_config())

def run(product_id=None):
    log.info("PostgresAgent investigating product_id=%s", product_id)
    try:
        conn = get_conn()
        cur  = conn.cursor()
        result = {
            "agent": "postgres_agent", "status": "SUCCESS",
            "product_id": product_id, "exists": False,
            "product_name": None, "brand": None, "category": None,
            "supplier_id": None, "cost_price": None, "selling_price": None,
            "duplicate_detected": False, "total_rows": 0,
        }
        if product_id:
            cur.execute("""
                SELECT product_id, product_name, brand, category,
                       supplier_id, cost_price, selling_price
                FROM product_master WHERE product_id=%s
            """, (int(product_id),))
            row = cur.fetchone()
            if row:
                result.update({"exists": True, "product_name": row[1], "brand": row[2],
                                "category": row[3], "supplier_id": row[4],
                                "cost_price": float(row[5]) if row[5] else None,
                                "selling_price": float(row[6]) if row[6] else None})
                log.info("Product found: %s (%s)", row[1], row[2])
            cur.execute("SELECT COUNT(*) FROM product_master WHERE product_id=%s", (int(product_id),))
            if cur.fetchone()[0] > 1:
                result["duplicate_detected"] = True
                log.warning("Duplicate detected for product_id=%s", product_id)
        cur.execute("SELECT COUNT(*) FROM product_master")
        result["total_rows"] = cur.fetchone()[0]
        cur.close()
        conn.close()
        return result
    except psycopg2.OperationalError as e:
        return {"agent": "postgres_agent", "status": "ERROR", "message": f"DB connection failed: {str(e)}"}
    except Exception as e:
        log.error("PostgresAgent failed: %s", str(e))
        return {"agent": "postgres_agent", "status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    result = run(product_id=1001)
    print("\n--- Postgres Agent Output ---")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
