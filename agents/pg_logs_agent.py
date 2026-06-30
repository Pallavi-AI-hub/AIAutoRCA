"""
PG Logs Agent
- Live PostgreSQL diagnostic snapshot — no log file parsing, no config changes
- Queries pg_stat_activity for slow/active queries during the incident window
- Queries pg_locks for lock contention (waiting locks, blocking PIDs)
- Queries pg_stat_database for checkpoint/IO pressure signals
"""

import os
import logging
import psycopg2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("PGLogsAgent")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

SLOW_QUERY_SECONDS = int(os.getenv("PG_LOGS_SLOW_QUERY_SEC", "1"))


def get_db_config() -> dict:
    return {
        "host"           : os.getenv("DB_HOST",     "localhost"),
        "database"       : os.getenv("DB_NAME",     "retail_db"),
        "user"           : os.getenv("DB_USER",     "airflow_user"),
        "password"       : os.getenv("DB_PASSWORD", "airflow123"),
        "connect_timeout": 10,
    }


def get_conn():
    return psycopg2.connect(**get_db_config())


def collect_active_queries(cur) -> list:
    cur.execute("""
        SELECT
            pid,
            usename,
            state,
            wait_event_type,
            wait_event,
            EXTRACT(EPOCH FROM (now() - query_start))::numeric(10,2) AS query_duration_sec,
            EXTRACT(EPOCH FROM (now() - xact_start))::numeric(10,2) AS txn_duration_sec,
            left(query, 200) AS query_excerpt
        FROM pg_stat_activity
        WHERE pid <> pg_backend_pid()
          AND state IS NOT NULL
          AND query IS NOT NULL
          AND query <> ''
        ORDER BY query_start ASC NULLS LAST;
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for r in rows:
        dur = r.get("query_duration_sec") or 0
        r["is_slow"] = float(dur) >= SLOW_QUERY_SECONDS
    return rows


def collect_lock_waits(cur) -> list:
    cur.execute("""
        SELECT
            blocked.pid           AS blocked_pid,
            blocked_act.usename   AS blocked_user,
            blocked_act.query     AS blocked_query,
            blocking.pid          AS blocking_pid,
            blocking_act.usename  AS blocking_user,
            blocking_act.query    AS blocking_query,
            blocked.locktype,
            blocked.mode          AS blocked_mode
        FROM pg_locks blocked
        JOIN pg_stat_activity blocked_act
            ON blocked_act.pid = blocked.pid
        JOIN pg_locks blocking
            ON blocking.locktype = blocked.locktype
           AND blocking.database IS NOT DISTINCT FROM blocked.database
           AND blocking.relation IS NOT DISTINCT FROM blocked.relation
           AND blocking.pid <> blocked.pid
           AND blocking.granted = true
        JOIN pg_stat_activity blocking_act
            ON blocking_act.pid = blocking.pid
        WHERE NOT blocked.granted;
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for r in rows:
        if r.get("blocked_query"):
            r["blocked_query"] = r["blocked_query"][:200]
        if r.get("blocking_query"):
            r["blocking_query"] = r["blocking_query"][:200]
    return rows


def collect_lock_summary(cur) -> dict:
    cur.execute("""
        SELECT
            locktype,
            granted,
            count(*) AS cnt
        FROM pg_locks
        GROUP BY locktype, granted
        ORDER BY locktype;
    """)
    rows = cur.fetchall()
    waiting_total = sum(r[2] for r in rows if r[1] is False)
    granted_total = sum(r[2] for r in rows if r[1] is True)
    return {
        "waiting_locks_total": waiting_total,
        "granted_locks_total": granted_total,
        "by_type": [{"locktype": r[0], "granted": r[1], "count": r[2]} for r in rows],
    }


def collect_database_stats(cur, dbname: str) -> dict:
    cur.execute("""
        SELECT
            numbackends, xact_commit, xact_rollback,
            blks_read, blks_hit, temp_files, temp_bytes,
            deadlocks, conflicts, checksum_failures
        FROM pg_stat_database
        WHERE datname = %s;
    """, (dbname,))
    row = cur.fetchone()
    if not row:
        return {}
    cols = ["numbackends", "xact_commit", "xact_rollback", "blks_read", "blks_hit",
            "temp_files", "temp_bytes", "deadlocks", "conflicts", "checksum_failures"]
    stats = dict(zip(cols, row))
    blks_read = stats.get("blks_read") or 0
    blks_hit  = stats.get("blks_hit") or 0
    total     = blks_read + blks_hit
    stats["buffer_hit_ratio_pct"] = round((blks_hit / total) * 100, 2) if total > 0 else None
    return stats


def assess_health(active_queries, lock_summary, db_stats) -> dict:
    slow_count    = sum(1 for q in active_queries if q.get("is_slow"))
    waiting_locks = lock_summary.get("waiting_locks_total", 0)
    deadlocks     = db_stats.get("deadlocks", 0) or 0
    temp_files    = db_stats.get("temp_files", 0) or 0
    buffer_hit    = db_stats.get("buffer_hit_ratio_pct")

    issues = []
    if slow_count > 0:
        issues.append(f"{slow_count} slow/active query(s) >= {SLOW_QUERY_SECONDS}s")
    if waiting_locks > 0:
        issues.append(f"{waiting_locks} waiting lock(s) — possible contention")
    if deadlocks > 0:
        issues.append(f"{deadlocks} deadlock(s) recorded since stats reset")
    if temp_files > 0:
        issues.append(f"{temp_files} temp file(s) used — possible work_mem pressure")
    if buffer_hit is not None and buffer_hit < 90:
        issues.append(f"Buffer hit ratio low at {buffer_hit}% — possible insufficient shared_buffers")

    if waiting_locks > 0 or deadlocks > 0:
        health = "CRITICAL"
    elif slow_count > 0 or temp_files > 0 or (buffer_hit is not None and buffer_hit < 90):
        health = "WARNING"
    else:
        health = "HEALTHY"

    return {"health": health, "issues": issues}


def run(dag_id: str = "", task_id: str = "") -> dict:
    log.info("PGLogsAgent starting | dag=%s task=%s", dag_id, task_id)
    try:
        conn = get_conn()
        cur  = conn.cursor()
        dbname = get_db_config()["database"]

        active_queries = collect_active_queries(cur)
        lock_waits     = collect_lock_waits(cur)
        lock_summary   = collect_lock_summary(cur)
        db_stats       = collect_database_stats(cur, dbname)

        cur.close()
        conn.close()

        health_info = assess_health(active_queries, lock_summary, db_stats)

        log.info(
            "PGLogsAgent done | health=%s | slow_queries=%d | waiting_locks=%d | deadlocks=%s",
            health_info["health"],
            sum(1 for q in active_queries if q.get("is_slow")),
            lock_summary.get("waiting_locks_total", 0),
            db_stats.get("deadlocks"),
        )

        return {
            "agent"            : "pg_logs_agent",
            "status"           : "SUCCESS",
            "snapshot_time"    : datetime.now().isoformat(),
            "active_queries"   : active_queries,
            "slow_query_count" : sum(1 for q in active_queries if q.get("is_slow")),
            "lock_waits"       : lock_waits,
            "lock_summary"     : lock_summary,
            "database_stats"   : db_stats,
            "health"           : health_info["health"],
            "issues"           : health_info["issues"],
        }

    except Exception as e:
        log.error("PGLogsAgent failed: %s", str(e), exc_info=True)
        return {
            "agent"  : "pg_logs_agent",
            "status" : "ERROR",
            "message": str(e),
            "health" : "UNKNOWN",
            "issues" : [],
        }


if __name__ == "__main__":
    result = run(dag_id="Daily_product_1Mdata_ETL_job", task_id="load_1m_data")
    print("\n--- PG Logs Agent Output ---")
    for k, v in result.items():
        if k in ("active_queries", "lock_waits"):
            print(f"  {k:20s}: {len(v)} item(s)")
            for item in v[:3]:
                print(f"      {item}")
        else:
            print(f"  {k:20s}: {v}")
