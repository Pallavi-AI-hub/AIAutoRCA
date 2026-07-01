"""
CPU Agent - OS CPU/Memory metrics + Postgres activity health check
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import psutil
import psycopg2

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("CPUAgent")

TREND_FILE = Path(os.getenv("CPU_AGENT_TREND_FILE", "/tmp/cpu_agent_trend.jsonl"))
TREND_WINDOW = int(os.getenv("CPU_AGENT_TREND_WINDOW", 10))
SLOW_QUERY_SECONDS = int(os.getenv("CPU_AGENT_SLOW_QUERY_SEC", 30))


def get_db_config():
    return {
        "host": os.environ["DB_HOST"],
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "connect_timeout": 10,
    }


def get_conn():
    return psycopg2.connect(**get_db_config())


#--------- OS: CPU ----------
def collect_cpu():
    cpu_percent = psutil.cpu_percent(interval=1)
    per_core = psutil.cpu_percent(interval=1, percpu=True)
    times = psutil.cpu_times_percent(interval=1)
    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):
        load1, load5, load15 = None, None, None
    return {
        "cpu_percent": cpu_percent,
        "load_avg_1min": load1,
        "load_avg_5min": load5,
        "load_avg_15min": load15,
        "core_usage": per_core,
        "core_count": psutil.cpu_count(),
        "user_percent": times.user,
        "system_percent": times.system,
        "idle_percent": times.idle,
        "iowait_percent": getattr(times, "iowait", None),
    }


# ---------- OS: Memory ----------
def detect_oom():
    """ Detect if the system has experienced an Out Of Memory (OOM) event.
    Returns a dictionary with keys: "oom_detected", "oom_count", "oom_source" """
    oom_count = 0
    source = None
    try:
        cg_path = Path("/sys/fs/cgroup/memory.events")
        if cg_path.exists():
            for line in cg_path.read_text().splitlines():
                if line.startswith("oom_kill"):
                    oom_count = int(line.split()[1])
                    source = "cgroup_v2"
    except Exception:
        pass
    if source is None:
        try:
            out = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            lines = [l for l in out.stdout.splitlines() if "Out of memory" in l or "oom-killer" in l]
            oom_count = len(lines)
            source = "dmesg"
        except Exception as e:
            source = f"unavailable ({e})"
    return {"oom_detected": oom_count > 0, "oom_count": oom_count, "oom_source": source}


def collect_memory():
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    oom = detect_oom()
    return {
        "ram_total": vm.total,
        "ram_used": vm.used,
        "ram_used_percent": vm.percent,
        "ram_free": vm.available,
        "swap_used": swap.used,
        "swap_percent": swap.percent,
        "cache": getattr(vm, "cached", None),
        "buffer": getattr(vm, "buffers", None),
        **oom,
    }

#--------- Memory trend ----------
def update_trend(ram_used_percent):
    """ Update the trend file with the latest RAM usage percentage and return the trend direction. """
    sample = {"ts": datetime.utcnow().isoformat(), "ram_used_percent": ram_used_percent}
    history = []
    if TREND_FILE.exists():
        try:
            history = [json.loads(l) for l in TREND_FILE.read_text().splitlines() if l.strip()]
        except Exception:
            history = []
    history.append(sample)
    history = history[-TREND_WINDOW:]
    TREND_FILE.write_text("\n".join(json.dumps(h) for h in history) + "\n")

    if len(history) < 2:
        return {"trend": "INSUFFICIENT_DATA", "delta_percent": 0.0}
    delta = history[-1]["ram_used_percent"] - history[0]["ram_used_percent"]
    direction = "RISING" if delta > 5 else "FALLING" if delta < -5 else "STABLE"
    return {"trend": direction, "delta_percent": round(delta, 2)}


# ---------- Postgres activity ----------
def collect_postgres_activity():
    """ Collect Postgres activity metrics: max connections, total connections, state counts, slow queries, waiting locks. """
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SHOW max_connections;")
        max_conn = int(cur.fetchone()[0])

        cur.execute("""
            SELECT state, count(*) FROM pg_stat_activity
            WHERE pid <> pg_backend_pid()
            GROUP BY state;
        """)
        state_counts = {row[0] or "unknown": row[1] for row in cur.fetchall()}

        cur.execute("""
            SELECT count(*) FROM pg_stat_activity
            WHERE state = 'active'
              AND now() - query_start > interval '%s seconds'
              AND pid <> pg_backend_pid();
        """, (SLOW_QUERY_SECONDS,))
        slow_queries = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM pg_locks WHERE NOT granted;")
        waiting_locks = cur.fetchone()[0]

        total_conn = sum(state_counts.values())
        cur.close()
        conn.close()

        return {
            "status": "SUCCESS",
            "max_connections": max_conn,
            "total_connections": total_conn,
            "connections_percent": round((total_conn / max_conn) * 100, 2) if max_conn else None,
            "state_counts": state_counts,
            "idle_in_transaction": state_counts.get("idle in transaction", 0),
            "slow_queries_over_threshold": slow_queries,
            "waiting_locks": waiting_locks,
        }
    except psycopg2.OperationalError as e:
        return {"status": "ERROR", "message": f"DB connection failed: {str(e)}"}
    except Exception as e:
        log.error("Postgres activity check failed: %s", str(e))
        return {"status": "ERROR", "message": str(e)}


# ---------- Health rules ----------
def cpu_health(cpu):
    core_count = cpu["core_count"] or 1
    load1 = cpu["load_avg_1min"] or 0
    iowait = cpu["iowait_percent"] or 0
    pct = cpu["cpu_percent"]
    if pct > 90 or load1 > core_count * 2 or iowait > 25:
        return "CRITICAL"
    if pct > 70 or load1 > core_count * 1.2 or iowait > 10:
        return "WARNING"
    return "HEALTHY"


def memory_health(mem, trend):
    if mem["ram_used_percent"] > 90 or mem["swap_percent"] > 40 or mem["oom_detected"]:
        return "CRITICAL"
    if mem["ram_used_percent"] > 85:
        return "WARNING"
    if mem["swap_percent"] > 10:
        return "WARNING"
    if trend["trend"] == "RISING":
        return "MONITOR"
    return "HEALTHY"


def postgres_health(pg):
    if pg["status"] != "SUCCESS":
        return "UNKNOWN"
    if (pg["connections_percent"] or 0) > 90 or pg["waiting_locks"] > 5 or pg["slow_queries_over_threshold"] > 5:
        return "CRITICAL"
    if (pg["connections_percent"] or 0) > 70 or pg["waiting_locks"] > 0 or pg["slow_queries_over_threshold"] > 0:
        return "WARNING"
    return "HEALTHY"


def run():
    log.info("CPUAgent collecting OS + Postgres metrics")
    cpu = collect_cpu()
    mem = collect_memory()
    trend = update_trend(mem["ram_used_percent"])
    pg = collect_postgres_activity()

    result = {
        "agent": "cpu_agent",
        "status": "SUCCESS",
        "cpu": cpu,
        "cpu_health": cpu_health(cpu),
        "memory": mem,
        "memory_trend": trend,
        "memory_health": memory_health(mem, trend),
        "postgres_activity": pg,
        "postgres_health": postgres_health(pg),
    }
    healths = (result["cpu_health"], result["memory_health"], result["postgres_health"])
    if "CRITICAL" in healths:
        overall = "CRITICAL"
    elif "WARNING" in healths:
        overall = "WARNING"
    elif "MONITOR" in healths:
        overall = "MONITOR"
    else:
        overall = "HEALTHY"
    result["overall_health"] = overall
    log.info("CPU=%s MEM=%s PG=%s -> overall=%s", *healths, overall)
    return result


if __name__ == "__main__":
    result = run()
    print("\n--- CPU Agent Output ---")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
