"""
Teams Summary Agent
Maps rca_agent output + all evidence into the teams_summary dict
that teams_publisher expects.

No LLM call — pure field mapping and safe string extraction.
"""

import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("TeamsSummaryAgent")

SEVERITY_MAP = {
    "UniqueViolation"  : "P1",
    "OperationalError" : "P1",
    "FileNotFoundError": "P2",
    "NullValueError"   : "P2",
    "SLAMiss"          : "P2",
    "Unknown"          : "P3",
}


def _safe(value: object, default: str = "Unknown", max_chars: int = 400) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text[:max_chars - 3].rstrip() + "..." if len(text) > max_chars else text


def _extract_constraint(error_message: str) -> str:
    match = re.search(r'unique constraint ["\']?([a-zA-Z0-9_]+)["\']?', error_message or "", re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'constraint ["\']?([a-zA-Z0-9_]+)["\']?', error_message or "", re.IGNORECASE)
    return match.group(1) if match else "Unknown"


def _extract_resolution_steps(resolution_text: str) -> tuple[str, str, str]:
    if not resolution_text:
        return "Manual investigation required", "Review logs", "Re-trigger DAG"
    lines = [line.strip() for line in resolution_text.splitlines() if line.strip()]
    steps = []
    for line in lines:
        if re.match(r"^\d+[\.\)]", line):
            steps.append(re.sub(r"^\d+[\.\)]\s*", "", line))
    while len(steps) < 3:
        steps.append("See full resolution in RCA report")
    return steps[0], steps[1], steps[2]


def _extract_preventive_steps(preventive_text: str) -> tuple[str, str, str]:
    if not preventive_text:
        return "Add data quality checks", "Implement UPSERT logic", "Add staging validation"
    lines = [line.strip() for line in preventive_text.splitlines() if line.strip()]
    actions = []
    for line in lines:
        if re.match(r"^\d+[\.\)]", line):
            actions.append(re.sub(r"^\d+[\.\)]\s*", "", line))
    while len(actions) < 3:
        actions.append("Review and implement best practices")
    return actions[0], actions[1], actions[2]


def _derive_impact_status(audit_evidence: dict, airflow_evidence: dict) -> tuple[str, str, str]:
    """
    SLA miss always implies downstream delay even though the task itself
    succeeds eventually — data freshness is delayed regardless of completion.
    """
    error_type   = airflow_evidence.get("error_type", "")
    error_msg    = str(airflow_evidence.get("error_message", ""))
    audit_status = audit_evidence.get("audit_status", "")
    agent_status = audit_evidence.get("status", "")

    if error_type in ("SLAMiss", "sla_miss") or "SLA" in error_msg:
        return "Delayed", "Delayed", "Impacted"
    elif audit_status == "FAILED" or agent_status == "ERROR":
        return "Failed", "Delayed", "Impacted"
    elif agent_status == "NO_FAILURE_FOUND":
        return "OK", "OK", "Not impacted"
    else:
        return "Failed", "Delayed", "Impacted"


def _extract_sla_summary(sla_analysis: str) -> str:
    if not sla_analysis or sla_analysis == "Unknown":
        return "SLA analysis not available"
    lines = [line.strip() for line in sla_analysis.splitlines() if line.strip()]
    summary = " | ".join(lines[:4])
    return _safe(summary, max_chars=400)


def _build_performance_summary(
    performance_insights: str,
    performance_data    : dict,
) -> str:
    """
    Build performance summary for Teams card.
    Includes current run metrics AND historical benchmark comparison
    extracted from the LLM's performance_insights text.
    """
    if not performance_data and not performance_insights:
        return "No performance data available"

    lines = []

    # Current run metrics from raw performance_data
    if performance_data:
        total        = performance_data.get("total_rows", 0)
        inserted     = performance_data.get("rows_inserted", 0)
        pct          = performance_data.get("pct_complete", 0)
        elapsed      = performance_data.get("elapsed_seconds", 0)
        rps          = performance_data.get("rows_per_second", 0)
        eta          = performance_data.get("eta_seconds", 0)
        sla_threshold= performance_data.get("sla_threshold", 0)
        sla_breached = performance_data.get("sla_breached_at", 0)
        method       = performance_data.get("insert_method", "Unknown")
        chunk        = performance_data.get("chunk_size", 0)

        eta_min     = round(eta / 60, 1) if eta else 0
        elapsed_min = round(elapsed / 60, 1) if elapsed else 0

        lines.append(f"Progress: {inserted:,} / {total:,} rows ({pct}% complete)")
        lines.append(f"Elapsed: {elapsed}s ({elapsed_min} min) | Throughput: {rps} rows/sec")
        lines.append(f"ETA: {eta}s ({eta_min} min remaining)")
        lines.append(f"SLA: threshold={sla_threshold}s | breached at={sla_breached}s | over by={round(sla_breached - sla_threshold, 1)}s")
        lines.append(f"Method: {method} | Chunk size: {chunk}")

    # Extract historical benchmark comparison block from LLM output
    if performance_insights:
        insight_lines = [l.strip() for l in performance_insights.splitlines() if l.strip()]

        hist_lines = []
        in_benchmark = False
        for line in insight_lines:
            if "HISTORICAL BENCHMARK" in line.upper():
                in_benchmark = True
                continue
            if in_benchmark:
                if any(section in line.upper() for section in
                       ["BOTTLENECK", "OPTIMIZATION", "INFRASTRUCTURE",
                        "MOST SUCCESSFUL", "CURRENT RUN"]):
                    break
                if line.startswith("-"):
                    hist_lines.append(line)

        if hist_lines:
            lines.append("Historical Comparison:")
            lines.extend(hist_lines[:6])
        else:
            # Fallback: pick up any line mentioning comparison keywords
            for line in insight_lines:
                if any(kw in line.lower() for kw in
                       ["historical average", "performance gap", "trend:",
                        "faster than", "slower than", "improving", "degrading"]):
                    lines.append(line[:200])

        # Pull the best historical fix line if present
        for line in insight_lines:
            if any(kw in line.lower() for kw in
                   ["most successful", "historical fix", "historical improvement"]):
                lines.append(f"Best Fix: {line[:200]}")
                break

    return "\n".join(lines) if lines else "No performance data available"


def _build_system_health_summary(cpu_evidence: dict) -> str:
    """
    Build a one-line system health summary from cpu_agent output for the Teams card.
    """
    if not cpu_evidence or cpu_evidence.get("status") != "SUCCESS":
        return "System health check unavailable"

    cpu    = cpu_evidence.get("cpu", {}) or {}
    mem    = cpu_evidence.get("memory", {}) or {}
    pg_act = cpu_evidence.get("postgres_activity", {}) or {}

    oom_note = ", OOM detected" if mem.get("oom_detected") else ""

    lines = [
        f"CPU: {cpu_evidence.get('cpu_health')} ({cpu.get('cpu_percent')}%)",
        f"Memory: {cpu_evidence.get('memory_health')} ({mem.get('ram_used_percent')}% used{oom_note})",
        f"Postgres Activity: {cpu_evidence.get('postgres_health')} "
        f"({pg_act.get('total_connections', 'N/A')}/{pg_act.get('max_connections', 'N/A')} conns, "
        f"{pg_act.get('waiting_locks', 0)} waiting locks)",
        f"Overall: {cpu_evidence.get('overall_health')}",
    ]
    return " | ".join(lines)


def run(
    final_rca        : dict,
    airflow_evidence : dict,
    audit_evidence   : dict,
    postgres_evidence: dict,
    rag_evidence     : dict,
    cpu_evidence     : dict = None,
    performance_data : dict = None,
) -> dict:
    log.info("TeamsSummaryAgent building teams_summary")

    error_type    = airflow_evidence.get("error_type",    "Unknown")
    error_message = airflow_evidence.get("error_message", "Unknown")
    product_id    = airflow_evidence.get("product_id",    "Unknown")
    dag_id        = airflow_evidence.get("dag_id",        "Unknown")
    task_id       = airflow_evidence.get("task_id",       "Unknown")

    incident_id          = final_rca.get("incident_id",          "Unknown")
    confidence           = final_rca.get("confidence_score",     "Unknown")
    root_cause           = final_rca.get("root_cause",           "Unknown")
    summary              = final_rca.get("summary",              "Unknown")
    resolution           = final_rca.get("resolution",            "")
    preventive           = final_rca.get("preventive_actions",   final_rca.get("preventive_action", ""))
    sla_analysis         = final_rca.get("sla_analysis",         "")
    performance_insights = final_rca.get("performance_insights", "")

    perf_data = performance_data or {}
    cpu_ev    = cpu_evidence or {}

    # PostgreSQL check — give SLA-aware message instead of generic "not found"
    product_name = postgres_evidence.get("product_name")
    pg_exists    = postgres_evidence.get("exists", False)
    pg_status    = postgres_evidence.get("status", "ERROR")

    if error_type in ("SLAMiss", "sla_miss"):
        total_rows = perf_data.get("total_rows", 0)
        pg_check = (
            f"SLA Miss — {total_rows:,} rows being processed. No product lookup required."
            if total_rows else "SLA Miss — no product lookup required."
        )
    elif pg_exists and product_name:
        pg_check = f"Product ID {product_id} exists — {product_name}"
    elif pg_status == "SUCCESS":
        pg_check = f"Product ID {product_id} not found in product_master"
    else:
        pg_check = "DB check unavailable"

    incidents_found  = rag_evidence.get("incidents_found", 0)
    top_match        = rag_evidence.get("top_match") or {}
    top_match_id     = top_match.get("incident_id", "")
    historical_cases = (
        f"{incidents_found} similar case(s) found — closest: {top_match_id}"
        if incidents_found > 0
        else "No similar historical cases found"
    )

    severity                                       = SEVERITY_MAP.get(error_type, "P3")
    constraint                                     = _extract_constraint(error_message)
    product_load, catalog_update, downstream_feeds = _derive_impact_status(audit_evidence, airflow_evidence)
    res_step1, res_step2, res_step3                = _extract_resolution_steps(resolution)
    prev_act1, prev_act2, prev_act3                = _extract_preventive_steps(preventive)
    sla_summary                                    = _extract_sla_summary(sla_analysis)
    perf_summary                                   = _build_performance_summary(performance_insights, perf_data)
    system_health_summary                          = _build_system_health_summary(cpu_ev)

    teams_summary = {
        "incident_id"            : _safe(incident_id),
        "dag"                    : _safe(dag_id),
        "task"                   : _safe(task_id),
        "severity"               : severity,
        "status"                 : "Confirmed" if final_rca.get("status") == "SUCCESS" else "Under Investigation",
        "confidence"             : _safe(confidence),

        "error"                  : _safe(error_message),
        "product_id"             : _safe(product_id),
        "constraint"             : constraint,
        "cause"                  : _safe(root_cause, max_chars=400),

        "airflow_logs"           : _safe(error_message),
        "postgresql_check"       : pg_check,
        "historical_cases"       : historical_cases,
        "sla_analysis"           : sla_summary,
        "performance_insights"   : perf_summary,
        "system_health"          : system_health_summary,

        "product_load"           : product_load,
        "catalog_update"         : catalog_update,
        "downstream_feeds"       : downstream_feeds,

        "remove_duplicate_record": _safe(res_step1),
        "reload_source_file"     : _safe(res_step2),
        "rerun_dag"              : _safe(res_step3),

        "prevention_high_1"      : _safe(prev_act1),
        "prevention_high_2"      : _safe(prev_act2),
        "prevention_medium_1"    : _safe(prev_act3),

        "final_verdict"          : _safe(summary, max_chars=900),
    }

    log.info(
        "TeamsSummaryAgent done | incident=%s | severity=%s | confidence=%s",
        teams_summary["incident_id"],
        teams_summary["severity"],
        teams_summary["confidence"],
    )

    return {
        **final_rca,
        "teams_summary": teams_summary,
    }


if __name__ == "__main__":
    result = run(
        final_rca={
            "agent": "rca_agent", "status": "SUCCESS",
            "incident_id": 5,
            "confidence_score": "0.8",
            "summary": "1M record ETL load exceeded SLA of 30s.",
            "root_cause": "execute_values insufficient for 1M records.",
            "sla_analysis": "SLA threshold 30s. Breached at 30.1s.",
            "performance_insights": (
                "CURRENT RUN STATUS:\n"
                "- Total records: 1,000,000\n"
                "HISTORICAL BENCHMARK COMPARISON:\n"
                "- Historical average throughput: 3206.5 rows/sec (from 7 incidents)\n"
                "- Historical average runtime: 265.6 seconds\n"
                "- Current throughput: 16931.0 rows/sec\n"
                "- Performance gap: -427.9% — current run is faster than historical average\n"
                "- Trend: IMPROVING\n"
                "- Most successful historical fix: Switched to execute_values with batch_size=50000\n"
                "- Historical improvement achieved: 53.0%\n"
                "BOTTLENECK DETECTION:\n"
                "- No bottleneck detected\n"
            ),
            "business_impact": "Catalog delayed. Downstream impacted.",
            "resolution": "1. Increase chunk size\n2. Switch to COPY\n3. Re-trigger DAG",
            "preventive_actions": "1. Use COPY command\n2. Partition table\n3. Tune shared_buffers",
        },
        airflow_evidence={
            "status": "SUCCESS", "dag_id": "Daily_product_1Mdata_ETL_job",
            "task_id": "load_1m_data", "error_type": "SLAMiss",
            "error_message": "Task exceeded SLA threshold of 30s",
            "product_id": None,
        },
        audit_evidence={
            "status": "SUCCESS", "audit_id": 5,
            "audit_status": "FAILED", "records_processed": 510000,
            "error_message": "SLA exceeded",
        },
        postgres_evidence={
            "status": "SUCCESS", "product_id": None,
            "exists": False, "product_name": None, "brand": None,
        },
        rag_evidence={
            "status": "SUCCESS", "incidents_found": 7,
            "top_match": {"incident_id": "INC052", "similarity_score": 0.91},
        },
        cpu_evidence={
            "status": "SUCCESS",
            "cpu": {"cpu_percent": 0.3},
            "cpu_health": "HEALTHY",
            "memory": {"ram_used_percent": 33.9, "oom_detected": False},
            "memory_health": "HEALTHY",
            "postgres_activity": {"total_connections": 9, "max_connections": 100, "waiting_locks": 0},
            "postgres_health": "HEALTHY",
            "overall_health": "HEALTHY",
        },
        performance_data={
            "total_rows": 1000000, "rows_inserted": 510000,
            "pct_complete": 51.0, "elapsed_seconds": 30.1,
            "rows_per_second": 16931.0, "eta_seconds": 28.9,
            "sla_threshold": 30, "sla_breached_at": 30.1,
            "chunk_size": 10000, "insert_method": "execute_values",
            "data_volume": "1M",
        },
    )
    print("\n--- Teams Summary ---")
    for k, v in result["teams_summary"].items():
        print(f"  {k:30s}: {v}")
