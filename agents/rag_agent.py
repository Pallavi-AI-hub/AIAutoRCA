"""
RAG Agent
- Uses retrieval.py from the RAG team for ChromaDB semantic search
- Extracts performance benchmark fields from full_cases
- Calculates historical benchmark statistics for comparison
- Used by RCA Agent for evidence-based root cause analysis
"""
 
import logging
import statistics
import sys
from pathlib import Path
from dotenv import load_dotenv
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("RAGAgent")
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
 
# ── Import retrieval from RAG team ───────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "rag"))
from retrieval import retrieval
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SAFE TYPE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _safe_float(value, default=0.0) -> float:
    try:
        v = float(value)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default
 
def _safe_int(value, default=0) -> int:
    try:
        v = int(float(value))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def calculate_benchmarks(incidents: list, current_volume: int = 0) -> dict:
    """
    Calculate benchmark statistics from retrieved historical full_cases.
    Filters by similar record volume if current_volume is provided.
    """
    if not incidents:
        return {}
 
    # Filter incidents with performance data
    perf_incidents = [
        inc for inc in incidents
        if _safe_float(inc.get("rows_per_second")) > 0
        or _safe_float(inc.get("elapsed_seconds")) > 0
    ]
 
    if not perf_incidents:
        log.info("No performance data found in retrieved incidents")
        return {}
 
    # Filter by similar volume if current_volume provided
    similar_volume = []
    if current_volume > 0:
        volume_tolerance = 0.5  # 50% tolerance
        similar_volume = [
            inc for inc in perf_incidents
            if abs(_safe_int(inc.get("record_volume", 0)) - current_volume)
<= current_volume * volume_tolerance
        ]
        if similar_volume:
            log.info(
                "Found %d incidents with similar volume to %d",
                len(similar_volume), current_volume
            )
            perf_incidents = similar_volume
 
    # Extract metrics
    elapsed_list  = [_safe_float(i.get("elapsed_seconds"))         for i in perf_incidents if _safe_float(i.get("elapsed_seconds"))         > 0]
    rps_list      = [_safe_float(i.get("rows_per_second"))         for i in perf_incidents if _safe_float(i.get("rows_per_second"))         > 0]
    improve_list  = [_safe_float(i.get("fix_reduced_time_by_pct")) for i in perf_incidents if _safe_float(i.get("fix_reduced_time_by_pct")) > 0]
    est_imp_list  = [_safe_float(i.get("estimated_improvement_pct")) for i in perf_incidents if _safe_float(i.get("estimated_improvement_pct")) > 0]
 
    # Most common insert method
    methods      = [i.get("insert_method", "") for i in perf_incidents if i.get("insert_method")]
    common_method = max(set(methods), key=methods.count) if methods else "Unknown"
 
    # Most common batch size
    batch_sizes  = [_safe_int(i.get("batch_size")) for i in perf_incidents if _safe_int(i.get("batch_size")) > 0]
    common_batch  = max(set(batch_sizes), key=batch_sizes.count) if batch_sizes else 0
 
    # Best fix from highest improvement incident
    best_fix = ""
    if improve_list:
        best_inc = max(perf_incidents, key=lambda x: _safe_float(x.get("fix_reduced_time_by_pct")))
        best_fix = best_inc.get("fix_applied", "")[:300]
 
    benchmarks = {
        "benchmark_incident_count"  : len(perf_incidents),
        "avg_elapsed_seconds"       : round(statistics.mean(elapsed_list), 1)  if elapsed_list  else 0,
        "best_elapsed_seconds"      : round(min(elapsed_list), 1)               if elapsed_list  else 0,
        "worst_elapsed_seconds"     : round(max(elapsed_list), 1)               if elapsed_list  else 0,
        "avg_rows_per_second"       : round(statistics.mean(rps_list), 1)       if rps_list      else 0,
        "best_rows_per_second"      : round(max(rps_list), 1)                   if rps_list      else 0,
        "avg_improvement_pct"       : round(statistics.mean(improve_list), 1)   if improve_list  else 0,
        "avg_estimated_improvement" : round(statistics.mean(est_imp_list), 1)   if est_imp_list  else 0,
        "common_insert_method"      : common_method,
        "common_batch_size"         : common_batch,
        "most_successful_fix"       : best_fix,
        "volume_matched"            : current_volume > 0 and bool(similar_volume),
    }
 
    log.info(
        "Benchmarks | incidents=%d | avg_rps=%.1f | avg_elapsed=%.1fs | method=%s",
        benchmarks["benchmark_incident_count"],
        benchmarks["avg_rows_per_second"],
        benchmarks["avg_elapsed_seconds"],
        benchmarks["common_insert_method"],
    )
 
    return benchmarks
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAP full_case TO INCIDENT DICT
# ══════════════════════════════════════════════════════════════════════════════
def _map_full_case_to_incident(full_case: dict, rank: int) -> dict:
    """
    Map a full_case from retrieval.py to the incident dict
    that rca_agent.py expects — including all performance fields.
    """
    return {
        # Core fields
        "incident_id"            : full_case.get("incident_id",     ""),
        "issue_title"            : full_case.get("issue_title",     ""),
        "priority"               : full_case.get("priority",        ""),
        "scenario_type"          : full_case.get("scenario_type",   ""),
        "dag_id"                 : full_case.get("dag_id",          ""),
        "task_id"                : full_case.get("task_id",         ""),
        "root_cause"             : full_case.get("root_cause",      ""),
        "fix_applied"            : full_case.get("fix_applied",     ""),
        "business_impact"        : full_case.get("business_impact", ""),
        "recurrence_flag"        : full_case.get("recurrence_flag", ""),
        "failure_pattern"        : full_case.get("failure_pattern", ""),
        "similar_incident_ids"   : full_case.get("similar_incident_ids", ""),
        "similarity_score"       : round(1 - (rank * 0.05), 4),  # approximated from rank
 
        # Performance fields
        "record_volume"          : _safe_int(full_case.get("record_volume")),
        "elapsed_seconds"        : _safe_float(full_case.get("elapsed_seconds")),
        "rows_per_second"        : _safe_float(full_case.get("rows_per_second")),
        "insert_method"          : full_case.get("insert_method",   ""),
        "batch_size"             : _safe_int(full_case.get("batch_size")),
        "sla_threshold_seconds"  : _safe_int(full_case.get("sla_threshold_seconds")),
        "sla_breached_by_seconds": _safe_float(full_case.get("sla_breached_by_seconds")),
        "recurrence_count"       : _safe_int(full_case.get("recurrence_count")),
        "fix_reduced_time_by_pct": _safe_float(full_case.get("fix_reduced_time_by_pct")),
        "estimated_improvement_pct": _safe_float(full_case.get("estimated_improvement_pct")),
        "airflow_worker_memory"  : full_case.get("airflow_worker_memory",   ""),
        "postgres_shared_buffers": full_case.get("postgres_shared_buffers", ""),
        "postgres_work_mem"      : full_case.get("postgres_work_mem",       ""),
 
        # Case document for top match
        "case_document"          : full_case.get("case_document", ""),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ══════════════════════════════════════════════════════════════════════════════
def run(
    error_type    : str,
    error_message : str,
    current_volume: int = 0,
) -> dict:
    """
    Retrieve similar historical incidents using retrieval.py.
    Returns incidents + performance benchmark statistics.
    """
    log.info(
        "RAGAgent starting | error_type=%s | current_volume=%d",
        error_type, current_volume
    )
 
    try:
        query = f"{error_type} {error_message}"
        log.info("RAG query: %s", query[:100])
 
        retrieval_result = retrieval(query=query, top_k_cases=7)
 
        full_cases = retrieval_result.get("full_cases", [])
        log.info("retrieval.py returned %d full_cases", len(full_cases))
 
        if not full_cases:
            log.warning("No full_cases returned from retrieval.py")
            return {
                "agent"          : "rag_agent",
                "status"         : "SUCCESS",
                "incidents_found": 0,
                "incidents"      : [],
                "top_match"      : None,
                "benchmarks"     : {},
            }
 
        # Map full_cases to incident dicts
        incidents = [
            _map_full_case_to_incident(fc, rank=i)
            for i, fc in enumerate(full_cases)
        ]
 
        top_match = incidents[0] if incidents else None
 
        # Calculate performance benchmarks
        benchmarks = calculate_benchmarks(incidents, current_volume=current_volume)
 
        log.info(
            "RAGAgent done | incidents=%d | top=%s | benchmarks=%s",
            len(incidents),
            top_match.get("incident_id") if top_match else None,
            bool(benchmarks),
        )
 
 
        return {
            "agent"          : "rag_agent",
            "status"         : "SUCCESS",
            "incidents_found": len(incidents),
            "incidents"      : incidents,
            "top_match"      : top_match,
            "benchmarks"     : benchmarks,
        }
    except Exception as e:
        log.error("RAGAgent failed: %s", str(e), exc_info=True)
        return {
            "agent"          : "rag_agent",
            "status"         : "ERROR",
            "message"        : str(e),
            "incidents_found": 0,
            "incidents"      : [],
            "top_match"      : None,
            "benchmarks"     : {},
        }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run(
        error_type     = "SLAMiss",
        error_message  = "Task exceeded SLA threshold of 90s for 3M record load",
        current_volume = 3000000,
    )
    print(f"\nIncidents Found : {result['incidents_found']}")
    print(f"Benchmarks      : {result['benchmarks']}")
    if result.get("top_match"):
        print(f"Top Match       : {result['top_match']['incident_id']}")
        print(f"Top Match RPS   : {result['top_match']['rows_per_second']}")
