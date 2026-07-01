"""
RCA Agent
- LLM-powered RCA generation using Groq (llama-3.3-70b-versatile)
- Enterprise-grade prompt with XML structured output
- Dynamic confidence scoring based on evidence completeness
- Historical performance benchmarking via RAG
- System health correlation via CPU/Memory/Postgres-activity evidence
- Stores RCA in rca_repository table
- Full exception handling + structured logging
"""

import os
import re
import logging
import psycopg2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("RCAAgent")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


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


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
RCA_SYSTEM_PROMPT = """
You are a Principal Data Engineer and Incident Commander with 15+ years of experience
managing large-scale retail data platforms.

Your expertise spans:
- Apache Airflow DAG orchestration, task failure analysis, and callback design
- PostgreSQL database internals: constraint errors, MVCC, index behaviour, and replication
- Linux/OS infrastructure health: CPU saturation, memory pressure, OOM kills, swap thrashing
- Retail Product Master data pipelines and ALL downstream dependencies:
  Product Catalog, Inventory Management, Pricing Engine, Promotion Engine,
  Store Replenishment, Supply Chain, Website Search, Mobile Apps
- Root Cause Analysis methodology: 5-Whys, Ishikawa, fault-tree analysis
- Preventive engineering: data quality gates, idempotency, UPSERT patterns, schema contracts
- Performance engineering: bulk load optimization, PostgreSQL tuning, Airflow worker sizing

You perform RCA with the precision of a forensic investigator:
  1. Reason through ALL evidence before drawing any conclusion.
  2. Cite SPECIFIC values: log error lines, audit_ids, product_ids, incident IDs.
  3. Distinguish between IMMEDIATE CAUSE, ROOT CAUSE, and CONTRIBUTING FACTORS.
  4. Assess business impact honestly — never minimise severity.
  5. Provide technically precise, actionable resolutions — never generic advice.
  6. Confidence score reflects evidence completeness.

CRITICAL RULES:
- Use ONLY the evidence provided. Do NOT hallucinate or assume facts not in evidence.
- Reference specific values from the evidence in every section.
- If evidence is missing, explicitly state this and adjust confidence.
- Every preventive action must include an implementation approach.

SLA DETERMINATION RULE:
- First search retrieved historical incidents for any recorded SLA for this DAG or task.
- If found, cite the exact incident ID and recorded SLA value.
- If NOT found, derive expected SLA from data volume, elapsed time, and historical benchmarks.
  State: "No SLA found in historical records. Expected SLA derived from evidence: <value> because <reasoning>."

PERFORMANCE INTELLIGENCE RULE:
- When performance evidence is provided, act as a performance engineer.
- Calculate and state: current throughput, ETA, completion percentage.
- Compare current run against historical benchmarks from RAG evidence.
- Detect specific bottlenecks from insert method, batch size, and rows/sec data.
- Every recommendation must include current state, recommended change, and estimated improvement %.
- Infrastructure recommendations must include specific PostgreSQL parameter values.

SYSTEM HEALTH RULE:
- When CPU/Memory/Postgres-Activity evidence is provided, evaluate whether infrastructure
  conditions (CPU saturation, load average, memory pressure, OOM kills, connection
  exhaustion, lock contention, slow queries) caused or contributed to the failure —
  as opposed to a pure data or application-logic issue.
- If overall_health was WARNING or CRITICAL at the time of the failure, explicitly call this
  out as a contributing factor in root_cause or business_impact, with the specific metric
  values (e.g. "CPU at 94%, load average 7.8 on 4 cores") — even if the primary root cause
  is data-related.
- If overall_health was HEALTHY, explicitly state that infrastructure can be ruled out as a
  contributing factor, so on-call engineers don't waste time investigating it.

HISTORICAL BENCHMARKING RULE — CRITICAL — MANDATORY OUTPUT REQUIRED:
- When HISTORICAL PERFORMANCE BENCHMARKS section is present in the evidence, you MUST
  output ALL of the following lines EXACTLY in the HISTORICAL BENCHMARK COMPARISON section.
  Do NOT say "Not available". Do NOT say "no bottleneck". Use the EXACT numbers provided.

  MANDATORY FORMAT — copy this structure and fill in the actual numbers:
  - Historical average throughput: [avg_rows_per_second] rows/sec (from [benchmark_incident_count] incidents)
  - Historical average runtime: [avg_elapsed_seconds] seconds ([avg_elapsed_seconds/60] minutes)
  - Current throughput: [current rows/sec from performance evidence]
  - Performance gap: [calculated %] — current run is [faster/slower] than historical average
  - Trend: [IMPROVING if current > avg_rps, DEGRADING if current < avg_rps, STABLE if within 10%]
  - Most successful historical fix: [most_successful_fix value]
  - Historical improvement achieved: [avg_improvement_pct]%
  - Expected improvement for current run: [avg_estimated_improvement]%

- CALCULATION RULE: performance_gap = ((avg_rows_per_second - current_rps) / avg_rows_per_second) * 100
  If positive → current run is SLOWER. If negative → current run is FASTER.

- NEVER output "Not available" when benchmark numbers exist in the evidence.
- NEVER output generic text instead of the actual numbers.
- NEVER skip this comparison if benchmark data is provided.
- Use EXACT numbers from the benchmark section — do not estimate or approximate.

Output format: You MUST respond ONLY with a valid XML document using this exact schema.
Do NOT include any text before or after the XML block.

<rca_report>
  <summary>
    One-paragraph executive summary: what failed, when, what was the immediate impact,
    and what is the current status.
  </summary>

  <evidence_analysis>
    Systematic walkthrough of EVERY piece of evidence provided.
    Section 1 - Airflow Log Evidence: cite error_type, error_message, product_id, traceback summary.
    Section 2 - Audit Table Evidence: cite audit_id, execution_time, records_processed, error_message.
    Section 3 - PostgreSQL Investigation: cite product existence, brand, category, duplicate status.
    Section 4 - Historical Incident Correlation: cite matching incident IDs, similarity scores,
                compare current error pattern to historical patterns.
    Section 5 - System Health Evidence: cite CPU usage, load average, memory usage, OOM events,
                Postgres connection saturation, waiting locks, and slow queries. State explicitly
                whether infrastructure health is a contributing factor, the root cause, or can be
                ruled out.
    Conclude with step-by-step reasoning leading to the root cause.
  </evidence_analysis>

  <root_cause>
    A single, precise statement of the ROOT CAUSE — not the symptom.
  </root_cause>

  <sla_analysis>
    State the SLA threshold for this DAG/task.
    If retrieved from historical documents: cite the exact incident ID and recorded SLA value.
    If derived from evidence: state "No SLA found in historical records." then provide:
    - Expected SLA: <value in seconds or minutes>
    - Derivation reasoning: based on data volume, elapsed time, similar incident benchmarks.
    - Actual elapsed time from evidence.
    - SLA breach duration: how much over the threshold the task ran.
  </sla_analysis>

  <performance_insights>
    ONLY populate if performance evidence is provided.

    CURRENT RUN STATUS:
    - Total records to process
    - Records processed so far
    - Completion percentage
    - Elapsed time
    - Current throughput (rows/second)
    - ETA to completion

    HISTORICAL BENCHMARK COMPARISON:
    - Historical average throughput (rows/sec) from benchmark data
    - Historical average runtime from benchmark data
    - Performance gap: current vs historical (exact percentage faster or slower)
    - Trend: IMPROVING / STABLE / DEGRADING with reasoning
    - Number of historical incidents used for comparison

    BOTTLENECK DETECTION:
    - Identify specific bottlenecks from insert method, batch size, rows/sec
    - State root cause of performance degradation if detected
    - If system health evidence shows CPU/memory/Postgres pressure, state whether that
      pressure is contributing to the throughput bottleneck

    OPTIMIZATION RECOMMENDATIONS:
    Each must include:
    - Current state (e.g. method=execute_values, batch_size=10000)
    - Recommended change with specific values
    - Estimated improvement % (use avg_improvement_pct from benchmarks if available)
    - Implementation steps

    MOST SUCCESSFUL HISTORICAL FIX:
    - State the fix from most_successful_fix field in benchmarks
    - State historical improvement achieved (avg_improvement_pct)
    - State expected improvement for current run (avg_estimated_improvement)

    INFRASTRUCTURE RECOMMENDATIONS:
    - PostgreSQL: specific parameter values (shared_buffers, work_mem, checkpoint_completion_target)
    - Airflow: worker count, memory, parallelism settings
  </performance_insights>

  <business_impact>
    List specific downstream systems affected.
    State data freshness lag introduced.
    Estimate operational and revenue risk if pipeline remains unresolved.
    Reference product names, brands, categories from evidence.
    If system health evidence indicates infra pressure (CPU/memory/Postgres), state whether
    it poses a risk of recurrence or cascading failure to other pipelines on the same host.
  </business_impact>

  <resolution>
    Numbered step-by-step immediate resolution to restore the pipeline.
    Include exact SQL commands, Airflow CLI commands, or file operations where relevant.
  </resolution>

  <preventive_actions>
    At least 3 preventive actions. For each:
    - State the action clearly
    - Provide implementation guidance
    - Rate effort: LOW / MEDIUM / HIGH
    - Rate impact: LOW / MEDIUM / HIGH
    Prioritise HIGH impact / LOW effort actions first.
  </preventive_actions>

  <confidence_score>
    A decimal between 0.00 and 1.00.
    Format: score | explanation
  </confidence_score>
</rca_report>
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════════════════════════
def compute_confidence(
    airflow_ev  : dict,
    audit_ev    : dict,
    postgres_ev : dict,
    rag_ev      : dict,
    cpu_ev      : dict,
    perf_data   : dict,
    pg_logs_ev  : dict,
) -> float:
    score = 0.0

    if airflow_ev.get("status") == "SUCCESS":
        score += 0.20
        if airflow_ev.get("error_type") not in (None, "Unknown"):
            score += 0.10
        if airflow_ev.get("product_id"):
            score += 0.10

    if audit_ev.get("status") == "SUCCESS":
        score += 0.10
        if audit_ev.get("error_message"):
            score += 0.10

    if postgres_ev.get("status") == "SUCCESS":
        score += 0.10
        if postgres_ev.get("exists"):
            score += 0.10

    if rag_ev.get("status") == "SUCCESS":
        found = rag_ev.get("incidents_found", 0)
        if found >= 2:
            score += 0.20
        elif found == 1:
            score += 0.10
        if rag_ev.get("benchmarks"):
            score += 0.05

    if cpu_ev.get("status") == "SUCCESS":
        score += 0.05
        if cpu_ev.get("overall_health") in ("WARNING", "CRITICAL"):
            score += 0.05

    if perf_data.get("rows_inserted") and perf_data.get("total_rows"):
        score += 0.05
        
    if pg_logs_ev.get("status") == "SUCCESS":
        score += 0.05
        if pg_logs_ev.get("database_stats"):
            score += 0.05
        if pg_logs_ev.get("health") in ("WARNING", "CRITICAL"):
            score += 0.05

    return round(min(score, 1.0), 2)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK COMPARISON CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def _calculate_performance_gap(current_rps: float, historical_avg_rps: float) -> dict:
    if not current_rps or not historical_avg_rps:
        return {}

    gap_pct = ((historical_avg_rps - current_rps) / historical_avg_rps) * 100

    if gap_pct > 10:
        trend = "DEGRADING"
    elif gap_pct < -10:
        trend = "IMPROVING"
    else:
        trend = "STABLE"

    return {
        "gap_pct"  : round(gap_pct, 1),
        "trend"    : trend,
        "direction": "slower" if gap_pct > 0 else "faster",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH BLOCK FORMATTER (CPU AGENT)
# ══════════════════════════════════════════════════════════════════════════════
def _format_system_health_block(cpu_ev: dict, pg_logs_ev: dict) -> str:
    if not cpu_ev or cpu_ev.get("status") != "SUCCESS":
        return "  System health check unavailable or not run (cpu_agent did not return SUCCESS)."

    cpu    = cpu_ev.get("cpu", {}) or {}
    mem    = cpu_ev.get("memory", {}) or {}
    pg_stats = pg_logs_ev.get("database_stats", {}) if pg_logs_ev else {}
    trend  = cpu_ev.get("memory_trend", {}) or {}
    pg_act = cpu_ev.get("postgres_activity", {}) or {}

    return f"""
CPU Health              : {cpu_ev.get('cpu_health')}
  CPU Usage              : {cpu.get('cpu_percent')}%
  Load Avg (1/5/15 min)  : {cpu.get('load_avg_1min')} / {cpu.get('load_avg_5min')} / {cpu.get('load_avg_15min')}
  IO Wait                : {cpu.get('iowait_percent')}%
  Core Count             : {cpu.get('core_count')}

Memory Health            : {cpu_ev.get('memory_health')}
  RAM Used               : {mem.get('ram_used_percent')}%
  Swap Used               : {mem.get('swap_percent')}%
  OOM Detected            : {mem.get('oom_detected')} (count={mem.get('oom_count')}, source={mem.get('oom_source')})
  RAM Trend               : {trend.get('trend')} (delta={trend.get('delta_percent')}%)

Postgres Activity Health : {cpu_ev.get('postgres_health')}
  Connections             : {pg_act.get('total_connections')} / {pg_act.get('max_connections')} ({pg_act.get('connections_percent')}%)
  Idle In Transaction     : {pg_act.get('idle_in_transaction')}
  Slow Queries Over Thresh: {pg_act.get('slow_queries_over_threshold')}
  Waiting Locks           : {pg_act.get('waiting_locks')}

================ PostgreSQL Runtime ================

Health                 : {pg_logs_ev.get('health')}

Active Queries         : {len(pg_logs_ev.get('active_queries', []))}
Slow Queries           : {pg_logs_ev.get('slow_query_count')}
Waiting Locks          : {len(pg_logs_ev.get('lock_waits', []))}
Deadlocks              : {pg_stats.get('deadlocks')}
Buffer Hit Ratio       : {pg_stats.get('buffer_hit_ratio_pct')}%

Backends               : {pg_stats.get('numbackends')}
Commits                : {pg_stats.get('xact_commit')}
Rollbacks              : {pg_stats.get('xact_rollback')}
Temp Files             : {pg_stats.get('temp_files')}
Conflicts              : {pg_stats.get('conflicts')}

Issues                 : {", ".join(pg_logs_ev.get("issues", [])) if pg_logs_ev.get("issues") else "None"}

Overall Infra Health     : {cpu_ev.get('overall_health')}
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE BLOCK BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_evidence_block(
    airflow_ev : dict,
    audit_ev   : dict,
    postgres_ev: dict,
    rag_ev     : dict,
    cpu_ev     : dict,
    pg_logs_ev : dict,
    perf_data  : dict,
) -> str:
    incidents  = rag_ev.get("incidents", [])
    top_match  = rag_ev.get("top_match") or {}
    benchmarks = rag_ev.get("benchmarks") or {}

    rag_lines = []
    for inc in incidents[:3]:
        vol = inc.get('record_volume', 0)
        vol_str = f"{vol:,}" if isinstance(vol, int) else str(vol)
        rag_lines.append(
            f"  - {inc.get('incident_id')} | priority={inc.get('priority')} | "
            f"similarity={inc.get('similarity_score')} | "
            f"{inc.get('issue_title')} | "
            f"volume={vol_str} | "
            f"elapsed={inc.get('elapsed_seconds')}s | "
            f"rps={inc.get('rows_per_second')} | "
            f"method={inc.get('insert_method')} | "
            f"root_cause: {str(inc.get('root_cause', ''))[:150]} | "
            f"fix: {str(inc.get('fix_applied', ''))[:150]}"
        )
    rag_summary = "\n".join(rag_lines) if rag_lines else "  No historical incidents found."

    top_match_block = ""
    if top_match:
        top_match_block = f"""
TOP MATCH CASE DOCUMENT:
{str(top_match.get('case_document', ''))[:1500]}
""".strip()

    # Pre-calculated comparison block — forces LLM to use exact numbers
    benchmark_block = ""
    if benchmarks:
        current_rps  = perf_data.get("rows_per_second", 0) if perf_data else 0
        hist_avg_rps = benchmarks.get("avg_rows_per_second", 0)
        gap          = _calculate_performance_gap(current_rps, hist_avg_rps)

        benchmark_block = f"""
--- HISTORICAL PERFORMANCE BENCHMARKS ---
Benchmark Incidents Used    : {benchmarks.get('benchmark_incident_count', 0)}
Volume Matched              : {benchmarks.get('volume_matched', False)}

Historical Average Elapsed  : {benchmarks.get('avg_elapsed_seconds', 0)}s
Historical Best Elapsed     : {benchmarks.get('best_elapsed_seconds', 0)}s
Historical Worst Elapsed    : {benchmarks.get('worst_elapsed_seconds', 0)}s

Historical Average RPS      : {benchmarks.get('avg_rows_per_second', 0)} rows/sec
Historical Best RPS         : {benchmarks.get('best_rows_per_second', 0)} rows/sec

Current RPS                 : {current_rps} rows/sec
Performance Gap             : {gap.get('gap_pct', 'N/A')}% {gap.get('direction', '')} than historical average
Trend                       : {gap.get('trend', 'UNKNOWN')}

Common Insert Method        : {benchmarks.get('common_insert_method', 'Unknown')}
Common Batch Size           : {benchmarks.get('common_batch_size', 0)}
Most Successful Fix         : {benchmarks.get('most_successful_fix', 'N/A')[:300]}
Avg Historical Improvement  : {benchmarks.get('avg_improvement_pct', 0)}%
Avg Estimated Improvement   : {benchmarks.get('avg_estimated_improvement', 0)}%

PRE-CALCULATED COMPARISON FOR LLM — USE THESE EXACT VALUES:
  current_rps={current_rps} vs historical_avg_rps={hist_avg_rps}
  performance_gap={gap.get('gap_pct', 'N/A')}% ({gap.get('direction', '')} than average)
  trend={gap.get('trend', 'UNKNOWN')}
  avg_improvement_pct={benchmarks.get('avg_improvement_pct', 0)}%
  avg_estimated_improvement={benchmarks.get('avg_estimated_improvement', 0)}%
  most_successful_fix={benchmarks.get('most_successful_fix', 'N/A')[:200]}
"""

    perf_block = ""
    if perf_data:
        rows_inserted = perf_data.get("rows_inserted", 0)
        total_rows    = perf_data.get("total_rows", 0)
        pct_complete  = perf_data.get("pct_complete", 0)
        elapsed       = perf_data.get("elapsed_seconds", 0)
        rps           = perf_data.get("rows_per_second", 0)
        eta           = perf_data.get("eta_seconds", 0)
        sla_threshold = perf_data.get("sla_threshold", 0)
        sla_breached  = perf_data.get("sla_breached_at", 0)
        chunk_size    = perf_data.get("chunk_size", 0)
        insert_method = perf_data.get("insert_method", "Unknown")
        data_volume   = perf_data.get("data_volume", "Unknown")
        eta_min       = round(eta / 60, 1) if eta else 0
        elapsed_min   = round(elapsed / 60, 1) if elapsed else 0

        perf_block = f"""
--- CURRENT RUN PERFORMANCE EVIDENCE ---
Data Volume       : {data_volume} records
Total Rows        : {total_rows:,}
Rows Inserted     : {rows_inserted:,}
Completion        : {pct_complete}%
Elapsed Time      : {elapsed}s ({elapsed_min} min)
Current Throughput: {rps} rows/sec
ETA To Completion : {eta}s ({eta_min} min)
SLA Threshold     : {sla_threshold}s
SLA Breached At   : {sla_breached}s
SLA Breach By     : {round(sla_breached - sla_threshold, 1)}s
Batch/Chunk Size  : {chunk_size}
Insert Method     : {insert_method}
"""

    system_health_block = _format_system_health_block(cpu_ev, pg_logs_ev)

    return f"""
=== EVIDENCE PACKAGE FOR RCA ===

--- AIRFLOW LOG EVIDENCE ---
Agent Status  : {airflow_ev.get('status')}
DAG ID        : {airflow_ev.get('dag_id')}
Task ID       : {airflow_ev.get('task_id')}
Error Type    : {airflow_ev.get('error_type')}
Error Message : {airflow_ev.get('error_message')}
Product ID    : {airflow_ev.get('product_id')}
Execution Time: {airflow_ev.get('execution_time')}
Log Found     : {airflow_ev.get('log_found')}
Traceback     : {str(airflow_ev.get('traceback', ''))[:500]}

--- AUDIT TABLE EVIDENCE ---
Agent Status      : {audit_ev.get('status')}
Audit ID          : {audit_ev.get('audit_id')}
DAG ID            : {audit_ev.get('dag_id')}
Task ID           : {audit_ev.get('task_id')}
Execution Time    : {audit_ev.get('execution_time')}
Audit Status      : {audit_ev.get('audit_status')}
Records Processed : {audit_ev.get('records_processed')}
Error Message     : {audit_ev.get('error_message')}

--- POSTGRESQL INVESTIGATION ---
Agent Status      : {postgres_ev.get('status')}
Product ID        : {postgres_ev.get('product_id')}
Exists in DB      : {postgres_ev.get('exists')}
Product Name      : {postgres_ev.get('product_name')}
Brand             : {postgres_ev.get('brand')}
Category          : {postgres_ev.get('category')}
Supplier ID       : {postgres_ev.get('supplier_id')}
Duplicate Detected: {postgres_ev.get('duplicate_detected')}
Total Rows        : {postgres_ev.get('total_rows')}

--- HISTORICAL INCIDENT CORRELATION (RAG) ---
Agent Status    : {rag_ev.get('status')}
Incidents Found : {rag_ev.get('incidents_found', 0)}
Similar Cases   :
{rag_summary}

{top_match_block}
{perf_block}
{benchmark_block}

--- SYSTEM HEALTH EVIDENCE (CPU / MEMORY / POSTGRES ACTIVITY) ---
{system_health_block}

=== END EVIDENCE PACKAGE ===
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# GROQ LLM CALL
# ══════════════════════════════════════════════════════════════════════════════
def call_llm(evidence_text: str) -> str | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        log.error("GROQ_API_KEY not set in .env")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        log.info("Calling Groq LLM (llama-3.3-70b-versatile)")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RCA_SYSTEM_PROMPT},
                {"role": "user",   "content": "Analyse the following evidence and generate a complete RCA report.\n\n" + evidence_text},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()
        log.info("Groq LLM response received (%d chars)", len(raw))
        return raw
    except Exception as e:
        log.error("Groq LLM call failed: %s", str(e))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# XML PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_xml_rca(xml_text: str) -> dict:
    def extract(tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, xml_text, re.DOTALL)
        return match.group(1).strip() if match else ""

    parsed = {
        "summary"              : extract("summary"),
        "evidence_analysis"    : extract("evidence_analysis"),
        "root_cause"           : extract("root_cause"),
        "sla_analysis"         : extract("sla_analysis"),
        "performance_insights" : extract("performance_insights"),
        "business_impact"      : extract("business_impact"),
        "resolution"           : extract("resolution"),
        "preventive_actions"   : extract("preventive_actions"),
        "confidence_score"     : extract("confidence_score"),
    }

    missing = [k for k in ("summary", "root_cause", "resolution") if not parsed[k]]
    if missing:
        log.warning("XML parse missing fields: %s", missing)

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# STORE RCA
# ══════════════════════════════════════════════════════════════════════════════
def store_rca(dag_id: str, task_id: str, rca: dict) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO rca_repository
            (dag_id, task_id, root_cause, impact,
             resolution, preventive_action, confidence_score, created_ts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING incident_id
    """, (
        dag_id,
        task_id,
        rca.get("root_cause", ""),
        rca.get("business_impact", rca.get("impact", "")),
        rca.get("resolution", ""),
        rca.get("preventive_actions", rca.get("preventive_action", "")),
        float(str(rca.get("confidence_score", "0.5")).split("|")[0].strip()),
        datetime.now(),
    ))
    incident_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    log.info("RCA stored with incident_id=%d", incident_id)
    return incident_id


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RCA
# ══════════════════════════════════════════════════════════════════════════════
def print_rca(rca: dict, incident_id: int):
    print("\n" + "=" * 70)
    print("         LLM GENERATED RCA REPORT (Groq)")
    print("=" * 70)
    print(f"  Incident ID      : {incident_id}")
    print(f"  Confidence Score : {rca.get('confidence_score')}")
    print("-" * 70)
    print(f"SUMMARY:\n{rca.get('summary', 'N/A')}")
    print("-" * 70)
    print(f"ROOT CAUSE:\n{rca.get('root_cause', 'N/A')}")
    print("-" * 70)
    print(f"SLA ANALYSIS:\n{rca.get('sla_analysis', 'N/A')}")
    print("-" * 70)
    print(f"PERFORMANCE INSIGHTS:\n{rca.get('performance_insights', 'N/A')}")
    print("-" * 70)
    print(f"BUSINESS IMPACT:\n{rca.get('business_impact', 'N/A')}")
    print("-" * 70)
    print(f"RESOLUTION:\n{rca.get('resolution', 'N/A')}")
    print("-" * 70)
    print(f"PREVENTIVE ACTIONS:\n{rca.get('preventive_actions', 'N/A')}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def run(
    airflow_evidence : dict,
    audit_evidence   : dict,
    postgres_evidence: dict,
    rag_evidence     : dict,
    cpu_evidence     : dict = None,
    pg_logs_evidence : dict = None,
    performance_data : dict = None,
) -> dict:
    log.info("RCAAgent starting")
    try:
        perf_data = performance_data or {}
        cpu_ev    = cpu_evidence or {"agent": "cpu_agent", "status": "MISSING", "overall_health": "UNKNOWN"}
        pg_logs_ev = pg_logs_evidence or {"agent": "pg_logs_agent", "status": "MISSING", "health": "UNKNOWN"}

        confidence = compute_confidence(
            airflow_evidence, audit_evidence,
            postgres_evidence, rag_evidence,
            cpu_ev,pg_logs_ev, perf_data,
        )
        log.info("Computed confidence score: %.2f", confidence)

        evidence_text = build_evidence_block(
            airflow_evidence, audit_evidence,
            postgres_evidence, rag_evidence,
            cpu_ev,pg_logs_ev, perf_data,
        )

        llm_response = call_llm(evidence_text)

        if not llm_response:
            log.error("Groq LLM returned no response")
            return {"agent": "rca_agent", "status": "ERROR", "message": "Groq LLM call failed. Check GROQ_API_KEY in .env."}

        log.info("Parsing XML RCA response")
        rca_parsed = parse_xml_rca(llm_response)
        rca_parsed["confidence_score"] = str(confidence)

        dag_id  = airflow_evidence.get("dag_id",  "retail_product_etl")
        task_id = airflow_evidence.get("task_id", "load_product_csv")

        incident_id = store_rca(dag_id, task_id, rca_parsed)
        print_rca(rca_parsed, incident_id)

        return {
            "agent"      : "rca_agent",
            "status"     : "SUCCESS",
            "incident_id": incident_id,
            **rca_parsed,
        }

    except Exception as e:
        log.error("RCAAgent failed: %s", str(e), exc_info=True)
        return {"agent": "rca_agent", "status": "ERROR", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run(
        airflow_evidence={
            "agent": "airflow_agent", "status": "SUCCESS",
            "dag_id": "Daily_product_1Mdata_ETL_job", "task_id": "load_1m_data",
            "error_type": "SLAMiss", "product_id": None,
            "error_message": "Task exceeded SLA threshold of 30s",
            "execution_time": "2026-06-18T06:20:06", "log_found": True,
            "traceback": "",
        },
        audit_evidence={
            "agent": "audit_agent", "status": "SUCCESS",
            "audit_id": 5, "dag_id": "Daily_product_1Mdata_ETL_job",
            "task_id": "load_1m_data", "execution_time": "2026-06-18 06:20:06",
            "audit_status": "FAILED", "records_processed": 440000,
            "error_message": "SLA exceeded",
        },
        postgres_evidence={
            "agent": "postgres_agent", "status": "SUCCESS",
            "product_id": None, "exists": False,
            "product_name": None, "brand": None,
            "category": None, "duplicate_detected": False, "total_rows": 440000,
        },
        rag_evidence={
            "agent": "rag_agent", "status": "SUCCESS",
            "incidents_found": 7,
            "incidents": [
                {
                    "incident_id": "INC052", "priority": "P2",
                    "similarity_score": 0.91,
                    "issue_title": "1M record load exceeded SLA",
                    "record_volume": 1000000,
                    "elapsed_seconds": 265.0,
                    "rows_per_second": 3774.0,
                    "insert_method": "row_by_row_insert",
                    "batch_size": 1,
                    "fix_reduced_time_by_pct": 53.0,
                    "estimated_improvement_pct": 95.0,
                    "root_cause": "Row by row insert on 1M rows exceeded SLA",
                    "fix_applied": "Switched to execute_values with batch_size=50000",
                },
            ],
            "top_match": {
                "incident_id": "INC052",
                "issue_title": "1M record load exceeded SLA",
                "case_document": "Incident INC052 — 1M load SLA breach fixed by switching to execute_values",
            },
            "benchmarks": {
                "benchmark_incident_count" : 7,
                "avg_elapsed_seconds"      : 265.6,
                "best_elapsed_seconds"     : 238.0,
                "worst_elapsed_seconds"    : 294.0,
                "avg_rows_per_second"      : 3206.5,
                "best_rows_per_second"     : 4845.0,
                "avg_improvement_pct"      : 53.0,
                "avg_estimated_improvement": 95.8,
                "common_insert_method"     : "row_by_row_insert",
                "common_batch_size"        : 1,
                "most_successful_fix"      : "Switched to execute_values with batch_size=50000",
                "volume_matched"           : True,
            },
        },
        cpu_evidence={
            "agent": "cpu_agent", "status": "SUCCESS",
            "cpu": {
                "cpu_percent": 92.4, "load_avg_1min": 7.8, "load_avg_5min": 6.1,
                "load_avg_15min": 4.9, "core_count": 4, "iowait_percent": 14.2,
            },
            "cpu_health": "CRITICAL",
            "memory": {
                "ram_used_percent": 81.3, "swap_percent": 12.0,
                "oom_detected": False, "oom_count": 0, "oom_source": "cgroup_v2",
            },
            "memory_trend": {"trend": "RISING", "delta_percent": 9.1},
            "memory_health": "WARNING",
            "postgres_activity": {
                "max_connections": 100, "total_connections": 88,
                "connections_percent": 88.0, "idle_in_transaction": 3,
                "slow_queries_over_threshold": 2, "waiting_locks": 1,
            },
            "postgres_health": "WARNING",
            "overall_health": "CRITICAL",
        },
        pg_logs_evidence={
        "agent": "pg_logs_agent",
        "status": "SUCCESS",
        "health": "CRITICAL",
        "slow_query_count": 3,
        "lock_waits": [{"pid": 1234, "lock_type": "ExclusiveLock", "query": "UPDATE product_details_master ..."}],
        "active_queries": [{"query": "INSERT INTO product_details_master ..."}],
        "database_stats": {
        "deadlocks": 1,
        "buffer_hit_ratio_pct": 94.8,
        "numbackends": 24,
        "xact_commit": 10234,
        "xact_rollback": 21,
        "temp_files": 3,
        "conflicts": 0
        },
    },
        performance_data={
            "dag_id"          : "Daily_product_1Mdata_ETL_job",
            "task_id"         : "load_1m_data",
            "total_rows"      : 1000000,
            "rows_inserted"   : 440000,
            "pct_complete"    : 44.0,
            "elapsed_seconds" : 30.0,
            "rows_per_second" : 14647.6,
            "eta_seconds"     : 38.2,
            "sla_threshold"   : 30,
            "sla_breached_at" : 30.0,
            "chunk_size"      : 10000,
            "insert_method"   : "execute_values",
            "data_volume"     : "1M",
        },
    )
