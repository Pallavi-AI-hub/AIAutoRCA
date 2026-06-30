"""
LangGraph RCA Workflow
Orchestrates all agents to produce a Groq LLM-generated RCA report
when an Airflow ETL task fails, then publishes to Microsoft Teams.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

# ── Path Setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── State ────────────────────────────────────────────────────────────────────
from core.state import RCAState

# ── Agents ───────────────────────────────────────────────────────────────────
from agents.airflow_agent       import run as airflow_run
from agents.audit_agent         import run as audit_run
from agents.postgres_agent      import run as postgres_run
from agents.rag_agent           import run as rag_run
from agents.cpu_agent           import run as cpu_run
from agents.rca_agent           import run as rca_run
from agents.teams_summary_agent import run as teams_summary_run

# ── Teams Publisher ──────────────────────────────────────────────────────────
from teams_publisher import publish_autorca_card_to_teams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("RCAWorkflow")


def supervisor_node(state: RCAState) -> RCAState:
    log.info("=" * 60)
    log.info("SUPERVISOR: Initialising RCA workflow")
    log.info("  DAG    : %s", state.get("dag_id"))
    log.info("  Task   : %s", state.get("task_id"))
    log.info("  Run ID : %s", state.get("run_id"))
    log.info("  Perf   : %s", state.get("performance_data"))
    log.info("=" * 60)
    return {
        **state,
        "run_airflow"      : True,
        "run_audit"        : True,
        "run_postgres"     : True,
        "run_rag"          : True,
        "run_cpu"          : True,
        "errors"           : [],
        "workflow_status"  : "RUNNING",
        "notification_sent": False,
        "airflow_data"     : None,
        "audit_data"       : None,
        "postgres_data"    : None,
        "rag_data"         : None,
        "cpu_data"         : None,
        "final_rca"        : None,
    }


def cpu_node(state: RCAState) -> RCAState:
    log.info("CPU NODE: Collecting OS + Postgres activity health")
    errors = list(state.get("errors", []))
    try:
        result = cpu_run()
        log.info(
            "  overall_health=%s | cpu=%s | mem=%s | pg=%s",
            result.get("overall_health"), result.get("cpu_health"),
            result.get("memory_health"), result.get("postgres_health"),
        )
        if result.get("status") == "ERROR":
            errors.append(f"cpu_agent: {result.get('message')}")
        return {**state, "cpu_data": result, "errors": errors}
    except Exception as e:
        msg = f"cpu_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {
            **state,
            "cpu_data": {
                "agent": "cpu_agent", "status": "ERROR", "message": msg,
                "overall_health": "UNKNOWN",
            },
            "errors": errors,
        }


def airflow_node(state: RCAState) -> RCAState:
    log.info("AIRFLOW NODE: Collecting log evidence")
    errors = list(state.get("errors", []))
    try:
        result = airflow_run(
            dag_id        =state["dag_id"],
            task_id       =state["task_id"],
            run_id        =state.get("run_id", ""),
            execution_date=state.get("execution_date", ""),
        )
        log.info("  error_type=%s | product_id=%s", result.get("error_type"), result.get("product_id"))
        if result.get("status") == "ERROR":
            errors.append(f"airflow_agent: {result.get('message')}")
        return {**state, "airflow_data": result, "errors": errors}
    except Exception as e:
        msg = f"airflow_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {
            **state,
            "airflow_data": {
                "agent": "airflow_agent", "status": "ERROR",
                "dag_id": state["dag_id"], "task_id": state["task_id"],
                "error_type": "Unknown", "product_id": None, "error_message": msg,
            },
            "errors": errors,
        }


def audit_node(state: RCAState) -> RCAState:
    log.info("AUDIT NODE: Querying etl_audit table")
    errors = list(state.get("errors", []))
    try:
        result = audit_run(dag_id=state["dag_id"], task_id=state["task_id"])
        log.info("  audit_id=%s | status=%s", result.get("audit_id"), result.get("status"))
        if result.get("status") == "ERROR":
            errors.append(f"audit_agent: {result.get('message')}")
        return {**state, "audit_data": result, "errors": errors}
    except Exception as e:
        msg = f"audit_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {**state, "audit_data": {"agent": "audit_agent", "status": "ERROR", "message": msg}, "errors": errors}


def postgres_node(state: RCAState) -> RCAState:
    log.info("POSTGRES NODE: Investigating product in database")
    errors     = list(state.get("errors", []))
    airflow_ev = state.get("airflow_data") or {}
    product_id = airflow_ev.get("product_id")
    if not product_id:
        log.warning("  No product_id from Airflow Agent")
    try:
        result = postgres_run(product_id=product_id)
        log.info("  product_id=%s | exists=%s | brand=%s", product_id, result.get("exists"), result.get("brand"))
        if result.get("status") == "ERROR":
            errors.append(f"postgres_agent: {result.get('message')}")
        return {**state, "postgres_data": result, "errors": errors}
    except Exception as e:
        msg = f"postgres_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {**state, "postgres_data": {"agent": "postgres_agent", "status": "ERROR", "message": msg}, "errors": errors}


def rag_node(state: RCAState) -> RCAState:
    log.info("RAG NODE: Searching historical incidents")
    errors        = list(state.get("errors", []))
    airflow_ev    = state.get("airflow_data") or {}
    perf_data     = state.get("performance_data") or {}
    error_type    = airflow_ev.get("error_type",    "Unknown")
    error_message = airflow_ev.get("error_message", "Unknown")

    # Pass current volume so RAG can filter similar volume incidents for benchmarking
    current_volume = int(perf_data.get("total_rows", 0))
    log.info("  current_volume=%d for benchmark filtering", current_volume)

    try:
        result = rag_run(
            error_type    =error_type,
            error_message =error_message,
            current_volume=current_volume,
        )
        log.info(
            "  incidents_found=%s | benchmarks=%s",
            result.get("incidents_found"),
            bool(result.get("benchmarks")),
        )
        if result.get("status") == "ERROR":
            errors.append(f"rag_agent: {result.get('message')}")
        return {**state, "rag_data": result, "errors": errors}
    except Exception as e:
        msg = f"rag_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {
            **state,
            "rag_data": {
                "agent"          : "rag_agent",
                "status"         : "ERROR",
                "incidents_found": 0,
                "incidents"      : [],
                "top_match"      : None,
                "benchmarks"     : {},
            },
            "errors": errors,
        }


def rca_node(state: RCAState) -> RCAState:
    log.info("RCA NODE: Generating Groq LLM Root Cause Analysis")
    errors      = list(state.get("errors", []))
    airflow_ev  = state.get("airflow_data")  or {"agent": "airflow_agent",  "status": "MISSING"}
    audit_ev    = state.get("audit_data")    or {"agent": "audit_agent",    "status": "MISSING"}
    postgres_ev = state.get("postgres_data") or {"agent": "postgres_agent", "status": "MISSING"}
    rag_ev      = state.get("rag_data")      or {
        "agent": "rag_agent", "status": "MISSING",
        "incidents_found": 0, "incidents": [], "top_match": None, "benchmarks": {},
    }
    cpu_ev      = state.get("cpu_data") or {"agent": "cpu_agent", "status": "MISSING", "overall_health": "UNKNOWN"}
    perf_data   = state.get("performance_data") or {}
    try:
        result = rca_run(
            airflow_evidence  =airflow_ev,
            audit_evidence    =audit_ev,
            postgres_evidence =postgres_ev,
            rag_evidence      =rag_ev,
            cpu_evidence      =cpu_ev,
            performance_data  =perf_data,
        )
        log.info("  RCA generated | incident_id=%s | confidence=%s", result.get("incident_id"), result.get("confidence_score"))
        if result.get("status") == "ERROR":
            errors.append(f"rca_agent: {result.get('message')}")
        return {**state, "final_rca": result, "workflow_status": "SUCCESS" if not errors else "PARTIAL", "errors": errors}
    except Exception as e:
        msg = f"rca_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {**state, "final_rca": {"status": "ERROR", "message": msg}, "workflow_status": "FAILED", "errors": errors}


def teams_summary_node(state: RCAState) -> RCAState:
    log.info("TEAMS SUMMARY NODE: Building structured Teams summary")
    errors = list(state.get("errors", []))
    try:
        enriched_rca = teams_summary_run(
            final_rca        =state.get("final_rca")        or {},
            airflow_evidence =state.get("airflow_data")     or {},
            audit_evidence   =state.get("audit_data")       or {},
            postgres_evidence=state.get("postgres_data")    or {},
            rag_evidence     =state.get("rag_data")         or {},
            cpu_evidence     =state.get("cpu_data")         or {},
            performance_data =state.get("performance_data") or {},
        )
        log.info("  teams_summary built for incident_id=%s", enriched_rca.get("incident_id"))
        return {**state, "final_rca": enriched_rca, "errors": errors}
    except Exception as e:
        msg = f"teams_summary_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {**state, "errors": errors}


def teams_notify_node(state: RCAState) -> RCAState:
    log.info("TEAMS NOTIFY NODE: Publishing Adaptive Card to Teams")
    errors    = list(state.get("errors", []))
    final_rca = state.get("final_rca")
    if not final_rca:
        msg = "teams_notify_node: no final_rca in state — skipping Teams publish"
        log.warning(msg)
        errors.append(msg)
        return {**state, "notification_sent": False, "errors": errors}
    try:
        notify_result = publish_autorca_card_to_teams(final_rca)
        status = notify_result.get("teams_notification_status", "unknown")
        log.info("  Teams notification status: %s", status)
        if status != "sent":
            errors.append(f"Teams publish returned unexpected status: {status}")
        return {**state, "notification_sent": status == "sent", "errors": errors}
    except Exception as e:
        msg = f"teams_notify_node crashed: {e}"
        log.error(msg, exc_info=True)
        errors.append(msg)
        return {**state, "notification_sent": False, "errors": errors}


def should_notify(state: RCAState) -> Literal["teams_summary", "end"]:
    if state.get("final_rca") and state.get("workflow_status") != "FAILED":
        return "teams_summary"
    log.warning("should_notify: skipping Teams — workflow_status=%s final_rca=%s",
                state.get("workflow_status"), bool(state.get("final_rca")))
    return "end"


def build_workflow() -> StateGraph:
    graph = StateGraph(RCAState)
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("cpu",           cpu_node)
    graph.add_node("airflow",       airflow_node)
    graph.add_node("audit",         audit_node)
    graph.add_node("postgres",      postgres_node)
    graph.add_node("rag",           rag_node)
    graph.add_node("rca",           rca_node)
    graph.add_node("teams_summary", teams_summary_node)
    graph.add_node("teams_notify",  teams_notify_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "cpu")
    graph.add_edge("cpu",        "airflow")
    graph.add_edge("airflow",    "audit")
    graph.add_edge("audit",      "postgres")
    graph.add_edge("postgres",   "rag")
    graph.add_edge("rag",        "rca")
    graph.add_conditional_edges("rca", should_notify, {"teams_summary": "teams_summary", "end": END})
    graph.add_edge("teams_summary", "teams_notify")
    graph.add_edge("teams_notify",  END)
    return graph.compile()


def run_rca_workflow(
    dag_id          : str,
    task_id         : str,
    run_id          : str  = "",
    execution_date  : str  = "",
    performance_data: dict = None,
) -> dict:
    log.info("Starting RCA Workflow | dag=%s task=%s run_id=%s", dag_id, task_id, run_id)
    workflow = build_workflow()
    initial_state: RCAState = {
        "dag_id"           : dag_id,
        "task_id"          : task_id,
        "run_id"           : run_id,
        "execution_date"   : execution_date,
        "performance_data" : performance_data or {},
        "run_airflow"      : False,
        "run_audit"        : False,
        "run_postgres"     : False,
        "run_rag"          : False,
        "run_cpu"          : False,
        "airflow_data"     : None,
        "audit_data"       : None,
        "postgres_data"    : None,
        "rag_data"         : None,
        "cpu_data"         : None,
        "final_rca"        : None,
        "notification_sent": False,
        "errors"           : [],
        "workflow_status"  : "PENDING",
    }
    final_state = workflow.invoke(initial_state)
    log.info("=" * 60)
    log.info("WORKFLOW COMPLETE")
    log.info("  Status         : %s", final_state.get("workflow_status"))
    log.info("  Incident ID    : %s", final_state.get("final_rca", {}).get("incident_id"))
    log.info("  Confidence     : %s", final_state.get("final_rca", {}).get("confidence_score"))
    log.info("  Teams notified : %s", final_state.get("notification_sent"))
    if final_state.get("errors"):
        log.warning("  Errors         : %s", final_state["errors"])
    log.info("=" * 60)
    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoRCA Workflow")
    parser.add_argument("--dag-id",         default="retail_product_etl")
    parser.add_argument("--task-id",        default="load_product_csv")
    parser.add_argument("--run-id",         default="")
    parser.add_argument("--execution-date", default="")
    args = parser.parse_args()
    result = run_rca_workflow(
        dag_id        =args.dag_id,
        task_id       =args.task_id,
        run_id        =args.run_id,
        execution_date=args.execution_date,
    )
    print("\n--- Final Workflow State ---")
    print(f"  Status         : {result.get('workflow_status')}")
    print(f"  Incident ID    : {result.get('final_rca', {}).get('incident_id')}")
    print(f"  Confidence     : {result.get('final_rca', {}).get('confidence_score')}")
    print(f"  Teams notified : {result.get('notification_sent')}")
    print(f"  Errors         : {result.get('errors')}")
