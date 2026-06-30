"""
RCA Workflow State
Strongly typed shared state object for LangGraph workflow.
All nodes read from and write to this state.
"""
from typing import TypedDict, Optional, List, Dict, Any

class RCAState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    dag_id          : str
    task_id         : str
    run_id          : str
    execution_date  : str
    # ── Performance Data ───────────────────────────────────
    performance_data: Optional[Dict[str, Any]]
    # ── Supervisor Routing Decisions ───────────────────────
    run_airflow     : bool
    run_audit       : bool
    run_postgres    : bool
    run_rag         : bool
    run_cpu         : bool
    # ── Agent Evidence ─────────────────────────────────────
    airflow_data    : Optional[Dict[str, Any]]
    audit_data      : Optional[Dict[str, Any]]
    postgres_data   : Optional[Dict[str, Any]]
    rag_data        : Optional[Dict[str, Any]]
    cpu_data        : Optional[Dict[str, Any]]
    # ── Final RCA ──────────────────────────────────────────
    final_rca       : Optional[Dict[str, Any]]
    # ── Notification ───────────────────────────────────────
    notification_sent : bool
    # ── Error Tracking ─────────────────────────────────────
    errors          : List[str]
    workflow_status : str   # RUNNING / SUCCESS / PARTIAL / FAILED
