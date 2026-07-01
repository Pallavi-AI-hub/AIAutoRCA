"""
RCA Workflow State
Strongly typed shared state object for LangGraph workflow.
All nodes read from and write to this state.
""" 

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator

class RCAState(TypedDict):
    dag_id          : str
    task_id         : str
    run_id          : str
    execution_date  : str
    performance_data: Optional[Dict[str, Any]]
    run_airflow     : bool
    run_audit       : bool
    run_postgres    : bool
    run_rag         : bool
    run_cpu         : bool
    run_pg_logs     : bool
    airflow_data    : Optional[Dict[str, Any]]
    audit_data      : Optional[Dict[str, Any]]
    postgres_data   : Optional[Dict[str, Any]]
    rag_data        : Optional[Dict[str, Any]]
    cpu_data        : Optional[Dict[str, Any]]
    pg_logs_data    : Optional[Dict[str, Any]]
    final_rca       : Optional[Dict[str, Any]]
    notification_sent: bool
    errors          : Annotated[List[str], operator.add]
    workflow_status : str
