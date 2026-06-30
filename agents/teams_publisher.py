from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()

class TeamsPublishError(RuntimeError):
    pass

@dataclass(frozen=True)
class TeamsConfig:
    enabled: bool
    team_name: str
    channel_name: str
    webhook_url: str
    timeout_seconds: int
    max_retries: int
    debug: bool

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("TEAMS_RCA_ENABLED","true").strip().lower() in {"1","true","yes","y"},
            team_name=os.getenv("TEAMS_RCA_TEAM_NAME","").strip(),
            channel_name=os.getenv("TEAMS_RCA_CHANNEL_NAME","").strip(),
            webhook_url=os.getenv("TEAMS_RCA_WEBHOOK_URL","").strip(),
            timeout_seconds=int(os.getenv("TEAMS_RCA_TIMEOUT_SECONDS","15")),
            max_retries=int(os.getenv("TEAMS_RCA_MAX_RETRIES","3")),
            debug=os.getenv("TEAMS_RCA_DEBUG","false").strip().lower() in {"1","true","yes","y"},
        )

    def validate(self):
        if not self.enabled: return
        if not self.webhook_url: raise TeamsPublishError("TEAMS_RCA_WEBHOOK_URL is missing.")
        if not self.webhook_url.startswith("https://"): raise TeamsPublishError("TEAMS_RCA_WEBHOOK_URL must be HTTPS.")

def _byte_len(v): return len(v.encode("utf-8"))

def _safe(value, default="Unknown", max_chars=450):
    if value is None: return default
    text = str(value).strip()
    if not text: return default
    text = "\n".join(line.strip() for line in text.replace("\r\n","\n").replace("\r","\n").split("\n") if line.strip())
    return text[:max_chars-3].rstrip()+"..." if len(text)>max_chars else text

def _fact(title, value): return {"title": title, "value": _safe(value)}

def _section(title, facts):
    return [
        {"type":"TextBlock","text":title,"weight":"Bolder","size":"Medium","spacing":"Medium","separator":True,"wrap":True},
        {"type":"FactSet","facts":facts},
    ]

def _get_summary(result):
    summary = result.get("teams_summary")
    if isinstance(summary, dict): return summary
    return {k:"Unknown" for k in [
        "incident_id","dag","task","severity","status","confidence",
        "error","product_id","constraint","cause",
        "airflow_logs","postgresql_check","historical_cases",
        "sla_analysis","performance_insights","system_health",
        "product_load","catalog_update","downstream_feeds",
        "remove_duplicate_record","reload_source_file","rerun_dag",
        "prevention_high_1","prevention_high_2","prevention_medium_1",
        "final_verdict",
    ]}

def build_autorca_summary_card(result):
    s = _get_summary(result)

    body = [
        {"type":"TextBlock","text":"🚨 AutoRCA Summary","weight":"Bolder","size":"Large","wrap":True},
        {"type":"FactSet","facts":[
            _fact("Incident ID", s.get("incident_id")),
            _fact("DAG",         s.get("dag")),
            _fact("Task",        s.get("task")),
            _fact("Severity",    s.get("severity")),
            _fact("Status",      s.get("status")),
            _fact("Confidence",  s.get("confidence")),
        ]},
    ]

    # Root Cause
    body.extend(_section("Root Cause", [
        _fact("Error",      s.get("error")),
        _fact("Product ID", s.get("product_id")),
        _fact("Constraint", s.get("constraint")),
        _fact("Cause",      s.get("cause")),
    ]))

    # Evidence
    body.extend(_section("Evidence", [
        _fact("Airflow Logs",     s.get("airflow_logs")),
        _fact("PostgreSQL Check", s.get("postgresql_check")),
        _fact("Historical Cases", s.get("historical_cases")),
    ]))

    # SLA Analysis
    body.extend(_section("SLA Analysis", [
        _fact("SLA Summary", s.get("sla_analysis")),
    ]))

    # Performance Insights — only show if data available
    perf = s.get("performance_insights", "")
    if perf and perf not in ("Unknown", "No performance data available"):
        body.extend(_section("⚡ Performance Insights", [
            _fact("Runtime Intelligence", perf),
        ]))

    # System Health — only show if data available
    sys_health = s.get("system_health", "")
    if sys_health and sys_health not in ("Unknown", "System health check unavailable"):
        body.extend(_section("🖥️ System Health", [
            _fact("Infrastructure Status", sys_health),
        ]))

    # Impact
    body.extend(_section("Impact", [
        _fact("Product Load",     s.get("product_load")),
        _fact("Catalog Update",   s.get("catalog_update")),
        _fact("Downstream Feeds", s.get("downstream_feeds")),
    ]))

    # Resolution
    body.extend(_section("Resolution", [
        _fact("Remove Duplicate Record", s.get("remove_duplicate_record")),
        _fact("Reload Source File",      s.get("reload_source_file")),
        _fact("Rerun DAG",               s.get("rerun_dag")),
    ]))

    # Prevention
    body.extend(_section("Prevention", [
        _fact("High",   s.get("prevention_high_1")),
        _fact("High",   s.get("prevention_high_2")),
        _fact("Medium", s.get("prevention_medium_1")),
    ]))

    # Final Verdict
    body.extend([
        {"type":"TextBlock","text":"Final Verdict","weight":"Bolder","size":"Medium","spacing":"Medium","separator":True,"wrap":True},
        {"type":"TextBlock","text":_safe(s.get("final_verdict"), max_chars=900),"wrap":True},
    ])

    return {
        "$schema" : "http://adaptivecards.io/schemas/adaptive-card.json",
        "type"    : "AdaptiveCard",
        "version" : "1.4",
        "msteams" : {"width": "Full"},
        "body"    : body,
    }

def build_teams_webhook_card_payload(card):
    return {
        "type"       : "message",
        "card_json"  : json.dumps(card, ensure_ascii=False),
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "contentUrl": None, "content": card}],
    }

def _post_payload_to_teams_webhook(config, payload):
    body       = json.dumps(payload, ensure_ascii=False)
    last_error = None
    for attempt in range(1, config.max_retries + 1):
        try:
            r = requests.post(
                config.webhook_url,
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=body.encode("utf-8"),
                timeout=config.timeout_seconds,
            )
            if config.debug:
                print("Status:", r.status_code)
                print("Body:",   r.text[:500])
            if 200 <= r.status_code < 300: return
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise TeamsPublishError(f"Webhook rejected. Status={r.status_code} Body={r.text[:500]}")
        except requests.RequestException as e:
            last_error = e
            if attempt < config.max_retries:
                time.sleep(min(2 ** attempt, 30))
    raise TeamsPublishError(f"Failed after {config.max_retries} attempts: {last_error}")

def publish_autorca_card_to_teams(result, config=None):
    cfg = config or TeamsConfig.from_env()
    if not cfg.enabled:
        return {
            "teams_notification_status": "disabled",
            "teams_team_name"          : cfg.team_name,
            "teams_channel_name"       : cfg.channel_name,
            "teams_payload_type"       : "adaptive_card",
        }
    cfg.validate()
    card    = build_autorca_summary_card(result)
    payload = build_teams_webhook_card_payload(card)
    _post_payload_to_teams_webhook(cfg, payload)
    body = json.dumps(payload, ensure_ascii=False)
    return {
        "teams_notification_status": "sent",
        "teams_team_name"          : cfg.team_name,
        "teams_channel_name"       : cfg.channel_name,
        "teams_payload_type"       : "adaptive_card",
        "teams_message_parts"      : 1,
        "teams_message_bytes"      : _byte_len(body),
    }

def publish_full_rca_to_teams(result, config=None):
    return publish_autorca_card_to_teams(result=result, config=config)
