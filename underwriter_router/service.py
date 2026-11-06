"""Entry point used by the CLI now and by the API/event loop later."""
from __future__ import annotations

from .config import settings
from .db import Repository
from .graph import build_graph
from .models import AppStatus, LoanApplicationIn, RoutingResult


class RouterService:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.graph = build_graph(repo)

    def submit(self, app_in: LoanApplicationIn) -> RoutingResult:
        """Idempotent submit + route. Re-submitting the same key returns the original outcome."""
        app, created = self.repo.create_application(app_in, settings.sla_hours[app_in.priority.value])
        if not created:
            return self.get_result(app.id, duplicate=True)
        return self.route(app.id)

    def route(self, app_id: str) -> RoutingResult:
        if not self.repo.claim_for_routing(app_id):
            # Someone else is routing (or already routed) it - never route twice.
            return self.get_result(app_id, duplicate=True)
        app = self.repo.get_application(app_id)
        out = self.graph.invoke(
            {"application_id": app_id, "llm_errors": [], "verification_log": []},
            config={
                "run_name": "route_application",
                "tags": ["underwriter-routing", settings.rules_version],
                "metadata": {
                    "application_id": app_id, "loan_type": app.loan_type.value,
                    "priority": app.priority.value, "risk_band": app.risk_band.value,
                },
            },
        )
        r = out["result"]
        return RoutingResult(
            application_id=app_id, status=AppStatus(r["status"]), underwriter_id=r.get("underwriter_id"),
            underwriter_name=r.get("underwriter_name"), decision_source=r.get("decision_source"),
            rationale=r["rationale"], complexity=out.get("complexity"),
            fairness_signals=r.get("fairness_signals", {}), decision_id=r.get("decision_id"),
        )

    def get_result(self, app_id: str, duplicate: bool = False) -> RoutingResult:
        app = self.repo.get_application(app_id)
        d = self.repo.get_latest_decision(app_id)
        names = {u.id: u.name for u in self.repo.list_underwriters()}
        uw = d.get("selected_underwriter_id") if d else None
        return RoutingResult(
            application_id=app_id, status=app.status, underwriter_id=uw, underwriter_name=names.get(uw),
            decision_source=d.get("decision_source") if d else None,
            rationale=d["rationale"] if d else f"Application is {app.status.value}; no decision yet.",
            complexity=(d.get("complexity") or "").split(":")[0] or None if d else None,
            fairness_signals=d.get("fairness_signals", {}) if d else {},
            decision_id=d["id"] if d else None, duplicate=duplicate,
        )
