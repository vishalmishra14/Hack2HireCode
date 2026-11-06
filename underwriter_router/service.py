"""Entry point used by the CLI now and by the API/event loop later."""
from __future__ import annotations

from .config import settings
from .db import Repository
from .graph import build_graph
from . import rules
from .models import AppStatus, DecisionSource, LoanApplicationIn, OverrideReason, RoutingResult


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

    # --- case lifecycle -------------------------------------------------------------
    def complete_case(self, app_id: str, sla_met: bool, turnaround_hours: float, reworked: bool = False) -> str:
        """Underwriter finished: frees capacity (emits CAPACITY_FREED) and records the learning outcome."""
        return self.repo.complete_case(app_id, sla_met, turnaround_hours, reworked)

    def override(
        self, app_id: str, to_underwriter_id: str, actor_id: str, reason_code: OverrideReason,
        reason_text: str, force: bool = False,
    ) -> RoutingResult:
        """Supervisor reassignment (spec §7.3). Reason is mandatory; target must pass hard filters unless
        force=True, which is itself recorded. Produces a new MANUAL_OVERRIDE decision in the audit trail."""
        if not reason_text or not reason_text.strip():
            raise ValueError("override reason_text is mandatory")
        reason_code = OverrideReason(reason_code)
        app = self.repo.get_application(app_id)
        if app is None or app.status != AppStatus.ASSIGNED:
            raise ValueError(f"only ASSIGNED applications can be overridden (got {app and app.status.value})")
        target = next((u for u in self.repo.list_underwriters() if u.id == to_underwriter_id), None)
        if target is None:
            raise ValueError(f"unknown underwriter {to_underwriter_id}")
        problems = rules.check_eligibility(app, target)
        if problems and not force:
            raise ValueError(f"{to_underwriter_id} is not eligible: {', '.join(problems)}")
        prev = self.repo.get_latest_decision(app_id)
        rationale = (f"Reassigned from {prev['selected_underwriter_id']} to {target.name} ({target.id}) by "
                     f"supervisor {actor_id}. Reason [{reason_code.value}]: {reason_text.strip()}"
                     + (f" FORCED despite: {', '.join(problems)}." if problems else ""))
        decision = {
            "input_snapshot": prev["input_snapshot"], "rules_version": prev["rules_version"],
            "weights_version": prev["weights_version"], "complexity": "override",
            "eligible": prev["eligible_candidates"], "excluded": prev["excluded_candidates"],
            "shortlist": prev["shortlist"], "verification": [], "selected_underwriter_id": target.id,
            "decision_source": DecisionSource.MANUAL_OVERRIDE.value, "rationale": rationale,
            "fairness_signals": {"protected_attrs_used": False, "override": True, "forced": bool(problems),
                                 "eligibility_issues": problems},
            "latency_ms": 0,
        }
        self.repo.override_assignment(app_id, target.id, actor_id, reason_code.value, reason_text.strip(),
                                      forced=bool(problems), decision=decision)
        return self.get_result(app_id)

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
