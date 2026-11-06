"""Event-driven loop (spec §9.1): transactional outbox -> dispatcher -> idempotent handlers.

Guarantees
  - No lost events      : events are written in the same transaction as the state change (outbox).
  - No missed routing   : APPLICATION_SUBMITTED is written at intake; sweeper re-queues expired ROUTING leases.
  - No duplicate effects: at-least-once delivery + idempotent handlers (routing claim, notification dedup).
  - Poison events       : exponential backoff, then dead-letter (status FAILED) after N attempts.

    run_once():  sweeper  ->  claim due events  ->  handler  ->  ack | nack(backoff / dead-letter)
"""
from __future__ import annotations

import logging
from typing import Callable

from langsmith import traceable

from . import rules
from .config import settings
from .db import Repository
from .models import AppStatus
from .service import RouterService

log = logging.getLogger("underwriter_router.events")


class EventLoop:
    def __init__(self, repo: Repository, service: RouterService, notify: Callable[[str, str], None] | None = None,
                 routing_lease_s: float | None = None):
        self.repo = repo
        self.service = service
        self.routing_lease_s = settings.routing_lease_s if routing_lease_s is None else routing_lease_s
        self.notify = notify or (lambda recipient, msg: log.info("notify %s: %s", recipient, msg))
        self.handlers: dict[str, Callable[[dict], None]] = {
            "APPLICATION_SUBMITTED": self.on_application_submitted,
            "UNDERWRITER_ASSIGNED": self.on_underwriter_assigned,
            "SUBMITTER_NOTIFIED": self.on_submitter_notified,
            "SUPERVISOR_ALERT_UNROUTABLE": self.on_unroutable,
            "ASSIGNMENT_OVERRIDDEN": self.on_override,
            "CASE_COMPLETED": lambda ev: None,  # outcome already persisted; consumed by the learning loop
            "CAPACITY_FREED": self.on_capacity_freed,
        }

    # --- loop ------------------------------------------------------------------------
    def run_once(self, batch: int = 50) -> dict:
        requeued = self.repo.expire_routing_leases(self.routing_lease_s)
        stats = {"requeued": len(requeued), "processed": 0, "retried": 0, "dead_lettered": 0}
        for ev in self.repo.claim_events(batch, settings.event_visibility_s):
            try:
                self._dispatch(ev)
                self.repo.ack_event(ev["id"])
                stats["processed"] += 1
            except Exception as e:  # noqa: BLE001
                status = self.repo.nack_event(ev["id"], f"{type(e).__name__}: {e}", ev["attempts"],
                                              settings.event_max_attempts, settings.event_backoff_s)
                stats["dead_lettered" if status == "FAILED" else "retried"] += 1
        return stats

    def drain(self, max_rounds: int = 20) -> dict:
        """Run until no due events remain (demo/tests)."""
        total = {"requeued": 0, "processed": 0, "retried": 0, "dead_lettered": 0}
        for _ in range(max_rounds):
            s = self.run_once()
            for k in total:
                total[k] += s[k]
            if s["processed"] + s["retried"] + s["dead_lettered"] + s["requeued"] == 0:
                break
        return total

    def _dispatch(self, ev: dict) -> None:
        handler = self.handlers.get(ev["event_type"])
        if handler is None:
            raise ValueError(f"no handler for {ev['event_type']}")
        traceable(name=f"event:{ev['event_type']}", metadata={"event_id": ev["id"]})(handler)(ev)

    def _send_once(self, ev: dict, recipient: str, message: str) -> None:
        if self.repo.record_notification(ev["id"], recipient, message):  # dedup on event id
            self.notify(recipient, message)

    # --- handlers (all idempotent) -------------------------------------------------------
    def on_application_submitted(self, ev: dict) -> None:
        app = self.repo.get_application(ev["payload"]["application_id"])
        if app and app.status == AppStatus.SUBMITTED:
            self.service.route(app.id)  # claim inside route() makes a second delivery a no-op

    def on_underwriter_assigned(self, ev: dict) -> None:
        p = ev["payload"]
        self._send_once(ev, f"underwriter:{p['underwriter_id']}", f"New application {p['application_id']} assigned")

    def on_submitter_notified(self, ev: dict) -> None:
        p = ev["payload"]
        self._send_once(ev, f"submitter:{p['submitter_id']}",
                        f"Application {p['application_id']} routed. Why: {p['rationale']}")

    def on_unroutable(self, ev: dict) -> None:
        self._send_once(ev, "supervisor", f"Application {ev['payload']['application_id']} is UNROUTABLE")

    def on_override(self, ev: dict) -> None:
        p = ev["payload"]
        self._send_once(ev, f"underwriter:{p['to_underwriter_id']}",
                        f"Application {p['application_id']} reassigned to you ({p['reason_code']}): {p['reason_text']}")

    def on_capacity_freed(self, ev: dict) -> None:
        """Capacity opened up: re-route the oldest UNROUTABLE application this underwriter can take."""
        uw = next((u for u in self.repo.list_underwriters() if u.id == ev["payload"]["underwriter_id"]), None)
        if uw is None:
            return
        for app_id in self.repo.list_application_ids(AppStatus.UNROUTABLE):
            app = self.repo.get_application(app_id)
            if not rules.check_eligibility(app, uw):
                self.service.route(app_id)  # full graph run: ranking, GenAI, verification, audit
                return
