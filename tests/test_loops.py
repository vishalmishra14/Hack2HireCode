"""Event-driven loop and self-improvement loop. No network: rules-only routing."""
from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from underwriter_router import events as events_mod
from underwriter_router import graph as graph_mod
from underwriter_router import learning
from underwriter_router.config import WEIGHT_KEYS, settings
from underwriter_router.db import Repository
from underwriter_router.events import EventLoop
from underwriter_router.models import AppStatus, LoanApplicationIn
from underwriter_router.seed import seed
from underwriter_router.service import RouterService


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "settings", replace(settings, llm_enabled=False))
    repo = Repository(tmp_path / "t.db")
    seed(repo)
    svc = RouterService(repo)
    sent = []
    loop = EventLoop(repo, svc, notify=lambda who, msg: sent.append((who, msg)))
    return repo, svc, loop, sent


def home(key, **kw):
    return LoanApplicationIn(**{"idempotency_key": key, "submitter_id": "s1", "loan_type": "HOME",
                                "amount": 300_000, "region": "TX", **kw})


def count(repo, sql):
    with sqlite3.connect(repo.db_path) as c:
        return c.execute(sql).fetchone()[0]


# --- event-driven loop -------------------------------------------------------------------
def test_intake_event_routes_even_if_process_died_before_routing(env):
    repo, svc, loop, sent = env
    app, _ = repo.create_application(home("k"), 72)  # crash: never routed inline
    loop.drain()
    assert repo.get_application(app.id).status == AppStatus.ASSIGNED
    whos = {w.split(":")[0] for w, _ in sent}
    assert whos == {"underwriter", "submitter"}  # assignment + rationale delivered


def test_inline_routing_then_event_is_a_noop(env):
    repo, svc, loop, _ = env
    r = svc.submit(home("k"))
    loop.drain()
    assert count(repo, "SELECT COUNT(*) FROM routing_decisions") == 1
    assert count(repo, "SELECT COUNT(*) FROM assignments") == 1
    assert repo.get_active_assignment(r.application_id)["underwriter_id"] == r.underwriter_id


def test_duplicate_event_delivery_sends_one_notification(env):
    repo, svc, loop, sent = env
    svc.submit(home("k"))
    ev = next(e for e in repo.claim_events(50, 0) if e["event_type"] == "SUBMITTER_NOTIFIED")
    loop._dispatch(ev)
    loop._dispatch(ev)  # redelivered (e.g. worker died before ack)
    assert sum(w.startswith("submitter") for w, _ in sent) == 1


def test_sweeper_recovers_expired_routing_lease(env):
    repo, svc, loop, _ = env
    app, _ = repo.create_application(home("k"), 72)
    repo.claim_for_routing(app.id)  # worker claims and dies
    loop.drain()
    assert repo.get_application(app.id).status == AppStatus.ROUTING  # lease still valid
    EventLoop(repo, svc, notify=lambda *_: None, routing_lease_s=0).drain()
    assert repo.get_application(app.id).status == AppStatus.ASSIGNED


def test_failing_handler_retries_then_dead_letters(env, monkeypatch):
    repo, svc, loop, _ = env
    monkeypatch.setattr(events_mod, "settings", replace(settings, event_backoff_s=0, event_max_attempts=3))
    svc.submit(home("k"))

    def boom(ev):
        raise RuntimeError("smtp down")

    loop.handlers["SUBMITTER_NOTIFIED"] = boom
    loop.drain()
    assert repo.event_counts().get("FAILED") == 1
    assert count(repo, "SELECT attempts FROM outbox_events WHERE status='FAILED'") == 3


def test_capacity_freed_reroutes_unroutable(env):
    repo, svc, loop, _ = env
    results = [svc.submit(home(f"f{i}", amount=1_800_000)) for i in range(20)]  # capacity for 19
    blocked = results[-1]
    assert blocked.status == AppStatus.UNROUTABLE
    loop.drain()
    svc.complete_case(results[0].application_id, sla_met=True, turnaround_hours=20)
    loop.drain()
    r = svc.get_result(blocked.application_id)
    assert r.status == AppStatus.ASSIGNED and r.underwriter_id == results[0].underwriter_id


def test_override_requires_reason_checks_eligibility_and_is_audited(env):
    repo, svc, loop, sent = env
    r = svc.submit(home("k"))
    target = next(u.id for u in repo.list_underwriters() if u.id in {"UW-001", "UW-006"} and u.id != r.underwriter_id)
    with pytest.raises(ValueError):
        svc.override(r.application_id, target, "sup", "WORKLOAD", "  ")
    with pytest.raises(ValueError):  # UW-004 has no HOME skill
        svc.override(r.application_id, "UW-004", "sup", "EXPERTISE", "needs specialist")
    o = svc.override(r.application_id, target, "sup", "WORKLOAD", "rebalance queue")
    assert o.underwriter_id == target and o.decision_source == "MANUAL_OVERRIDE"
    assert "rebalance queue" in o.rationale
    loads = {u.id: u.active_cases for u in repo.list_underwriters()}
    assert loads[target] == 1 and loads[r.underwriter_id] == 0
    loop.drain()
    assert any(w == f"underwriter:{target}" and "rebalance queue" in m for w, m in sent)
    forced = svc.override(r.application_id, "UW-004", "sup", "OTHER", "exception approved", force=True)
    assert forced.fairness_signals["forced"] is True


# --- self-improvement loop -------------------------------------------------------------------
def _traffic(svc, repo, n=60):
    """Overloaded underwriters miss SLA: a signal the loop should pick up."""
    for i in range(n):
        r = svc.submit(home(f"t{i}", amount=200_000 + i * 1000))
        uw = next(u for u in repo.list_underwriters() if u.id == r.underwriter_id)
        busy = uw.active_cases / uw.max_active_cases > 0.4
        svc.complete_case(r.application_id, sla_met=not busy, turnaround_hours=30, reworked=False) if i % 2 else None


def test_learning_skips_without_enough_outcomes(env):
    repo, svc, *_ = env
    assert learning.run_cycle(repo)["status"] == "SKIPPED"


def test_learning_cycle_is_bounded_gated_versioned_and_reversible(env):
    repo, svc, *_ = env
    _traffic(svc, repo)
    before = repo.get_active_weights()
    rep = learning.run_cycle(repo)
    assert rep["status"] in {"ACCEPTED", "REJECTED"}
    for k in WEIGHT_KEYS:  # bounded update
        assert abs(rep["weights_after"][k] - rep["weights_before"][k]) <= settings.learn_max_delta + 1e-6
    assert abs(sum(rep["weights_after"].values()) - 1) < 1e-3
    if rep["status"] == "ACCEPTED":  # gate
        assert rep["train"]["after"] > rep["train"]["before"]
        assert rep["holdout"]["after"] > rep["holdout"]["before"]
        assert repo.get_active_weights().version == rep["new_version"]
        repo.activate_weights(before.version)  # rollback
        assert repo.get_active_weights() == before
    assert rep["performance_updates"]  # metrics learned from outcomes


def test_new_decisions_record_the_weights_version_used(env):
    repo, svc, *_ = env
    from underwriter_router.config import Weights
    repo.save_weights(Weights(version="w-test", skill=0.2, load=0.4, sla=0.2, speed=0.1, fairness=0.1),
                      parent="w-0.1", train=0, holdout=0, notes={}, activate=True)
    r = svc.submit(home("k"))
    assert repo.get_latest_decision(r.application_id)["weights_version"] == "w-test"
