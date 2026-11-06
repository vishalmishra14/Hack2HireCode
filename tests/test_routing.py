"""Acceptance tests for the base routing loop (spec §12). No network: LLM calls are stubbed."""
from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from underwriter_router import graph as graph_mod
from underwriter_router import optimizer, rules, verifier
from underwriter_router.config import settings, weights
from underwriter_router.db import Repository
from underwriter_router.models import LLMChoice, LoanApplicationIn
from underwriter_router.seed import SEED_UNDERWRITERS, seed
from underwriter_router.service import RouterService


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / "test.db")
    seed(r)
    return r


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(graph_mod, "settings", replace(settings, llm_enabled=False))


def home_app(key="k1", **kw) -> LoanApplicationIn:
    base = dict(idempotency_key=key, submitter_id="s1", loan_type="HOME", amount=300_000, region="TX")
    return LoanApplicationIn(**{**base, **kw})


def stub_choice(pick_index: int = 0, confidence: float = 0.9):
    def _f(payload, *a, **k):
        c = payload["shortlist"][pick_index]
        return LLMChoice(selected_underwriter_id=c["underwriter_id"], confidence=confidence,
                         rationale=f"Stub picked {c['underwriter_id']}."), "stub"
    return _f


def fail(*a, **k):
    raise optimizer.OptimizerError("boom")


# --- rules engine ---------------------------------------------------------------
def test_ranking_is_deterministic(repo):
    app, _ = repo.create_application(home_app(), 72)
    uws = repo.list_underwriters()
    elig, _ = rules.filter_eligible(app, uws)
    r1 = rules.rank(app, elig, {}, 50, weights)
    r2 = rules.rank(app, list(reversed(elig)), {}, 50, weights)
    assert [c.underwriter_id for c in r1] == [c.underwriter_id for c in r2]


def test_hard_filters_record_reasons(repo):
    app, _ = repo.create_application(home_app(amount=1_800_000), 72)
    elig, excl = rules.filter_eligible(app, repo.list_underwriters())
    ids = {u.id for u in elig}
    assert ids == {"UW-001", "UW-006"}  # only these have HOME + TX + authority >= 1.8M
    reasons = {e.underwriter_id: e.reasons for e in excl}
    assert any("status is ON_LEAVE" in r for r in reasons["UW-008"])
    assert any("authority limit" in r for r in reasons["UW-002"])


def test_llm_payload_has_no_protected_or_identifying_fields(repo):
    app, _ = repo.create_application(home_app(), 72)
    elig, _ = rules.filter_eligible(app, repo.list_underwriters())
    from datetime import datetime, timezone
    payload = rules.build_llm_payload(app, rules.rank(app, elig, {}, 50, weights), datetime.now(timezone.utc))
    blob = json.dumps(payload).lower()
    assert set(payload["application"]) == set(rules.LLM_APPLICATION_FIELDS) | {"hours_until_sla"}
    for attr in rules.PROTECTED_ATTRIBUTES + ("submitter_id", "idempotency_key"):
        assert f'"{attr}"' not in blob


def test_complexity_classification(repo):
    simple, _ = repo.create_application(home_app("a"), 72)
    cx, _ = repo.create_application(home_app("b", risk_band="HIGH", complexity_flags=["SELF_EMPLOYED", "MULTI_PROPERTY"]), 72)
    for app, expected in ((simple, "simple"), (cx, "complex")):
        elig, _ = rules.filter_eligible(app, repo.list_underwriters())
        label, reasons = rules.classify_complexity(app, rules.rank(app, elig, {}, 50, weights), settings)
        assert label == expected
    assert "high risk band" in reasons


# --- routing loop ------------------------------------------------------------------
def test_simple_path_uses_llm_choice(repo, monkeypatch):
    monkeypatch.setattr(optimizer, "optimize_simple", stub_choice(pick_index=1))
    monkeypatch.setattr(graph_mod, "settings", replace(settings, close_call_score_gap=0.0))
    r = RouterService(repo).submit(home_app())
    assert r.status.value == "ASSIGNED"
    assert r.decision_source == "GENAI_SIMPLE"
    assert r.fairness_signals["rank_in_rules"] == 2
    assert r.rationale.startswith("Stub picked")
    assert repo.get_latest_decision(r.application_id)["verification"][0]["passed"]


def test_complex_falls_back_to_simple_then_rules(repo, monkeypatch):
    cx = home_app(risk_band="HIGH")
    monkeypatch.setattr(optimizer, "optimize_complex", fail)
    monkeypatch.setattr(optimizer, "optimize_simple", stub_choice())
    r = RouterService(repo).submit(cx)
    assert r.decision_source == "GENAI_SIMPLE"

    monkeypatch.setattr(optimizer, "optimize_simple", fail)
    r2 = RouterService(repo).submit(home_app("k2", risk_band="HIGH"))
    assert r2.status.value == "ASSIGNED" and r2.decision_source == "RULES_FALLBACK"
    d = repo.get_latest_decision(r2.application_id)
    assert len(d["llm_errors"]) == 2


# --- verifier -----------------------------------------------------------------------
def _ctx(repo, **kw):
    app, _ = repo.create_application(home_app("v", **kw), 72)
    elig, _ = rules.filter_eligible(app, repo.list_underwriters())
    return app, rules.rank(app, elig, {}, 50, weights)


def _v(repo, app, ranked, uid, rationale, confidence=0.9):
    c = LLMChoice(selected_underwriter_id=uid, confidence=confidence, rationale=rationale)
    return verifier.verify(c, app, ranked[:5], repo.list_underwriters(), attempt=1)


def test_verifier_accepts_grounded_rationale(repo):
    app, ranked = _ctx(repo)
    top = ranked[0]
    pct = round(top.sla_hit_rate * 100)
    v = _v(repo, app, ranked, top.underwriter_id,
           f"{top.underwriter_id} has the best fit, {top.active_cases}/{top.max_active_cases} open cases and a {pct}% SLA hit rate.")
    assert v.passed, v.issues


@pytest.mark.parametrize("case,check", [
    ("outside_shortlist", "in_shortlist"),
    ("low_confidence", "confidence"),
    ("pronoun", "neutral_language"),
    ("bogus_flag", "flags_grounded"),
    ("bogus_number", "numbers_grounded"),
    ("wrong_name", "names_selected"),
    ("foreign_name", "no_foreign_names"),
])
def test_verifier_rejects(repo, case, check):
    app, ranked = _ctx(repo)
    top = ranked[0]
    uid, conf, text = top.underwriter_id, 0.9, f"{top.underwriter_id} is the best fit."
    if case == "outside_shortlist":
        uid = "UW-999"
    elif case == "low_confidence":
        conf = 0.2
    elif case == "pronoun":
        text = f"{top.underwriter_id} is chosen because she is fast."
    elif case == "bogus_flag":
        text = f"{top.underwriter_id} handles the foreign income on this file."
    elif case == "bogus_number":
        text = f"{top.underwriter_id} has a 42% SLA hit rate."
    elif case == "wrong_name":
        text = "This underwriter is the best fit."
    elif case == "foreign_name":
        text = f"{top.underwriter_id} beats Ivan Petrov here."  # UW-009 not eligible for HOME
    v = _v(repo, app, ranked, uid, text, conf)
    assert not v.passed and v.checks[check] is False


def test_verifier_rejects_pick_that_became_ineligible(repo):
    app, ranked = _ctx(repo)
    top = ranked[0]
    uw = next(u for u in repo.list_underwriters() if u.id == top.underwriter_id)
    repo.upsert_underwriter(uw.model_copy(update={"status": "OFFLINE"}))  # changed during the LLM call
    v = _v(repo, app, ranked, top.underwriter_id, f"{top.underwriter_id} is the best fit.")
    assert v.checks["still_eligible"] is False


def test_verifier_rejects_large_rank_deviation(repo, monkeypatch):
    app, ranked = _ctx(repo)
    monkeypatch.setattr(verifier, "settings", replace(settings, max_score_deviation=0.0))
    low = ranked[1]
    v = _v(repo, app, ranked, low.underwriter_id, f"{low.underwriter_id} is the best fit.")
    assert v.checks["rank_deviation"] is False


# --- verification loop -------------------------------------------------------------
def test_loop_retries_with_feedback_then_passes(repo, monkeypatch):
    calls = []

    def flaky(payload, *a, feedback=None, **k):
        calls.append(feedback)
        c = payload["shortlist"][0]
        text = f"{c['underwriter_id']} is fast; she is great." if not feedback else f"{c['underwriter_id']} is the best fit."
        return LLMChoice(selected_underwriter_id=c["underwriter_id"], confidence=0.9, rationale=text), "stub"

    monkeypatch.setattr(optimizer, "optimize_simple", flaky)
    r = RouterService(repo).submit(home_app())
    assert r.decision_source == "GENAI_SIMPLE"
    assert not calls[0] and any("gendered" in i for i in calls[1])
    d = repo.get_latest_decision(r.application_id)
    assert [v["passed"] for v in d["verification"]] == [False, True]


def test_loop_falls_back_to_rules_after_max_attempts(repo, monkeypatch):
    def always_bad(payload, *a, **k):
        c = payload["shortlist"][0]
        return LLMChoice(selected_underwriter_id=c["underwriter_id"], confidence=0.9,
                         rationale=f"{c['underwriter_id']}: his record is 42% better."), "stub"

    monkeypatch.setattr(optimizer, "optimize_simple", always_bad)
    r = RouterService(repo).submit(home_app())
    assert r.status.value == "ASSIGNED" and r.decision_source == "RULES_FALLBACK"
    assert "failed verification 2x" in r.rationale
    d = repo.get_latest_decision(r.application_id)
    assert len(d["verification"]) == settings.max_verification_attempts
    assert all(not v["passed"] for v in d["verification"])


def test_loop_terminates_when_retry_errors(repo, monkeypatch):
    calls = []

    def bad_then_crash(payload, *a, feedback=None, **k):
        calls.append(1)
        if feedback:
            raise optimizer.OptimizerError("down")
        c = payload["shortlist"][0]
        return LLMChoice(selected_underwriter_id=c["underwriter_id"], confidence=0.1, rationale=c["underwriter_id"]), "stub"

    monkeypatch.setattr(optimizer, "optimize_simple", bad_then_crash)
    r = RouterService(repo).submit(home_app())
    assert r.decision_source == "RULES_FALLBACK" and len(calls) == 2


def test_idempotent_submission(repo, no_llm):
    svc = RouterService(repo)
    r1 = svc.submit(home_app())
    r2 = svc.submit(home_app())
    assert r1.application_id == r2.application_id and r2.duplicate
    assert r1.underwriter_id == r2.underwriter_id
    assert sum(u.active_cases for u in repo.list_underwriters()) == 1


def test_unroutable_is_recorded(repo, no_llm):
    r = RouterService(repo).submit(home_app(amount=50_000_000))
    assert r.status.value == "UNROUTABLE"
    d = repo.get_latest_decision(r.application_id)
    assert d["selected_underwriter_id"] is None and d["excluded_candidates"]


def test_concurrent_routing_never_double_assigns_or_overfills(repo, no_llm):
    svc = RouterService(repo)
    n = 40  # HOME/TX capacity across eligible underwriters is 45
    results, errors = [], []

    def worker(i):
        try:
            results.append(svc.submit(home_app(f"c{i}")))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sum(r.status.value == "ASSIGNED" for r in results) == n
    caps = {u.id: u.max_active_cases for u in SEED_UNDERWRITERS}
    for u in repo.list_underwriters():
        assert u.active_cases <= caps[u.id]
    assert sum(u.active_cases for u in repo.list_underwriters()) == n


def test_full_capacity_leads_to_unroutable(repo, no_llm):
    svc = RouterService(repo)
    results = [svc.submit(home_app(f"f{i}", amount=1_800_000)) for i in range(20)]
    assigned = [r for r in results if r.status.value == "ASSIGNED"]
    assert len(assigned) == 19  # UW-001 (10) + UW-006 (9)
    assert results[-1].status.value == "UNROUTABLE"


def test_llm_never_sees_underwriter_names(repo):
    app, ranked = _ctx(repo)
    from datetime import datetime, timezone
    blob = json.dumps(rules.build_llm_payload(app, ranked, datetime.now(timezone.utc)))
    assert not any(u.name in blob for u in SEED_UNDERWRITERS)
    assert rules.render_names(f"{ranked[0].underwriter_id} fits.", ranked) == f"{ranked[0].name} fits."
