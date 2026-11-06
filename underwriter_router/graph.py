"""The routing agent loop as a LangGraph StateGraph (spec §4).

    load_context -> eligibility -> rank ──┬─(none)──────────────────────────────────┐
                                          ├─(simple)─> genai_simple ──┐             │
                                          └─(complex)> genai_complex ─┴─> verify ─┬─┴─> commit -> END
                                                ^                                 │
                                                └──── retry with feedback ────────┘

Deterministic nodes decide what is allowed; GenAI nodes only propose a pick from the
shortlist; `verify` checks the pick and its rationale (retry once with the issues as feedback,
then rules fallback); `commit` writes atomically. Every node shows up as a span in LangSmith.
"""
from __future__ import annotations

import operator
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from . import optimizer, rules, verifier
from .config import settings, weights
from .db import CapacityExhausted, Repository
from .models import (
    DecisionSource,
    Exclusion,
    LLMChoice,
    LoanApplication,
    ScoredCandidate,
    Underwriter,
)


class RoutingState(TypedDict, total=False):
    application_id: str
    started_at: float
    application: LoanApplication
    underwriters: list[Underwriter]
    recent_counts: dict[str, int]
    eligible: list[Underwriter]
    excluded: list[Exclusion]
    ranked: list[ScoredCandidate]
    shortlist: list[ScoredCandidate]
    complexity: str
    complexity_reasons: list[str]
    llm_payload: dict
    llm_choice: LLMChoice | None
    llm_source: DecisionSource | None
    llm_model: str | None
    llm_errors: Annotated[list[str], operator.add]
    verification_log: Annotated[list[dict], operator.add]
    verification_feedback: list[str]
    result: dict[str, Any]


def build_graph(repo: Repository):
    # --- deterministic nodes -------------------------------------------------
    def load_context(state: RoutingState) -> dict:
        app = repo.get_application(state["application_id"])
        if app is None:
            raise ValueError(f"Unknown application {state['application_id']}")
        return {
            "application": app,
            "underwriters": repo.list_underwriters(),
            "recent_counts": repo.recent_assignment_counts(settings.fairness_window),
            "started_at": time.perf_counter(),
            "llm_choice": None,
            "llm_source": None,
            "llm_model": None,
            "verification_feedback": [],
        }

    def eligibility(state: RoutingState) -> dict:
        eligible, excluded = rules.filter_eligible(state["application"], state["underwriters"])
        return {"eligible": eligible, "excluded": excluded}

    def rank(state: RoutingState) -> dict:
        ranked = rules.rank(
            state["application"], state["eligible"], state["recent_counts"], settings.fairness_window, weights
        )
        shortlist = ranked[: settings.shortlist_size]
        complexity, reasons = rules.classify_complexity(state["application"], shortlist, settings)
        if not settings.llm_enabled and complexity != "none":
            complexity, reasons = "none", reasons + ["LLM disabled by config"]
        payload = rules.build_llm_payload(state["application"], shortlist, datetime.now(timezone.utc))
        return {
            "ranked": ranked, "shortlist": shortlist, "complexity": complexity,
            "complexity_reasons": reasons, "llm_payload": payload,
        }

    def route_by_complexity(state: RoutingState) -> str:
        return {"simple": "genai_simple", "complex": "genai_complex"}.get(state["complexity"], "commit")

    # --- GenAI nodes ----------------------------------------------------------
    def genai_simple(state: RoutingState, config: RunnableConfig) -> dict:
        fb = state.get("verification_feedback")
        try:
            choice, model = optimizer.optimize_simple(state["llm_payload"], config, feedback=fb)
            return {"llm_choice": choice, "llm_model": model, "llm_source": DecisionSource.GENAI_SIMPLE}
        except Exception as e:  # noqa: BLE001 - any failure means rules fallback
            return {"llm_errors": [f"simple[{settings.simple_model}]: {type(e).__name__}: {e}"[:500]]}

    def genai_complex(state: RoutingState, config: RunnableConfig) -> dict:
        fb = state.get("verification_feedback")
        errors = []
        try:
            choice, model = optimizer.optimize_complex(
                state["llm_payload"], state["complexity_reasons"], config, feedback=fb)
            return {"llm_choice": choice, "llm_model": model, "llm_source": DecisionSource.GENAI_COMPLEX}
        except Exception as e:  # noqa: BLE001
            errors.append(f"complex[{settings.complex_model}]: {type(e).__name__}: {e}"[:500])
        # Degrade to the local model before falling all the way back to rules.
        try:
            choice, model = optimizer.optimize_simple(state["llm_payload"], config, feedback=fb)
            return {"llm_choice": choice, "llm_model": model, "llm_source": DecisionSource.GENAI_SIMPLE,
                    "llm_errors": errors}
        except Exception as e:  # noqa: BLE001
            errors.append(f"simple[{settings.simple_model}]: {type(e).__name__}: {e}"[:500])
            return {"llm_errors": errors}

    # --- verification loop ---------------------------------------------------------
    def verify(state: RoutingState) -> dict:
        choice = state.get("llm_choice")
        attempt = len(state.get("verification_log", [])) + 1
        if choice is None:
            if attempt == 1:  # GenAI unavailable on first try -> straight to rules fallback
                return {}
            # The retry itself failed: record it so the loop terminates.
            return {"verification_log": [{
                "attempt": attempt, "passed": False, "selected_underwriter_id": None, "checks": {},
                "issues": ["GenAI produced no answer on retry"], "rejected_rationale": None,
                "model": state.get("llm_model")}]}
        v = verifier.verify(
            choice, state["application"], state["shortlist"],
            fresh_underwriters=repo.list_underwriters(), attempt=attempt,
        )
        update = {"verification_log": [{**v.model_dump(), "model": state.get("llm_model")}]}
        if not v.passed:
            update |= {"llm_choice": None, "llm_source": None, "verification_feedback": v.issues}
        return update

    def after_verify(state: RoutingState) -> str:
        log = state.get("verification_log", [])
        if state.get("llm_choice") is not None or not log or log[-1]["passed"]:
            return "commit"
        if len(log) < settings.max_verification_attempts:
            return "genai_complex" if state["complexity"] == "complex" else "genai_simple"
        return "commit"

    # --- commit: re-validate, assign atomically, fall back in rules order -------
    def commit(state: RoutingState) -> dict:
        app = state["application"]
        ranked = state["ranked"]
        by_id = {c.underwriter_id: c for c in ranked}
        choice = state.get("llm_choice")

        order: list[tuple[ScoredCandidate, DecisionSource]] = []
        if choice:
            order.append((by_id[choice.selected_underwriter_id], state["llm_source"]))
        order += [(c, DecisionSource.RULES_FALLBACK) for c in ranked
                  if not choice or c.underwriter_id != choice.selected_underwriter_id]

        base = {
            "input_snapshot": {
                "application": app.model_dump(mode="json"),
                "underwriters": [u.model_dump(mode="json") for u in state["underwriters"]],
                "recent_counts": state["recent_counts"],
            },
            "rules_version": settings.rules_version,
            "weights_version": weights.version,
            "complexity": f"{state['complexity']}: {'; '.join(state['complexity_reasons'])}",
            "eligible": [c.model_dump() for c in ranked],
            "excluded": [e.model_dump() for e in state["excluded"]],
            "shortlist": [c.underwriter_id for c in state["shortlist"]],
            "llm_model": state.get("llm_model"),
            "llm_prompt_version": settings.prompt_version if state.get("llm_model") else None,
            "llm_response": choice.model_dump() if choice else None,
            "llm_errors": state.get("llm_errors", []),
            "verification": state.get("verification_log", []),
        }
        vlog = state.get("verification_log", [])
        commit_errors = []
        for cand, source in order:
            if source == DecisionSource.RULES_FALLBACK:
                why = (f"GenAI decision failed verification {len(vlog)}x" if vlog and not vlog[-1]["passed"]
                       else "GenAI unavailable or invalid" if state.get("llm_errors")
                       else "single eligible candidate" if state["complexity"] == "none" and len(ranked) == 1
                       else "GenAI pick no longer available" if choice
                       else "; ".join(state["complexity_reasons"]))
                rationale = rules.template_rationale(cand, ranked, why)
            else:
                rationale = rules.render_names(choice.rationale, ranked)  # verified text, ids -> names
            signals = rules.fairness_signals(cand, ranked, settings.fairness_window)
            decision = {
                **base, "selected_underwriter_id": cand.underwriter_id, "decision_source": source.value,
                "rationale": rationale, "fairness_signals": signals,
                "latency_ms": int((time.perf_counter() - state["started_at"]) * 1000),
            }
            decision["llm_errors"] = base["llm_errors"] + commit_errors
            events = [
                {"event_type": "UNDERWRITER_ASSIGNED",
                 "payload": {"application_id": app.id, "underwriter_id": cand.underwriter_id}},
                {"event_type": "SUBMITTER_NOTIFIED",
                 "payload": {"application_id": app.id, "submitter_id": app.submitter_id, "rationale": rationale}},
            ]
            try:
                decision_id = repo.commit_assignment(app.id, cand.underwriter_id, decision, events)
            except CapacityExhausted:
                commit_errors.append(f"commit: {cand.underwriter_id} filled up before assignment")
                continue
            return {"result": {
                "status": "ASSIGNED", "underwriter_id": cand.underwriter_id, "underwriter_name": cand.name,
                "decision_source": source.value, "rationale": rationale, "fairness_signals": signals,
                "decision_id": decision_id,
            }}

        # Nobody could take it.
        reasons = "; ".join(f"{e.underwriter_id}: {', '.join(e.reasons)}" for e in state["excluded"])
        rationale = "No eligible underwriter is currently available. Escalated to a supervisor. " + (
            f"Exclusions: {reasons}" if reasons else "")
        decision = {**base, "selected_underwriter_id": None, "decision_source": None, "rationale": rationale,
                    "fairness_signals": {"protected_attrs_used": False},
                    "latency_ms": int((time.perf_counter() - state["started_at"]) * 1000)}
        decision["llm_errors"] = base["llm_errors"] + commit_errors
        decision_id = repo.mark_unroutable(app.id, decision)
        return {"result": {"status": "UNROUTABLE", "rationale": rationale, "decision_id": decision_id}}

    g = StateGraph(RoutingState)
    g.add_node("load_context", load_context)
    g.add_node("eligibility", eligibility)
    g.add_node("rank", rank)
    g.add_node("genai_simple", genai_simple)
    g.add_node("genai_complex", genai_complex)
    g.add_node("verify", verify)
    g.add_node("commit", commit)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "eligibility")
    g.add_edge("eligibility", "rank")
    g.add_conditional_edges("rank", route_by_complexity, ["genai_simple", "genai_complex", "commit"])
    g.add_edge("genai_simple", "verify")
    g.add_edge("genai_complex", "verify")
    g.add_conditional_edges("verify", after_verify, ["genai_simple", "genai_complex", "commit"])
    g.add_edge("commit", END)
    return g.compile()
