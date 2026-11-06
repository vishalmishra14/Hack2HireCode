"""Deterministic rules engine (spec §4 steps 2-3). No LLM, no randomness, no clock reads.

Same inputs + same weights version => same ranking. Every exclusion carries a reason.
"""
from __future__ import annotations

from datetime import datetime

from .config import Settings, Weights
from .models import Exclusion, LoanApplication, ScoredCandidate, Underwriter

# Fields the LLM is allowed to see about an application. Anything else (names, ids of
# applicants, protected attributes) is never sent. Enforced by build_llm_payload + tests.
LLM_APPLICATION_FIELDS = (
    "loan_type", "amount", "currency", "region", "priority", "risk_band", "complexity_flags",
)
PROTECTED_ATTRIBUTES = (
    "race", "ethnicity", "gender", "sex", "religion", "age", "date_of_birth", "marital_status",
    "nationality", "national_origin", "disability", "applicant_name",
)


# --- Step 2: hard filters ------------------------------------------------------
def check_eligibility(app: LoanApplication, uw: Underwriter) -> list[str]:
    """Return a list of reasons the underwriter is NOT eligible (empty == eligible)."""
    reasons = []
    if uw.status != "AVAILABLE":
        reasons.append(f"status is {uw.status}")
    if uw.active_cases >= uw.max_active_cases:
        reasons.append(f"at capacity ({uw.active_cases}/{uw.max_active_cases})")
    if uw.authority_limit < app.amount:
        reasons.append(f"authority limit {uw.authority_limit:,.0f} < amount {app.amount:,.0f}")
    if app.region not in uw.licensed_regions:
        reasons.append(f"not licensed in {app.region}")
    if app.loan_type.value not in uw.skills:
        reasons.append(f"no {app.loan_type.value} skill")
    return reasons


def filter_eligible(app: LoanApplication, underwriters: list[Underwriter]) -> tuple[list[Underwriter], list[Exclusion]]:
    eligible, excluded = [], []
    for uw in underwriters:
        reasons = check_eligibility(app, uw)
        if reasons:
            excluded.append(Exclusion(underwriter_id=uw.id, reasons=reasons))
        else:
            eligible.append(uw)
    return eligible, excluded


# --- Step 3: scoring & ranking -------------------------------------------------
def score_candidate(
    app: LoanApplication, uw: Underwriter, recent_counts: dict[str, int], window: int, w: Weights
) -> ScoredCandidate:
    flags = app.complexity_flags
    skill_match = 1.0 if not flags else sum(f in uw.skills for f in flags) / len(flags)
    load_headroom = 1.0 - uw.active_cases / uw.max_active_cases
    sla = uw.performance.sla_hit_rate
    speed = 1.0 - min(uw.performance.avg_turnaround_hours / 72.0, 1.0)
    recent = recent_counts.get(uw.id, 0)
    fairness = 1.0 - min(recent / max(window, 1) * 5, 1.0)  # >=20% of recent volume -> 0

    components = {
        "skill_match": round(skill_match, 4),
        "load_headroom": round(load_headroom, 4),
        "sla_hit_rate": round(sla, 4),
        "speed": round(speed, 4),
        "fairness": round(fairness, 4),
    }
    score = (
        w.skill * skill_match + w.load * load_headroom + w.sla * sla + w.speed * speed + w.fairness * fairness
    )
    return ScoredCandidate(
        underwriter_id=uw.id, name=uw.name, score=round(score, 6), components=components,
        active_cases=uw.active_cases, max_active_cases=uw.max_active_cases, skills=uw.skills,
        sla_hit_rate=sla, avg_turnaround_hours=uw.performance.avg_turnaround_hours, recent_assignments=recent,
    )


def rank(
    app: LoanApplication, eligible: list[Underwriter], recent_counts: dict[str, int], window: int, w: Weights
) -> list[ScoredCandidate]:
    scored = [score_candidate(app, uw, recent_counts, window, w) for uw in eligible]
    # Deterministic tie-break: score desc, then lower load, then id.
    return sorted(scored, key=lambda c: (-c.score, c.active_cases, c.underwriter_id))


# --- Complexity: which optimizer handles the decision ----------------------------
def classify_complexity(app: LoanApplication, shortlist: list[ScoredCandidate], s: Settings) -> tuple[str, list[str]]:
    """Return ("none" | "simple" | "complex", reasons). Deterministic."""
    if len(shortlist) <= 1:
        return "none", ["only one eligible candidate" if shortlist else "no eligible candidates"]
    reasons = []
    if app.risk_band.value == "HIGH":
        reasons.append("high risk band")
    if app.amount >= s.complex_amount_threshold:
        reasons.append(f"amount >= {s.complex_amount_threshold:,.0f}")
    if len(app.complexity_flags) >= s.complex_flag_count:
        reasons.append(f"{len(app.complexity_flags)} complexity flags")
    if app.priority.value == "URGENT":
        reasons.append("urgent priority (SLA risk)")
    if reasons:
        return "complex", reasons
    # A close call alone is not complex - the local model can break the tie.
    if shortlist[0].score - shortlist[1].score < s.close_call_score_gap:
        return "simple", ["routine application", "close call between top candidates"]
    return "simple", ["routine application"]


# --- Fairness & explainability helpers -------------------------------------------
def gini(values: list[float]) -> float:
    vals = sorted(v for v in values if v >= 0)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return round((2 * cum) / (n * sum(vals)) - (n + 1) / n, 4)


def fairness_signals(selected: ScoredCandidate, ranked: list[ScoredCandidate], window: int) -> dict:
    ids = [c.underwriter_id for c in ranked]
    return {
        "load_ratio_selected": round(selected.active_cases / selected.max_active_cases, 4),
        "team_load_gini": gini([c.active_cases / c.max_active_cases for c in ranked]),
        "selected_recent_share": round(selected.recent_assignments / max(window, 1), 4),
        "rank_in_rules": ids.index(selected.underwriter_id) + 1,
        "protected_attrs_used": False,
    }


def template_rationale(selected: ScoredCandidate, ranked: list[ScoredCandidate], why_rules: str) -> str:
    c = selected.components
    return (
        f"Assigned to {selected.name} by the rules engine ({why_rules}). "
        f"Score {selected.score:.3f}, rank {[r.underwriter_id for r in ranked].index(selected.underwriter_id) + 1} "
        f"of {len(ranked)} eligible: skill match {c['skill_match']:.0%}, load headroom {c['load_headroom']:.0%}, "
        f"SLA hit rate {c['sla_hit_rate']:.0%}, current load {selected.active_cases}/{selected.max_active_cases}."
    )


def render_names(text: str, candidates: list[ScoredCandidate]) -> str:
    """Replace underwriter ids in a verified rationale with display names, for the submitter."""
    for c in candidates:
        text = text.replace(c.underwriter_id, c.name)
    return text


def build_llm_payload(app: LoanApplication, shortlist: list[ScoredCandidate], now: datetime) -> dict:
    """The ONLY data the LLM sees. Whitelisted application fields + shortlist metrics."""
    app_view = {k: (v.value if hasattr(v, "value") else v) for k, v in app.model_dump().items()
                if k in LLM_APPLICATION_FIELDS}
    sla_due = datetime.fromisoformat(app.sla_due_at)
    app_view["hours_until_sla"] = round((sla_due - now).total_seconds() / 3600, 1)
    return {
        "application": app_view,
        "shortlist": [
            {
                # No names: a name can signal gender/ethnicity, so the model sees ids only.
                # render_names() swaps ids for names after verification.
                "underwriter_id": c.underwriter_id,
                "rules_score": c.score,
                "score_components": c.components,
                "current_open_cases": c.active_cases,
                "max_open_cases": c.max_active_cases,
                "skills": c.skills,
                "historical_sla_hit_rate": c.sla_hit_rate,
                "avg_turnaround_hours": c.avg_turnaround_hours,
                "recent_assignments": c.recent_assignments,
            }
            for c in shortlist
        ],
    }
