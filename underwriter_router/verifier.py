"""Verification loop (spec §14 "Agent loops" #2). Deterministic checks on every GenAI decision.

Decision checks  - is the pick allowed and defensible?
  in_shortlist      pick is one of the shortlisted ids
  confidence        >= settings.min_confidence
  still_eligible    passes hard filters against FRESH underwriter state (may change during LLM call)
  rank_deviation    score gap to the rules top-1 <= settings.max_score_deviation

Rationale checks - is the explanation true and fair?
  names_selected    rationale refers to the selected underwriter (by id)
  no_foreign_names  no underwriter outside the shortlist is mentioned (by id or name)
  flags_grounded    every complexity flag mentioned is actually on the application
  numbers_grounded  every "N%" and "a/b" figure matches the shortlist data
  neutral_language  no gendered pronouns or protected-attribute terms
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .config import settings
from .models import LLMChoice, LoanApplication, ScoredCandidate, Underwriter
from .rules import check_eligibility

KNOWN_FLAGS = ("SELF_EMPLOYED", "MULTI_PROPERTY", "FOREIGN_INCOME")
GENDERED = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.I)
PROTECTED_TERMS = re.compile(
    r"\b(race|racial|ethnicity|ethnic|gender|religion|religious|marital|married|nationality|"
    r"national origin|disability|disabled|age|aged|elderly)\b", re.I,
)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
FRACTION = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


class Verification(BaseModel):
    attempt: int
    passed: bool
    selected_underwriter_id: str | None
    checks: dict[str, bool]
    issues: list[str]
    rejected_rationale: str | None = None


def _mentions(text: str, uw_id: str, name: str) -> bool:
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in (uw_id, name, name.split()[0]))


def verify(
    choice: LLMChoice,
    app: LoanApplication,
    shortlist: list[ScoredCandidate],
    fresh_underwriters: list[Underwriter],
    attempt: int,
) -> Verification:
    issues: dict[str, str] = {}
    by_id = {c.underwriter_id: c for c in shortlist}
    sel = by_id.get(choice.selected_underwriter_id)
    text = choice.rationale or ""

    # --- decision checks ---------------------------------------------------
    if sel is None:
        issues["in_shortlist"] = f"{choice.selected_underwriter_id} is not in the shortlist {list(by_id)}"
    if choice.confidence < settings.min_confidence:
        issues["confidence"] = f"confidence {choice.confidence} < {settings.min_confidence}"
    if sel is not None:
        fresh = next((u for u in fresh_underwriters if u.id == sel.underwriter_id), None)
        reasons = check_eligibility(app, fresh) if fresh else ["underwriter no longer exists"]
        if reasons:
            issues["still_eligible"] = f"{sel.underwriter_id} is no longer eligible: {', '.join(reasons)}"
        gap = shortlist[0].score - sel.score
        if gap > settings.max_score_deviation:
            issues["rank_deviation"] = (
                f"picked {sel.underwriter_id} scores {gap:.3f} below the rules top candidate "
                f"{shortlist[0].underwriter_id} (max allowed {settings.max_score_deviation})"
            )

    # --- rationale checks ----------------------------------------------------
    if not text.strip():
        issues["names_selected"] = "rationale is empty"
    elif sel is not None and not re.search(rf"\b{re.escape(sel.underwriter_id)}\b", text):
        issues["names_selected"] = f"rationale does not refer to the selected underwriter {sel.underwriter_id}"

    outsiders = [u.id for u in fresh_underwriters if u.id not in by_id and _mentions(text, u.id, u.name)]
    if outsiders:
        issues["no_foreign_names"] = f"rationale mentions non-shortlisted underwriters: {outsiders}"

    norm = text.upper().replace("-", "_").replace(" ", "_")
    bogus_flags = [f for f in KNOWN_FLAGS if f in norm and f not in app.complexity_flags]
    if bogus_flags:
        issues["flags_grounded"] = (
            f"rationale cites complexity flags not on the application: {bogus_flags} "
            f"(application flags: {app.complexity_flags or 'none'})"
        )

    allowed_pct = {0.0, 100.0}
    allowed_frac = set()
    for c in shortlist:
        allowed_pct.update(v * 100 for v in c.components.values())
        allowed_pct.update({c.sla_hit_rate * 100, c.active_cases / c.max_active_cases * 100})
        allowed_frac.add((c.active_cases, c.max_active_cases))
    bad_pct = [p for p in PERCENT.findall(text) if not any(abs(float(p) - a) <= 1.0 for a in allowed_pct)]
    bad_frac = [f"{a}/{b}" for a, b in FRACTION.findall(text) if (int(a), int(b)) not in allowed_frac]
    if bad_pct or bad_frac:
        issues["numbers_grounded"] = f"figures not supported by the data: {bad_pct + bad_frac}"

    lang = sorted({m.lower() for m in GENDERED.findall(text)} | {m.lower() for m in PROTECTED_TERMS.findall(text)})
    if lang:
        issues["neutral_language"] = f"use names only; remove gendered/protected-attribute words: {lang}"

    check_names = ("in_shortlist", "confidence", "still_eligible", "rank_deviation", "names_selected",
                   "no_foreign_names", "flags_grounded", "numbers_grounded", "neutral_language")
    return Verification(
        attempt=attempt,
        passed=not issues,
        selected_underwriter_id=choice.selected_underwriter_id,
        checks={k: k not in issues for k in check_names},
        issues=list(issues.values()),
        rejected_rationale=None if not issues else text,
    )
