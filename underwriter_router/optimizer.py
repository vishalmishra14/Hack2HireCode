"""GenAI optimizers (spec §4 step 4).

- simple path : one structured-output call to the local model (qwen3.5:2b via Ollama)
- complex path: a Deep Agent (deepagents harness on LangGraph) driven by Claude, which can
                inspect the shortlist through read-only tools before deciding

Both return an LLMChoice; verifier.verify() decides whether it is acceptable.
The rules engine is always the final safety net; nothing in this module can assign anyone.
"""
from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from .config import settings
from .models import LLMChoice

SYSTEM_PROMPT = """You are the routing optimizer for a bank's loan underwriting team.

A deterministic rules engine has already removed every underwriter who is not allowed to take
this application (licence, authority limit, capacity, skill). You receive the remaining shortlist,
ranked by a rules score. Your job is to pick the single best underwriter FROM THE SHORTLIST,
balancing three things:

1. Contextual fit - does the underwriter's skill set cover the application's complexity flags?
2. SLA risk - with hours_until_sla in mind, prefer underwriters with spare capacity
   (current_open_cases well below max_open_cases), fast avg_turnaround_hours and a high
   historical_sla_hit_rate.
3. Fairness - avoid piling work on the same people; recent_assignments and current_open_cases
   show who is already busy.

The rules score is a strong default. Deviate from the top-ranked candidate only when the factors
above clearly justify it, and say why.

Rules:
- selected_underwriter_id MUST be one of the shortlisted underwriter_id values, copied exactly.
- Decide only from the data provided. Never infer or consider applicant demographics.
- rationale: 2-4 plain-English sentences addressed to the loan submitter. Refer to every
  underwriter by underwriter_id (e.g. "UW-001"), never with pronouns. Only mention complexity
  flags that are actually present on the application, and only quote figures from the data.
- confidence: 0-1, how clearly this candidate is the best choice."""


class OptimizerError(Exception):
    pass


# --- model factory ------------------------------------------------------------
def make_chat_model(spec: str, timeout_s: float) -> BaseChatModel:
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model, base_url=settings.ollama_base_url, temperature=0, reasoning=False,
            client_kwargs={"timeout": timeout_s},
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.claude_api_key:
            raise OptimizerError("CLAUDE_API_KEY is not set")
        # No temperature: current Claude models reject non-default sampling params.
        return ChatAnthropic(
            model=model, api_key=settings.claude_api_key, max_tokens=4096,
            timeout=timeout_s, max_retries=1,
        )
    raise ValueError(f"Unknown model provider in spec {spec!r}")


def _run_with_timeout(fn, timeout_s: float):
    pool = ThreadPoolExecutor(max_workers=1)
    # copy_context keeps the LangSmith parent run, so LLM calls nest under the graph trace.
    fut = pool.submit(contextvars.copy_context().run, fn)
    try:
        return fut.result(timeout=timeout_s)
    except FutureTimeout as e:
        raise OptimizerError(f"timed out after {timeout_s}s") from e
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def validate_choice(choice: LLMChoice | None, payload: dict) -> LLMChoice:
    """Only checks that a structured answer exists; content is judged by verifier.verify()."""
    if choice is None:
        raise OptimizerError("model returned no structured choice")
    return choice


# --- simple path: single structured call ----------------------------------------
def _feedback_text(feedback: list[str] | None) -> str:
    if not feedback:
        return ""
    return ("\n\nYour previous answer was REJECTED by the verifier. Fix every issue below and answer again:\n- "
            + "\n- ".join(feedback))


def optimize_simple(
    payload: dict, run_config: dict | None = None, feedback: list[str] | None = None
) -> tuple[LLMChoice, str]:
    llm = make_chat_model(settings.simple_model, settings.simple_timeout_s).with_structured_output(LLMChoice)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", "Choose the underwriter for this application.\n\n" + json.dumps(payload, indent=2)
         + _feedback_text(feedback)),
    ]
    choice = _run_with_timeout(lambda: llm.invoke(messages, config=run_config), settings.simple_timeout_s)
    return validate_choice(choice, payload), settings.simple_model


# --- complex path: Deep Agent harness ----------------------------------------------
def _build_tools(payload: dict):
    by_id = {c["underwriter_id"]: c for c in payload["shortlist"]}

    @tool
    def get_application() -> str:
        """Return the loan application details (routing-relevant fields only)."""
        return json.dumps(payload["application"])

    @tool
    def list_shortlist() -> str:
        """Return the shortlisted underwriters with rules scores, ranked best first."""
        return json.dumps(
            [{"underwriter_id": c["underwriter_id"], "rules_score": c["rules_score"]}
             for c in payload["shortlist"]]
        )

    @tool
    def get_underwriter_profile(underwriter_id: str) -> str:
        """Return full metrics for one shortlisted underwriter: skills, load, SLA history, score breakdown."""
        if underwriter_id not in by_id:
            return json.dumps({"error": f"{underwriter_id} is not in the shortlist", "allowed": list(by_id)})
        return json.dumps(by_id[underwriter_id])

    @tool
    def assess_sla_risk(underwriter_id: str) -> str:
        """Estimate SLA risk if this underwriter takes the application, from turnaround, load and hours left."""
        c = by_id.get(underwriter_id)
        if c is None:
            return json.dumps({"error": f"{underwriter_id} is not in the shortlist"})
        hours_left = payload["application"]["hours_until_sla"]
        queue_factor = 1 + c["current_open_cases"] / max(c["max_open_cases"], 1)
        expected = c["avg_turnaround_hours"] * queue_factor
        risk = "LOW" if expected < hours_left * 0.6 else "MEDIUM" if expected < hours_left else "HIGH"
        return json.dumps({
            "underwriter_id": underwriter_id, "expected_hours": round(expected, 1),
            "hours_until_sla": hours_left, "risk": risk,
        })

    return [get_application, list_shortlist, get_underwriter_profile, assess_sla_risk]


def optimize_complex(
    payload: dict, reasons: list[str], run_config: dict | None = None, feedback: list[str] | None = None
) -> tuple[LLMChoice, str]:
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=make_chat_model(settings.complex_model, settings.complex_timeout_s),
        tools=_build_tools(payload),
        system_prompt=SYSTEM_PROMPT
        + "\n\nUse the tools to inspect candidates (especially get_underwriter_profile and "
          "assess_sla_risk for the top candidates) before deciding.",
        response_format=LLMChoice,
        name="complex-routing-agent",
    )
    prompt = (
        "Route this loan application. It was flagged as complex because: "
        + "; ".join(reasons)
        + ". Inspect the shortlist with your tools, then return your decision."
        + _feedback_text(feedback)
    )
    cfg = {**(run_config or {}), "recursion_limit": 40}
    result = _run_with_timeout(
        lambda: agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=cfg),
        settings.complex_timeout_s,
    )
    return validate_choice(result.get("structured_response"), payload), settings.complex_model
