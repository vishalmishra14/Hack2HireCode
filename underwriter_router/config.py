"""Central configuration: env loading, model choices, thresholds, rule weights."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- LangSmith tracing -------------------------------------------------------
# LangGraph / LangChain pick these up automatically; every graph run becomes a trace.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "underwriter-router")


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("ROUTER_DB_PATH", PROJECT_ROOT / "data" / "router.db"))

    # Models
    # Model specs are "<provider>:<model>", provider in {ollama, anthropic}.
    simple_model: str = os.getenv("SIMPLE_MODEL", "ollama:qwen3.5:2b")
    complex_model: str = os.getenv("COMPLEX_MODEL", "anthropic:claude-sonnet-5-5")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    claude_api_key: str | None = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    llm_enabled: bool = os.getenv("ROUTER_LLM_ENABLED", "true").lower() == "true"

    simple_timeout_s: float = float(os.getenv("SIMPLE_TIMEOUT_S", "45"))
    complex_timeout_s: float = float(os.getenv("COMPLEX_TIMEOUT_S", "90"))

    # Routing
    shortlist_size: int = 5
    min_confidence: float = 0.5
    max_verification_attempts: int = 2   # GenAI tries (1 + 1 retry with feedback) before rules fallback
    max_score_deviation: float = 0.10    # GenAI may not pick a candidate scoring >0.10 below rules top-1
    rules_version: str = "rules-0.1"
    prompt_version: str = "prompt-0.1"

    # Complexity classifier thresholds (deterministic; decides qwen vs Claude)
    complex_amount_threshold: float = 1_000_000
    complex_flag_count: int = 2
    close_call_score_gap: float = 0.03

    # Fairness window: how many recent assignments to consider
    fairness_window: int = 50

    # SLA hours by priority
    sla_hours: dict = field(default_factory=lambda: {"URGENT": 4, "HIGH": 24, "NORMAL": 72})


@dataclass(frozen=True)
class Weights:
    """Versioned scoring weights. The hill-climbing loop will produce new versions."""
    version: str = "w-0.1"
    skill: float = 0.30
    load: float = 0.25
    sla: float = 0.20
    speed: float = 0.10
    fairness: float = 0.15


settings = Settings()
weights = Weights()
