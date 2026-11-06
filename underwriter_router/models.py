"""Domain models (spec §3). Protected attributes are intentionally absent."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LoanType(str, Enum):
    HOME = "HOME"
    AUTO = "AUTO"
    PERSONAL = "PERSONAL"
    BUSINESS = "BUSINESS"
    LAP = "LAP"


class Priority(str, Enum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AppStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ROUTING = "ROUTING"
    ASSIGNED = "ASSIGNED"
    UNROUTABLE = "UNROUTABLE"
    COMPLETED = "COMPLETED"


class OverrideReason(str, Enum):
    EXPERTISE = "EXPERTISE"
    WORKLOAD = "WORKLOAD"
    CONFLICT_OF_INTEREST = "CONFLICT_OF_INTEREST"
    CUSTOMER_REQUEST = "CUSTOMER_REQUEST"
    OTHER = "OTHER"


class DecisionSource(str, Enum):
    GENAI_SIMPLE = "GENAI_SIMPLE"      # qwen
    GENAI_COMPLEX = "GENAI_COMPLEX"    # Claude deep agent
    RULES_FALLBACK = "RULES_FALLBACK"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class LoanApplicationIn(BaseModel):
    """What a submitter sends."""
    idempotency_key: str = Field(min_length=1)
    submitter_id: str = Field(min_length=1)
    loan_type: LoanType
    amount: float = Field(gt=0)
    currency: str = "USD"
    region: str = Field(min_length=2)
    priority: Priority = Priority.NORMAL
    risk_band: RiskBand = RiskBand.MEDIUM
    complexity_flags: list[str] = Field(default_factory=list)


class LoanApplication(LoanApplicationIn):
    id: str
    status: AppStatus
    sla_due_at: str
    created_at: str
    version: int = 0


class Performance(BaseModel):
    avg_turnaround_hours: float = 48.0
    sla_hit_rate: float = 0.8
    rework_rate: float = 0.05


class Underwriter(BaseModel):
    id: str
    name: str
    skills: list[str]
    licensed_regions: list[str]
    authority_limit: float
    max_active_cases: int
    active_cases: int = 0
    status: str = "AVAILABLE"
    performance: Performance = Field(default_factory=Performance)


class Exclusion(BaseModel):
    underwriter_id: str
    reasons: list[str]


class ScoredCandidate(BaseModel):
    underwriter_id: str
    name: str
    score: float
    components: dict[str, float]
    active_cases: int
    max_active_cases: int
    skills: list[str]
    sla_hit_rate: float
    avg_turnaround_hours: float
    recent_assignments: int


class LLMChoice(BaseModel):
    """Structured output every GenAI optimizer must return (spec §4 step 4)."""
    selected_underwriter_id: str = Field(description="Must be one of the shortlisted underwriter ids")
    confidence: float = Field(ge=0, le=1, description="0-1 confidence in this choice")
    rationale: str = Field(description="2-4 plain-English sentences for the submitter")
    factors: list[str] = Field(default_factory=list, description="Short list of key factors behind the choice")
    alternatives_considered: list[str] = Field(
        default_factory=list, description="For other candidates: '<id>: why not chosen'"
    )


class RoutingResult(BaseModel):
    application_id: str
    status: AppStatus
    underwriter_id: str | None = None
    underwriter_name: str | None = None
    decision_source: DecisionSource | None = None
    rationale: str
    complexity: str | None = None
    fairness_signals: dict = Field(default_factory=dict)
    decision_id: str | None = None
    duplicate: bool = False
