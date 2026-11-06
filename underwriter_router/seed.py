"""Seed underwriters for local development and the demo (fictional people)."""
from __future__ import annotations

from .db import Repository
from .models import Performance, Underwriter

P = Performance

SEED_UNDERWRITERS = [
    Underwriter(id="UW-001", name="Asha Rao", skills=["HOME", "LAP", "SELF_EMPLOYED", "MULTI_PROPERTY"],
                licensed_regions=["TX", "CA"], authority_limit=2_000_000, max_active_cases=10,
                performance=P(avg_turnaround_hours=20, sla_hit_rate=0.95, rework_rate=0.02)),
    Underwriter(id="UW-002", name="Ben Carter", skills=["HOME", "PERSONAL"],
                licensed_regions=["TX", "NY"], authority_limit=750_000, max_active_cases=12,
                performance=P(avg_turnaround_hours=30, sla_hit_rate=0.88, rework_rate=0.04)),
    Underwriter(id="UW-003", name="Chen Li", skills=["BUSINESS", "SELF_EMPLOYED", "FOREIGN_INCOME"],
                licensed_regions=["CA", "NY", "TX"], authority_limit=5_000_000, max_active_cases=8,
                performance=P(avg_turnaround_hours=36, sla_hit_rate=0.90, rework_rate=0.03)),
    Underwriter(id="UW-004", name="Diana Lopez", skills=["AUTO", "PERSONAL"],
                licensed_regions=["TX", "FL", "CA"], authority_limit=150_000, max_active_cases=20,
                performance=P(avg_turnaround_hours=8, sla_hit_rate=0.97, rework_rate=0.05)),
    Underwriter(id="UW-005", name="Ethan Brooks", skills=["HOME", "LAP", "FOREIGN_INCOME"],
                licensed_regions=["NY", "FL"], authority_limit=1_500_000, max_active_cases=10,
                performance=P(avg_turnaround_hours=28, sla_hit_rate=0.85, rework_rate=0.06)),
    Underwriter(id="UW-006", name="Fatima Khan", skills=["HOME", "BUSINESS", "MULTI_PROPERTY", "SELF_EMPLOYED"],
                licensed_regions=["TX", "FL", "NY"], authority_limit=3_000_000, max_active_cases=9,
                performance=P(avg_turnaround_hours=24, sla_hit_rate=0.93, rework_rate=0.02)),
    Underwriter(id="UW-007", name="George Miller", skills=["AUTO", "PERSONAL", "HOME"],
                licensed_regions=["CA", "FL"], authority_limit=500_000, max_active_cases=15,
                performance=P(avg_turnaround_hours=16, sla_hit_rate=0.82, rework_rate=0.08)),
    Underwriter(id="UW-008", name="Hana Sato", skills=["HOME", "LAP", "SELF_EMPLOYED"],
                licensed_regions=["TX", "CA", "NY", "FL"], authority_limit=1_000_000, max_active_cases=10,
                status="ON_LEAVE", performance=P(avg_turnaround_hours=22, sla_hit_rate=0.91, rework_rate=0.03)),
    Underwriter(id="UW-009", name="Ivan Petrov", skills=["BUSINESS", "LAP", "MULTI_PROPERTY"],
                licensed_regions=["TX", "NY"], authority_limit=10_000_000, max_active_cases=6,
                performance=P(avg_turnaround_hours=48, sla_hit_rate=0.80, rework_rate=0.02)),
    Underwriter(id="UW-010", name="Julia Nunes", skills=["HOME", "AUTO", "PERSONAL", "SELF_EMPLOYED"],
                licensed_regions=["TX", "CA"], authority_limit=600_000, max_active_cases=14,
                performance=P(avg_turnaround_hours=18, sla_hit_rate=0.92, rework_rate=0.04)),
]


def seed(repo: Repository) -> int:
    for uw in SEED_UNDERWRITERS:
        repo.upsert_underwriter(uw)
    return len(SEED_UNDERWRITERS)
