"""End-to-end simulation of the event-driven and self-improvement loops.

    python -m underwriter_router.simulate            # rules-only routing (fast, deterministic)
    python -m underwriter_router.simulate --llm      # route with GenAI too (slow on local qwen)

Phase 1  traffic with the initial weights: submit -> route -> events -> cases complete -> outcomes,
         plus supervisor overrides, a crashed routing recovered by the sweeper, and UNROUTABLE
         applications re-routed when capacity frees.
Learn    one self-improvement cycle (traced in LangSmith).
Phase 2  A/B: identical seeded traffic replayed on two copies of the DB, old weights vs learned weights.

Ground truth is hidden from the router: some underwriters' configured SLA history is wrong, and
every underwriter misses SLAs more when overloaded. The loop has to discover this from outcomes.
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true", help="use GenAI in routing (slow)")
ap.add_argument("--rounds", type=int, default=12)
ap.add_argument("--per-round", type=int, default=18)
ARGS = ap.parse_args()
if not ARGS.llm:
    os.environ["ROUTER_LLM_ENABLED"] = "false"

from langsmith import tracing_context  # noqa: E402

from .config import PROJECT_ROOT  # noqa: E402
from .db import Repository  # noqa: E402
from .events import EventLoop  # noqa: E402
from .learning import run_cycle  # noqa: E402
from .models import AppStatus, LoanApplicationIn  # noqa: E402
from .seed import seed  # noqa: E402
from .service import RouterService  # noqa: E402

# Hidden truth (the router never sees this): real SLA ability and turnaround.
TRUE_SLA = {"UW-001": 0.93, "UW-002": 0.95, "UW-003": 0.88, "UW-004": 0.95, "UW-005": 0.70,
            "UW-006": 0.92, "UW-007": 0.50, "UW-008": 0.90, "UW-009": 0.80, "UW-010": 0.62}
LOAD_PENALTY = 0.55  # SLA probability drops by this * load ratio at assignment

PRODUCTS = [("HOME", 120_000, 900_000), ("AUTO", 10_000, 90_000), ("PERSONAL", 5_000, 60_000),
            ("BUSINESS", 200_000, 4_000_000), ("LAP", 100_000, 1_200_000)]
REGIONS = ["TX", "CA", "NY", "FL"]
FLAGS = ["SELF_EMPLOYED", "MULTI_PROPERTY", "FOREIGN_INCOME"]


def gen_app(rng: random.Random, key: str) -> LoanApplicationIn:
    lt, lo, hi = rng.choice(PRODUCTS)
    return LoanApplicationIn(
        idempotency_key=key, submitter_id=f"officer-{rng.randint(1, 9)}", loan_type=lt,
        amount=round(rng.uniform(lo, hi), -3), region=rng.choice(REGIONS),
        priority=rng.choices(["NORMAL", "HIGH", "URGENT"], [0.7, 0.25, 0.05])[0],
        risk_band=rng.choice(["LOW", "MEDIUM", "HIGH"]),
        complexity_flags=rng.sample(FLAGS, k=rng.choice([0, 0, 1, 2])),
    )


def run_traffic(repo, svc, loop, rng, rounds, per_round, tag, overrides=False) -> dict:
    load_at_assign, sla = {}, []
    for rnd in range(rounds):
        for i in range(per_round):
            r = svc.submit(gen_app(rng, f"{tag}-{rnd}-{i}"))
            if r.status == AppStatus.ASSIGNED:
                uw = next(u for u in repo.list_underwriters() if u.id == r.underwriter_id)
                load_at_assign[r.application_id] = uw.active_cases / uw.max_active_cases
        loop.drain()

        assigned = repo.list_application_ids(AppStatus.ASSIGNED)
        if overrides and rnd % 4 == 1:  # supervisor moves a case away from UW-007 with a reason
            for app_id in assigned:
                a = repo.get_active_assignment(app_id)
                if a["underwriter_id"] != "UW-007":
                    continue
                d = repo.get_latest_decision(app_id)
                alt = next((c["underwriter_id"] for c in d["eligible_candidates"]
                            if c["underwriter_id"] != "UW-007"), None)
                if alt:
                    try:
                        svc.override(app_id, alt, "supervisor-1", "WORKLOAD",
                                     "UW-007 has a backlog of missed SLAs this week")
                        break
                    except Exception:  # noqa: BLE001 - target may be full; try the next case
                        continue

        # ~60% of open cases complete each round; outcome drawn from the hidden truth.
        for app_id in repo.list_application_ids(AppStatus.ASSIGNED):
            if rng.random() > 0.6:
                continue
            uw_id = repo.get_active_assignment(app_id)["underwriter_id"]
            p = max(0.05, TRUE_SLA[uw_id] - LOAD_PENALTY * load_at_assign.get(app_id, 0.5))
            met = rng.random() < p
            sla.append(met)
            svc.complete_case(app_id, sla_met=met, turnaround_hours=round(rng.uniform(8, 60), 1),
                              reworked=rng.random() < (0.15 if not met else 0.03))
        loop.drain()
    return {"completed": len(sla), "sla_hit_rate": round(sum(sla) / max(len(sla), 1), 3)}


def copy_db(src: Path, dst: Path) -> None:
    for p in (dst, Path(f"{dst}-wal"), Path(f"{dst}-shm")):
        p.unlink(missing_ok=True)
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)


def main() -> None:
    data = PROJECT_ROOT / "data"
    base = data / "sim.db"
    for p in data.glob("sim*.db*"):
        p.unlink()
    repo = Repository(base)
    seed(repo)
    svc = RouterService(repo)
    notes = []
    loop = EventLoop(repo, svc, notify=lambda who, msg: notes.append((who, msg)))
    rng = random.Random(7)

    print("== Phase 1: traffic with initial weights ==")
    with tracing_context(enabled=False):  # bulk traffic not traced; the learning cycle is
        p1 = run_traffic(repo, svc, loop, rng, ARGS.rounds, ARGS.per_round, "p1", overrides=True)

        # Crash recovery: an application is claimed by a worker that dies before routing.
        app, _ = repo.create_application(LoanApplicationIn(
            idempotency_key="crash-1", submitter_id="officer-1", loan_type="HOME", amount=300_000, region="TX"), 72)
        repo.claim_for_routing(app.id)   # worker claims it ... and dies
        loop.drain()                     # lease still valid: nobody else may touch it
        stuck = repo.get_application(app.id).status.value
        EventLoop(repo, svc, notify=lambda *_: None, routing_lease_s=0.0).drain()  # lease expired
        recovered = repo.get_application(app.id).status.value

    counts = repo.event_counts()
    conn = sqlite3.connect(base)
    overrides = conn.execute("SELECT COUNT(*) FROM overrides").fetchone()[0]
    rerouted = conn.execute("SELECT COUNT(DISTINCT application_id) FROM routing_decisions WHERE attempt > 1 "
                            "AND decision_source IS NOT NULL AND decision_source != 'MANUAL_OVERRIDE'").fetchone()[0]
    unroutable = conn.execute("SELECT COUNT(*) FROM applications WHERE status='UNROUTABLE'").fetchone()[0]
    conn.close()
    print(f"  completed cases: {p1['completed']}  SLA hit rate: {p1['sla_hit_rate']:.1%}")
    print(f"  events: {counts}  notifications sent: {len(notes)}")
    print(f"  overrides (with reason): {overrides}  UNROUTABLE re-routed after capacity freed: {rerouted}"
          f"  still UNROUTABLE (no eligible underwriter exists): {unroutable}")
    print(f"  crash recovery: stuck app status {stuck} -> sweeper -> {recovered}")

    print("\n== Self-improvement cycle ==")
    with tracing_context(enabled=bool(os.getenv("LANGSMITH_API_KEY"))):
        rep = run_cycle(repo)
    print(f"  samples: {rep['samples']}  status: {rep['status']} ({rep['decision']})  iterations: {rep['iterations']}")
    print(f"  weights {rep['current_version']}: {rep['weights_before']}")
    print(f"  weights {rep['new_version'] or 'candidate'}: {rep['weights_after']}")
    print(f"  objective train {rep['train']['before']} -> {rep['train']['after']} | "
          f"holdout {rep['holdout']['before']} -> {rep['holdout']['after']}")
    for uw_id, ch in rep["performance_updates"].items():
        b, a = ch["before"]["sla_hit_rate"], ch["after"]["sla_hit_rate"]
        if abs(a - b) >= 0.05:
            print(f"  metric update {uw_id}: sla_hit_rate {b:.2f} -> {a:.2f} (n={ch['n']})")

    print("\n== Phase 2: A/B on identical traffic (before vs after the learning cycle) ==")
    old_db, new_db = data / "sim_old.db", data / "sim_new.db"
    copy_db(base, old_db)
    copy_db(base, new_db)
    results = {}
    for label, path in (("before learning", old_db), ("after learning", new_db)):
        r = Repository(path)
        if label == "before learning":
            r.activate_weights(rep["current_version"])  # roll back weights ...
            seed(r)                                     # ... and metrics to the configured baseline
        s = RouterService(r)
        with tracing_context(enabled=False):
            results[label] = run_traffic(r, s, EventLoop(r, s, notify=lambda *_: None), random.Random(99),
                                         ARGS.rounds, ARGS.per_round, "p2")
        print(f"  {label:16s} active={r.get_active_weights().version:6s} completed={results[label]['completed']}"
              f"  SLA hit rate={results[label]['sla_hit_rate']:.1%}")


if __name__ == "__main__":
    main()
