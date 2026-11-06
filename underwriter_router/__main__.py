"""Demo: seed underwriters, submit sample applications, print routing outcomes.

    python -m underwriter_router            # run demo against data/router.db
    python -m underwriter_router --reset    # wipe the demo DB first
    python -m underwriter_router --no-llm   # rules-only (deterministic fallback path)
"""
from __future__ import annotations

import argparse
import os
import uuid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the demo database first")
    ap.add_argument("--no-llm", action="store_true", help="disable GenAI (rules only)")
    args = ap.parse_args()
    if args.no_llm:
        os.environ["ROUTER_LLM_ENABLED"] = "false"

    from .config import settings
    from .db import Repository
    from .models import LoanApplicationIn
    from .seed import seed
    from .service import RouterService

    if args.reset and settings.db_path.exists():
        settings.db_path.unlink()
    repo = Repository(settings.db_path)
    seed(repo)
    svc = RouterService(repo)

    run = uuid.uuid4().hex[:6]
    samples = [
        # routine -> qwen (simple path)
        LoanApplicationIn(idempotency_key=f"{run}-1", submitter_id="officer-17", loan_type="HOME",
                          amount=320_000, region="TX", priority="NORMAL", risk_band="LOW"),
        # complex -> Claude deep agent (falls back to qwen, then rules)
        LoanApplicationIn(idempotency_key=f"{run}-2", submitter_id="broker-04", loan_type="BUSINESS",
                          amount=2_400_000, region="NY", priority="HIGH", risk_band="HIGH",
                          complexity_flags=["SELF_EMPLOYED", "FOREIGN_INCOME"]),
        # only one eligible -> rules direct
        LoanApplicationIn(idempotency_key=f"{run}-3", submitter_id="officer-17", loan_type="PERSONAL",
                          amount=20_000, region="NY", priority="NORMAL", risk_band="LOW"),
        # nobody eligible -> UNROUTABLE
        LoanApplicationIn(idempotency_key=f"{run}-4", submitter_id="broker-04", loan_type="BUSINESS",
                          amount=20_000_000, region="FL", priority="URGENT", risk_band="HIGH"),
    ]
    # duplicate submission of #1 -> same result, no second routing
    samples.append(samples[0])

    print(f"LLM enabled: {settings.llm_enabled} | simple={settings.simple_model} | complex={settings.complex_model}")
    print(f"LangSmith tracing: {os.getenv('LANGSMITH_TRACING', 'false')} "
          f"(project={os.getenv('LANGSMITH_PROJECT', '-')})\n")
    for i, s in enumerate(samples, 1):
        r = svc.submit(s)
        print(f"#{i} {s.loan_type.value} {s.amount:,.0f} {s.region} {s.priority.value}/{s.risk_band.value}"
              f"{'  [DUPLICATE]' if r.duplicate else ''}")
        src = r.decision_source.value if r.decision_source else None
        print(f"   status={r.status.value} complexity={r.complexity} source={src}")
        print(f"   -> {r.underwriter_name or '-'} ({r.underwriter_id or '-'})")
        print(f"   rationale: {r.rationale}")
        d = repo.get_latest_decision(r.application_id)
        for v in (d or {}).get("verification") or []:
            status = "PASS" if v["passed"] else "FAIL"
            print(f"   verify#{v['attempt']} [{v.get('model')}] {status} {'; '.join(v['issues'])[:220]}")
        if d and d.get("llm_errors"):
            for e in d["llm_errors"]:
                print(f"   llm_error: {e[:160]}")
        if r.fairness_signals:
            print(f"   fairness: {r.fairness_signals}")
        print()


if __name__ == "__main__":
    main()
