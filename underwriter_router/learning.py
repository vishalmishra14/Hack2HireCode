"""Self-improvement loop (spec §8): learn from outcomes + overrides, hill-climb the scoring weights.

run_cycle():
  1. recompute_performance  observed SLA / turnaround / rework per underwriter, shrunk toward the
                            configured baseline (pseudo-count k) so a few cases can't swing it.
  2. objective(weights)     REPLAY every completed application's original decision snapshot with
                            candidate weights; reward each labelled underwriter by its rank position:
                              + outcome reward for whoever completed the case (SLA met = +1, missed = -1,
                                rework -0.5), -1 for a pick a supervisor overrode,
                              - fairness penalty if top-1 picks concentrate on one underwriter.
  3. hill_climb             coordinate moves of +/- step on each weight (renormalised to sum 1),
                            bounded to +/- max_delta from the current version, best-improvement ascent.
  4. gate                   accept only if train improves AND the chronological holdout strictly improves.
  5. version                every candidate is stored (accepted => active); rollback = activate_weights().
"""
from __future__ import annotations

from collections import Counter

from langsmith import traceable

from .config import WEIGHT_KEYS, Weights, settings
from .db import Repository
from .models import LoanApplication, Performance, Underwriter
from .rules import filter_eligible, rank
from .seed import SEED_UNDERWRITERS

_BASELINE = {u.id: u.performance for u in SEED_UNDERWRITERS}
MIN_OUTCOMES_PER_UNDERWRITER = 3


# --- 1. metrics ------------------------------------------------------------------------
def recompute_performance(repo: Repository) -> dict[str, dict]:
    k = settings.learn_prior_strength
    changes = {}
    stats = repo.outcome_stats()
    for uw in repo.list_underwriters():
        s = stats.get(uw.id)
        if not s or s["n"] < MIN_OUTCOMES_PER_UNDERWRITER:
            continue
        prior = _BASELINE.get(uw.id, uw.performance)
        n = s["n"]
        new = Performance(
            sla_hit_rate=round((n * s["sla_rate"] + k * prior.sla_hit_rate) / (n + k), 4),
            avg_turnaround_hours=round((n * s["avg_tat"] + k * prior.avg_turnaround_hours) / (n + k), 2),
            rework_rate=round((n * s["rework_rate"] + k * prior.rework_rate) / (n + k), 4),
        )
        if new != uw.performance:
            repo.update_performance(uw.id, new)
            changes[uw.id] = {"n": n, "before": uw.performance.model_dump(), "after": new.model_dump()}
    return changes


# --- 2. objective ------------------------------------------------------------------------
def prepare(samples: list[dict]) -> list[dict]:
    """Parse snapshots once and pre-apply hard filters (they don't depend on weights)."""
    out = []
    for s in samples:
        snap = s["snapshot"]
        app = LoanApplication(**snap["application"])
        eligible, _ = filter_eligible(app, [Underwriter(**u) for u in snap["underwriters"]])
        if len(eligible) >= 2:  # with one candidate there is nothing to learn
            out.append({"app": app, "eligible": eligible, "recent": snap["recent_counts"], "labels": s["labels"]})
    return out


def objective(w: dict[str, float], prepared: list[dict]) -> float:
    weights = Weights(version="candidate", **w)
    total, n_labels, top1 = 0.0, 0, Counter()
    for s in prepared:
        ranked = rank(s["app"], s["eligible"], s["recent"], settings.fairness_window, weights)
        n = len(ranked)
        pos = {c.underwriter_id: 1 - i / (n - 1) for i, c in enumerate(ranked)}
        top1[ranked[0].underwriter_id] += 1
        for uw_id, reward in s["labels"]:
            if uw_id in pos:
                total += reward * pos[uw_id]
                n_labels += 1
    if not n_labels:
        return 0.0
    max_share = max(top1.values()) / sum(top1.values())
    return total / n_labels - settings.learn_fairness_lambda * max(0.0, max_share - 0.3)


# --- 3. hill climbing ----------------------------------------------------------------------
def _normalise(w: dict[str, float]) -> dict[str, float]:
    s = sum(w.values())
    return {k: round(v / s, 4) for k, v in w.items()}


def hill_climb(start: dict[str, float], prepared: list[dict]) -> tuple[dict[str, float], float, list[dict]]:
    best, best_score = dict(start), objective(start, prepared)
    trail = [{"iter": 0, "weights": best, "score": round(best_score, 4)}]
    for it in range(1, settings.learn_max_iters + 1):
        step_best, step_score = None, best_score
        for key in WEIGHT_KEYS:
            for sign in (1, -1):
                cand = dict(best)
                cand[key] = cand[key] + sign * settings.learn_step
                if cand[key] < 0.02:
                    continue
                cand = _normalise(cand)
                if any(abs(cand[k] - start[k]) > settings.learn_max_delta + 1e-9 for k in WEIGHT_KEYS):
                    continue  # bounded update
                sc = objective(cand, prepared)
                if sc > step_score + 1e-6:
                    step_best, step_score = cand, sc
        if step_best is None:
            break  # local optimum
        best, best_score = step_best, step_score
        trail.append({"iter": it, "weights": best, "score": round(best_score, 4)})
    return best, best_score, trail


# --- 4/5. one learning cycle ------------------------------------------------------------------
@traceable(name="self_improvement_cycle", run_type="chain")
def run_cycle(repo: Repository) -> dict:
    perf_changes = recompute_performance(repo)
    samples = repo.learning_samples()
    current = repo.get_active_weights()
    report = {"samples": len(samples), "current_version": current.version, "performance_updates": perf_changes}
    if len(samples) < settings.learn_min_samples:
        return {**report, "status": "SKIPPED", "reason": f"need >= {settings.learn_min_samples} outcomes"}

    cut = int(len(samples) * (1 - settings.learn_holdout_ratio))  # chronological split
    train, holdout = prepare(samples[:cut]), prepare(samples[cut:])
    start = current.as_dict()
    base_train, base_hold = objective(start, train), objective(start, holdout)
    new, train_score, trail = hill_climb(start, train)
    hold_score = objective(new, holdout)

    # Strict gate: the change must generalise (holdout strictly better), not just fit the training data.
    accepted = new != start and train_score > base_train and hold_score > base_hold + 1e-6
    version = f"w-0.{len(repo.list_weights_versions()) + 1}"
    notes = {
        "baseline": {"train": round(base_train, 4), "holdout": round(base_hold, 4)},
        "trail": trail, "train_n": len(train), "holdout_n": len(holdout),
        "decision": "accepted" if accepted else
                    ("no improvement found" if new == start else "rejected: no holdout improvement"),
    }
    if new != start:  # store every real candidate for audit; only accepted ones go live
        repo.save_weights(Weights(version=version, **new), parent=current.version, train=round(train_score, 4),
                          holdout=round(hold_score, 4), notes=notes, activate=accepted)
    return {
        **report, "status": "ACCEPTED" if accepted else "REJECTED",
        "new_version": version if accepted else None, "weights_before": start, "weights_after": new,
        "train": {"before": round(base_train, 4), "after": round(train_score, 4)},
        "holdout": {"before": round(base_hold, 4), "after": round(hold_score, 4)},
        "iterations": len(trail) - 1, "decision": notes["decision"],
    }
