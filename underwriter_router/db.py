"""SQLite persistence with the reliability guarantees from spec §9.1.

- idempotency_key UNIQUE                      -> no duplicate applications
- partial UNIQUE index on active assignment   -> no duplicate routings
- conditional capacity UPDATE                 -> no over-allocation under races
- status claim (SUBMITTED -> ROUTING)          -> only one worker routes an application
- routing_decisions is append-only            -> audit trail
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Weights
from .models import (
    AppStatus,
    LoanApplication,
    LoanApplicationIn,
    Performance,
    Underwriter,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    submitter_id TEXT NOT NULL,
    loan_type TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    region TEXT NOT NULL,
    priority TEXT NOT NULL,
    risk_band TEXT NOT NULL,
    complexity_flags TEXT NOT NULL,
    status TEXT NOT NULL,
    sla_due_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS underwriters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    skills TEXT NOT NULL,
    licensed_regions TEXT NOT NULL,
    authority_limit REAL NOT NULL,
    max_active_cases INTEGER NOT NULL,
    active_cases INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    performance TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL REFERENCES applications(id),
    attempt INTEGER NOT NULL,
    input_snapshot TEXT NOT NULL,
    rules_version TEXT NOT NULL,
    weights_version TEXT NOT NULL,
    complexity TEXT,
    eligible_candidates TEXT NOT NULL,
    excluded_candidates TEXT NOT NULL,
    shortlist TEXT NOT NULL,
    llm_model TEXT,
    llm_prompt_version TEXT,
    llm_response TEXT,
    llm_errors TEXT,
    verification TEXT,
    selected_underwriter_id TEXT,
    decision_source TEXT,
    rationale TEXT NOT NULL,
    fairness_signals TEXT NOT NULL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assignments (
    id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL REFERENCES applications(id),
    underwriter_id TEXT NOT NULL REFERENCES underwriters(id),
    routing_decision_id TEXT NOT NULL REFERENCES routing_decisions(id),
    active INTEGER NOT NULL DEFAULT 1,
    assigned_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_assignment_active
    ON assignments(application_id) WHERE active = 1;

CREATE TABLE IF NOT EXISTS outbox_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',      -- PENDING | IN_PROGRESS | DONE | FAILED
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    available_at TEXT,                           -- visibility timeout / retry backoff
    last_error TEXT,
    processed_at TEXT
);

-- Idempotent consumer: one notification per event, even if the event is delivered twice.
CREATE TABLE IF NOT EXISTS notifications (
    event_id TEXT PRIMARY KEY,
    recipient TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    application_id TEXT PRIMARY KEY REFERENCES applications(id),
    underwriter_id TEXT NOT NULL,
    sla_met INTEGER NOT NULL,
    turnaround_hours REAL NOT NULL,
    reworked INTEGER NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS overrides (
    id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL REFERENCES applications(id),
    from_underwriter_id TEXT NOT NULL,
    to_underwriter_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    reason_text TEXT NOT NULL,
    forced INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weights_versions (
    version TEXT PRIMARY KEY,
    parent_version TEXT,
    weights TEXT NOT NULL,
    train_score REAL,
    holdout_score REAL,
    notes TEXT,
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

# Columns added after the first schema version (applied to existing DBs on startup).
MIGRATIONS = {
    "routing_decisions": {"verification": "TEXT"},
    "applications": {"routing_started_at": "TEXT"},
    "outbox_events": {"available_at": "TEXT", "last_error": "TEXT", "processed_at": "TEXT"},
    "assignments": {"completed_at": "TEXT"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CapacityExhausted(Exception):
    """The chosen underwriter filled up between ranking and commit."""


class Repository:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # -- connection -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def tx(self):
        """Write transaction; BEGIN IMMEDIATE serialises writers in SQLite."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            if conn.execute("SELECT COUNT(*) FROM weights_versions").fetchone()[0] == 0:
                w = Weights()
                conn.execute(
                    "INSERT INTO weights_versions (version, weights, notes, active, created_at) VALUES (?,?,?,1,?)",
                    (w.version, json.dumps(w.as_dict()), "initial hand-set weights", now_iso()),
                )
        finally:
            conn.close()

    @staticmethod
    def _emit(conn: sqlite3.Connection, event_type: str, payload: dict, ts: str) -> None:
        """Transactional outbox: the event commits (or rolls back) with the state change."""
        conn.execute(
            "INSERT INTO outbox_events (id, event_type, payload, created_at, available_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), event_type, json.dumps(payload), ts, ts),
        )

    # -- applications -------------------------------------------------------
    def create_application(self, app: LoanApplicationIn, sla_hours: int) -> tuple[LoanApplication, bool]:
        """Idempotent insert. Returns (application, created)."""
        existing = self.get_application_by_key(app.idempotency_key)
        if existing:
            return existing, False
        created = datetime.now(timezone.utc)
        app_id = str(uuid.uuid4())
        try:
            with self.tx() as conn:
                conn.execute(
                    """INSERT INTO applications (id, idempotency_key, submitter_id, loan_type, amount, currency,
                       region, priority, risk_band, complexity_flags, status, sla_due_at, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        app_id, app.idempotency_key, app.submitter_id, app.loan_type.value, app.amount,
                        app.currency, app.region, app.priority.value, app.risk_band.value,
                        json.dumps(sorted(app.complexity_flags)), AppStatus.SUBMITTED.value,
                        (created + timedelta(hours=sla_hours)).isoformat(), created.isoformat(),
                    ),
                )
                # If the process dies before routing, the event loop still routes it.
                self._emit(conn, "APPLICATION_SUBMITTED", {"application_id": app_id}, created.isoformat())
        except sqlite3.IntegrityError:
            # Lost a race with a concurrent submit of the same key.
            return self.get_application_by_key(app.idempotency_key), False
        return self.get_application(app_id), True

    def _row_to_app(self, row: sqlite3.Row | None) -> LoanApplication | None:
        if row is None:
            return None
        d = dict(row)
        d["complexity_flags"] = json.loads(d["complexity_flags"])
        return LoanApplication(**d)

    def get_application(self, app_id: str) -> LoanApplication | None:
        conn = self._connect()
        try:
            return self._row_to_app(conn.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone())
        finally:
            conn.close()

    def get_application_by_key(self, key: str) -> LoanApplication | None:
        conn = self._connect()
        try:
            return self._row_to_app(
                conn.execute("SELECT * FROM applications WHERE idempotency_key=?", (key,)).fetchone()
            )
        finally:
            conn.close()

    def claim_for_routing(self, app_id: str) -> bool:
        """SUBMITTED/UNROUTABLE -> ROUTING. Only one caller wins."""
        with self.tx() as conn:
            cur = conn.execute(
                "UPDATE applications SET status=?, routing_started_at=?, version=version+1 "
                "WHERE id=? AND status IN (?,?)",
                (AppStatus.ROUTING.value, now_iso(), app_id, AppStatus.SUBMITTED.value, AppStatus.UNROUTABLE.value),
            )
            return cur.rowcount == 1

    # -- underwriters -------------------------------------------------------
    def upsert_underwriter(self, uw: Underwriter) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO underwriters (id, name, skills, licensed_regions, authority_limit,
                   max_active_cases, active_cases, status, performance) VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, skills=excluded.skills,
                   licensed_regions=excluded.licensed_regions, authority_limit=excluded.authority_limit,
                   max_active_cases=excluded.max_active_cases, status=excluded.status,
                   performance=excluded.performance""",
                (
                    uw.id, uw.name, json.dumps(uw.skills), json.dumps(uw.licensed_regions), uw.authority_limit,
                    uw.max_active_cases, uw.active_cases, uw.status, uw.performance.model_dump_json(),
                ),
            )

    def list_underwriters(self) -> list[Underwriter]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM underwriters ORDER BY id").fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["skills"] = json.loads(d["skills"])
            d["licensed_regions"] = json.loads(d["licensed_regions"])
            d["performance"] = Performance(**json.loads(d["performance"]))
            out.append(Underwriter(**d))
        return out

    def recent_assignment_counts(self, window: int) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT underwriter_id, COUNT(*) AS n FROM (
                       SELECT underwriter_id FROM assignments ORDER BY assigned_at DESC LIMIT ?
                   ) GROUP BY underwriter_id""",
                (window,),
            ).fetchall()
        finally:
            conn.close()
        return {r["underwriter_id"]: r["n"] for r in rows}

    # -- decisions & assignment ----------------------------------------------
    def commit_assignment(self, app_id: str, underwriter_id: str, decision: dict, events: list[dict]) -> str:
        """Atomically: write decision, reserve capacity, create assignment, mark ASSIGNED, enqueue events.

        Raises CapacityExhausted if the underwriter filled up since ranking.
        """
        decision_id = str(uuid.uuid4())
        ts = now_iso()
        with self.tx() as conn:
            cur = conn.execute(
                """UPDATE underwriters SET active_cases = active_cases + 1
                   WHERE id=? AND status='AVAILABLE' AND active_cases < max_active_cases""",
                (underwriter_id,),
            )
            if cur.rowcount != 1:
                raise CapacityExhausted(underwriter_id)
            self._insert_decision(conn, decision_id, app_id, decision, ts)
            conn.execute(
                "INSERT INTO assignments (id, application_id, underwriter_id, routing_decision_id, active, assigned_at)"
                " VALUES (?,?,?,?,1,?)",
                (str(uuid.uuid4()), app_id, underwriter_id, decision_id, ts),
            )
            cur = conn.execute(
                "UPDATE applications SET status=?, version=version+1 WHERE id=? AND status=?",
                (AppStatus.ASSIGNED.value, app_id, AppStatus.ROUTING.value),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"Application {app_id} not in ROUTING state; aborting assignment")
            for ev in events:
                self._emit(conn, ev["event_type"], ev["payload"], ts)
        return decision_id

    def mark_unroutable(self, app_id: str, decision: dict) -> str:
        decision_id = str(uuid.uuid4())
        ts = now_iso()
        with self.tx() as conn:
            self._insert_decision(conn, decision_id, app_id, decision, ts)
            conn.execute(
                "UPDATE applications SET status=?, version=version+1 WHERE id=?",
                (AppStatus.UNROUTABLE.value, app_id),
            )
            self._emit(conn, "SUPERVISOR_ALERT_UNROUTABLE", {"application_id": app_id}, ts)
        return decision_id

    @staticmethod
    def _insert_decision(conn: sqlite3.Connection, decision_id: str, app_id: str, d: dict, ts: str) -> None:
        attempt = conn.execute(
            "SELECT COALESCE(MAX(attempt), 0) + 1 FROM routing_decisions WHERE application_id=?", (app_id,)
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO routing_decisions (id, application_id, attempt, input_snapshot, rules_version,
               weights_version, complexity, eligible_candidates, excluded_candidates, shortlist, llm_model,
               llm_prompt_version, llm_response, llm_errors, verification, selected_underwriter_id,
               decision_source, rationale, fairness_signals, latency_ms, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision_id, app_id, attempt, json.dumps(d["input_snapshot"]), d["rules_version"],
                d["weights_version"], d.get("complexity"), json.dumps(d["eligible"]), json.dumps(d["excluded"]),
                json.dumps(d["shortlist"]), d.get("llm_model"), d.get("llm_prompt_version"),
                json.dumps(d.get("llm_response")), json.dumps(d.get("llm_errors", [])),
                json.dumps(d.get("verification", [])),
                d.get("selected_underwriter_id"), d.get("decision_source"), d["rationale"],
                json.dumps(d["fairness_signals"]), d.get("latency_ms"), ts,
            ),
        )

    def get_latest_decision(self, app_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM routing_decisions WHERE application_id=? ORDER BY attempt DESC LIMIT 1", (app_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        for k in ("input_snapshot", "eligible_candidates", "excluded_candidates", "shortlist",
                  "llm_response", "llm_errors", "verification", "fairness_signals"):
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        return d

    def get_active_assignment(self, app_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM assignments WHERE application_id=? AND active=1", (app_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    # -- event loop: outbox consumption -------------------------------------------
    def claim_events(self, limit: int, visibility_s: float) -> list[dict]:
        """Claim due events. A claimed event becomes invisible for `visibility_s`; if the worker
        dies before ack, it re-appears (at-least-once delivery, like an SQS visibility timeout)."""
        now = datetime.now(timezone.utc)
        with self.tx() as conn:
            rows = conn.execute(
                """SELECT * FROM outbox_events WHERE status IN ('PENDING','IN_PROGRESS')
                   AND available_at <= ? ORDER BY created_at, id LIMIT ?""",
                (now.isoformat(), limit),
            ).fetchall()
            hidden_until = (now + timedelta(seconds=visibility_s)).isoformat()
            for r in rows:
                conn.execute(
                    "UPDATE outbox_events SET status='IN_PROGRESS', attempts=attempts+1, available_at=? WHERE id=?",
                    (hidden_until, r["id"]),
                )
        return [{**dict(r), "payload": json.loads(r["payload"]), "attempts": r["attempts"] + 1} for r in rows]

    def ack_event(self, event_id: str) -> None:
        with self.tx() as conn:
            conn.execute("UPDATE outbox_events SET status='DONE', processed_at=? WHERE id=?", (now_iso(), event_id))

    def nack_event(self, event_id: str, error: str, attempts: int, max_attempts: int, backoff_s: float) -> str:
        """Retry with exponential backoff; dead-letter (FAILED) after max_attempts."""
        status = "FAILED" if attempts >= max_attempts else "PENDING"
        retry_at = (datetime.now(timezone.utc) + timedelta(seconds=backoff_s * 2 ** (attempts - 1))).isoformat()
        with self.tx() as conn:
            conn.execute("UPDATE outbox_events SET status=?, last_error=?, available_at=? WHERE id=?",
                         (status, error[:500], retry_at, event_id))
        return status

    def record_notification(self, event_id: str, recipient: str, message: str) -> bool:
        """Returns False if this event was already delivered (duplicate delivery is a no-op)."""
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO notifications (event_id, recipient, message, created_at) VALUES (?,?,?,?)",
                (event_id, recipient, message, now_iso()),
            )
            return cur.rowcount == 1

    def expire_routing_leases(self, lease_s: float) -> list[str]:
        """Sweeper: applications stuck in ROUTING past the lease go back to SUBMITTED and are re-queued."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=lease_s)).isoformat()
        ts = now_iso()
        with self.tx() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM applications WHERE status='ROUTING' AND routing_started_at < ?", (cutoff,))]
            for app_id in ids:
                conn.execute("UPDATE applications SET status='SUBMITTED', version=version+1 "
                             "WHERE id=? AND status='ROUTING'", (app_id,))
                self._emit(conn, "APPLICATION_SUBMITTED",
                           {"application_id": app_id, "reason": "routing lease expired"}, ts)
        return ids

    def list_application_ids(self, status: AppStatus, limit: int = 1000) -> list[str]:
        conn = self._connect()
        try:
            return [r["id"] for r in conn.execute(
                "SELECT id FROM applications WHERE status=? ORDER BY created_at LIMIT ?", (status.value, limit))]
        finally:
            conn.close()

    def event_counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            return {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) n FROM outbox_events GROUP BY status")}
        finally:
            conn.close()

    # -- case lifecycle: completion & override ---------------------------------------
    def complete_case(self, app_id: str, sla_met: bool, turnaround_hours: float, reworked: bool) -> str:
        """Close the active assignment, free capacity, record the outcome (learning signal)."""
        ts = now_iso()
        with self.tx() as conn:
            a = conn.execute("SELECT * FROM assignments WHERE application_id=? AND active=1", (app_id,)).fetchone()
            if a is None:
                raise ValueError(f"{app_id} has no active assignment")
            conn.execute("UPDATE assignments SET active=0, completed_at=? WHERE id=?", (ts, a["id"]))
            conn.execute("UPDATE underwriters SET active_cases = MAX(active_cases - 1, 0) WHERE id=?",
                         (a["underwriter_id"],))
            conn.execute("UPDATE applications SET status=?, version=version+1 WHERE id=?",
                         (AppStatus.COMPLETED.value, app_id))
            conn.execute(
                "INSERT INTO outcomes (application_id, underwriter_id, sla_met, turnaround_hours, reworked, completed_at)"
                " VALUES (?,?,?,?,?,?)",
                (app_id, a["underwriter_id"], int(sla_met), turnaround_hours, int(reworked), ts),
            )
            self._emit(conn, "CASE_COMPLETED", {"application_id": app_id, "underwriter_id": a["underwriter_id"],
                                                "sla_met": sla_met}, ts)
            self._emit(conn, "CAPACITY_FREED", {"underwriter_id": a["underwriter_id"]}, ts)
        return a["underwriter_id"]

    def override_assignment(
        self, app_id: str, to_uw: str, actor_id: str, reason_code: str, reason_text: str, forced: bool, decision: dict
    ) -> str:
        """Supervisor reassignment: swap assignment + capacity atomically, new MANUAL_OVERRIDE decision."""
        ts = now_iso()
        with self.tx() as conn:
            a = conn.execute("SELECT * FROM assignments WHERE application_id=? AND active=1", (app_id,)).fetchone()
            if a is None:
                raise ValueError(f"{app_id} has no active assignment to override")
            if a["underwriter_id"] == to_uw:
                raise ValueError("target underwriter already holds this application")
            cap_check = "" if forced else " AND status='AVAILABLE' AND active_cases < max_active_cases"
            if conn.execute(f"UPDATE underwriters SET active_cases = active_cases + 1 WHERE id=?{cap_check}",
                            (to_uw,)).rowcount != 1:
                raise CapacityExhausted(to_uw)
            conn.execute("UPDATE underwriters SET active_cases = MAX(active_cases - 1, 0) WHERE id=?",
                         (a["underwriter_id"],))
            conn.execute("UPDATE assignments SET active=0, completed_at=? WHERE id=?", (ts, a["id"]))
            decision_id = str(uuid.uuid4())
            self._insert_decision(conn, decision_id, app_id, decision, ts)
            conn.execute(
                "INSERT INTO assignments (id, application_id, underwriter_id, routing_decision_id, active, assigned_at)"
                " VALUES (?,?,?,?,1,?)", (str(uuid.uuid4()), app_id, to_uw, decision_id, ts))
            conn.execute(
                "INSERT INTO overrides (id, application_id, from_underwriter_id, to_underwriter_id, actor_id,"
                " reason_code, reason_text, forced, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), app_id, a["underwriter_id"], to_uw, actor_id, reason_code, reason_text,
                 int(forced), ts))
            self._emit(conn, "ASSIGNMENT_OVERRIDDEN", {
                "application_id": app_id, "from_underwriter_id": a["underwriter_id"], "to_underwriter_id": to_uw,
                "reason_code": reason_code, "reason_text": reason_text}, ts)
        return decision_id

    # -- self-improvement loop: data + versioned weights ---------------------------------
    def learning_samples(self) -> list[dict]:
        """One sample per completed application: the FIRST routing decision's snapshot plus labels.

        labels: list of (underwriter_id, reward). The outcome reward goes to whoever completed the
        case; an override adds -1 for the original pick (the supervisor disagreed with it)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT o.*, d.input_snapshot FROM outcomes o
                   JOIN routing_decisions d ON d.application_id = o.application_id AND d.attempt = (
                       SELECT MIN(attempt) FROM routing_decisions x
                       WHERE x.application_id = o.application_id AND x.selected_underwriter_id IS NOT NULL)
                   ORDER BY d.created_at, o.application_id"""
            ).fetchall()
            ovr: dict[str, list[str]] = {}
            for r in conn.execute("SELECT * FROM overrides ORDER BY created_at"):
                ovr.setdefault(r["application_id"], []).append(r["from_underwriter_id"])
        finally:
            conn.close()
        out = []
        for r in rows:
            reward = (1.0 if r["sla_met"] else -1.0) - (0.5 if r["reworked"] else 0.0)
            labels = [(r["underwriter_id"], reward)] + [(u, -1.0) for u in ovr.get(r["application_id"], [])]
            out.append({"application_id": r["application_id"], "snapshot": json.loads(r["input_snapshot"]),
                        "labels": labels})
        return out

    def outcome_stats(self) -> dict[str, dict]:
        conn = self._connect()
        try:
            return {r["underwriter_id"]: dict(r) for r in conn.execute(
                """SELECT underwriter_id, COUNT(*) n, AVG(sla_met) sla_rate, AVG(turnaround_hours) avg_tat,
                   AVG(reworked) rework_rate FROM outcomes GROUP BY underwriter_id""")}
        finally:
            conn.close()

    def update_performance(self, uw_id: str, perf: Performance) -> None:
        with self.tx() as conn:
            conn.execute("UPDATE underwriters SET performance=? WHERE id=?", (perf.model_dump_json(), uw_id))

    def get_active_weights(self) -> Weights:
        conn = self._connect()
        try:
            r = conn.execute("SELECT version, weights FROM weights_versions WHERE active=1").fetchone()
        finally:
            conn.close()
        return Weights(version=r["version"], **json.loads(r["weights"])) if r else Weights()

    def save_weights(self, w: Weights, parent: str | None, train: float, holdout: float, notes: dict,
                     activate: bool) -> None:
        with self.tx() as conn:
            if activate:
                conn.execute("UPDATE weights_versions SET active=0")
            conn.execute(
                "INSERT INTO weights_versions (version, parent_version, weights, train_score, holdout_score, notes,"
                " active, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (w.version, parent, json.dumps(w.as_dict()), train, holdout, json.dumps(notes), int(activate),
                 now_iso()),
            )

    def activate_weights(self, version: str) -> None:
        """Rollback / roll-forward to any stored version."""
        with self.tx() as conn:
            if not conn.execute("SELECT 1 FROM weights_versions WHERE version=?", (version,)).fetchone():
                raise ValueError(f"unknown weights version {version}")
            conn.execute("UPDATE weights_versions SET active = (version = ?)", (version,))

    def list_weights_versions(self) -> list[dict]:
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM weights_versions ORDER BY created_at")]
        finally:
            conn.close()
