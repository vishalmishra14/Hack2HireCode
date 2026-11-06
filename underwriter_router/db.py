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
    status TEXT NOT NULL DEFAULT 'PENDING',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


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
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(routing_decisions)")}
            if "verification" not in cols:  # migrate DBs created before the verification loop
                conn.execute("ALTER TABLE routing_decisions ADD COLUMN verification TEXT")
        finally:
            conn.close()

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
                "UPDATE applications SET status=?, version=version+1 WHERE id=? AND status IN (?,?)",
                (AppStatus.ROUTING.value, app_id, AppStatus.SUBMITTED.value, AppStatus.UNROUTABLE.value),
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
                conn.execute(
                    "INSERT INTO outbox_events (id, event_type, payload, created_at) VALUES (?,?,?,?)",
                    (str(uuid.uuid4()), ev["event_type"], json.dumps(ev["payload"]), ts),
                )
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
            conn.execute(
                "INSERT INTO outbox_events (id, event_type, payload, created_at) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), "SUPERVISOR_ALERT_UNROUTABLE", json.dumps({"application_id": app_id}), ts),
            )
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
