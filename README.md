# Hack2HireCode: GenAI Loan Underwriter Router

Routes loan applications to underwriters. Hard rules decide **who is allowed**, GenAI decides **who is best**,
and every decision is recorded with a rationale and fairness signals. Full design: [spec.md](spec.md).

## Architecture (base loop)

```
load_context -> eligibility -> rank --+--(none)--------------------------------+
                                      +--(simple)---> genai_simple  (qwen3.5:2b) +--> commit -> END
                                      +--(complex)--> genai_complex (Deep Agent  |
                                                       + Claude Sonnet 5.5) -----+
```

| Piece | Tech | File |
|---|---|---|
| Agent loop / runtime | LangGraph `StateGraph` | `underwriter_router/graph.py` |
| Complex-decision harness | `deepagents.create_deep_agent` + read-only tools | `underwriter_router/optimizer.py` |
| Simple decisions | Ollama `qwen3.5:2b`, structured output | `underwriter_router/optimizer.py` |
| Rules engine (filters, scoring, complexity) | Plain Python, deterministic | `underwriter_router/rules.py` |
| Persistence and reliability | SQLite (idempotency, unique active assignment, atomic capacity, outbox) | `underwriter_router/db.py` |
| Tracing | LangSmith (automatic for every graph run) | `underwriter_router/config.py` |

**Fallback chain:** Claude Deep Agent → qwen → rules top-ranked. An LLM pick outside the shortlist, or one
with confidence below 0.5, is rejected. Routing never blocks on GenAI.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python on macOS/Linux
ollama pull qwen3.5:2b
```

`.env` (git-ignored):
```
LANGSMITH_API_KEY=...
CLAUDE_API_KEY=...
```

Optional overrides: `SIMPLE_MODEL`, `COMPLEX_MODEL` (format `provider:model`, e.g. `ollama:qwen3.5:2b`,
`anthropic:claude-sonnet-5-5`), `ROUTER_LLM_ENABLED=false`, `LANGSMITH_PROJECT`.

## Run

```bash
.venv/Scripts/python -m underwriter_router --reset          # demo: 4 applications + 1 duplicate
.venv/Scripts/python -m underwriter_router --reset --no-llm # rules only
.venv/Scripts/python -m pytest -q                           # tests (no network)
```

## Roadmap
- [x] Base loop: intake → eligibility → rank → GenAI pick → validated atomic assignment
- [x] Verification loop: 9 deterministic checks on each GenAI pick and rationale; 1 retry with feedback, then rules fallback (`underwriter_router/verifier.py`)
- [x] Event-driven loop: transactional outbox, at-least-once delivery with idempotent handlers, retry/backoff/dead-letter, stuck-routing sweeper, capacity-freed re-routing, supervisor overrides with mandatory reason (`underwriter_router/events.py`)
- [x] Self-improvement (hill-climbing) loop: metrics re-estimated from outcomes, weights hill-climbed by replaying past decisions, strict holdout gate, versioned + rollback (`underwriter_router/learning.py`)

```bash
.venv/Scripts/python -m underwriter_router.simulate   # both loops end to end + before/after A/B (~90 s)
```
