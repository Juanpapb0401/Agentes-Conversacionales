# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Conversational AI agent for analyzing digital conversations from social media datasets. Users ask natural language questions; the agent routes them to one of four analytical microservices and synthesizes the response.

## Commands

**Setup:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys
```

**Run (two terminals required):**
```bash
# Terminal 1 — FastAPI backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app.py
```

**Test:**
```bash
pytest tests/ -v          # run all 23 tests
pytest tests/test_tools.py -v -k "test_name"  # single test
```

**API docs:** `http://127.0.0.1:8000/docs`

## Architecture

The system has four layers:

1. **Streamlit UI** (`app.py`) — Multi-turn chat interface. Manages `thread_id` per browser session, streams agent responses, renders intermediate tool steps.

2. **LangGraph Agent** (`agent/graph.py`, `agent/tools.py`) — Stateful graph with two nodes: `llamar_modelo` → `ejecutar_herramienta`. LLM decides which tool to invoke; MemorySaver persists full message history keyed by `thread_id`. Supports both OpenAI and Gemini via `LLM_PROVIDER` env var.

3. **FastAPI services** (`main.py`, `routers/`, `services/`) — Four endpoints:
   - `POST /analisis/sentimientos` — LLM-based sentiment (climate, score, justification)
   - `POST /analisis/resumen` — LLM-based conversation summary
   - `GET /analisis/propagacion?post_id=X` — BFS propagation tree with impact scoring
   - `POST /analisis/metricas` — Pandas aggregations (likes, influencers, platforms)

4. **Data layer** (`data_loader.py`) — Loads `.parquet` dataset as a global singleton `dataframe_principal`. Maps 30+ column aliases (Brandwatch/Talkwalker format) to 7 canonical names; handles epoch ms/s and ISO timestamp formats.

### Agent → API call flow

```
User message → graph.invoke(HumanMessage, thread_id)
  → LLM evaluates system_prompt + history + tool definitions
  → if tool_calls: ToolNode runs @tool function via httpx → FastAPI
  → loops back to LLM with ToolMessage
  → final AIMessage (no tool_calls) → Streamlit
```

### Propagation algorithm

BFS builds full descendant tree from `post_id`. Impact score is weighted: reach 35%, unique users 20%, content matching 15%, velocity 15%, engagement 15%. Level labels: Muy Alto (≥75), Alto (50–74), Medio (25–49), Bajo (<25).

## Key Conventions

- **Tool functions** in `agent/tools.py` use `httpx` to call FastAPI; they're decorated with `@tool` and prefixed `tool_`.
- **LLM provider** is selected at graph construction time via `LLM_PROVIDER=gemini|openai`; both Gemini native and LangChain-OpenAI are supported.
- **JSON extraction** from LLM responses uses `_safe_json_parse()` in `nlp_service.py` to handle markdown-wrapped JSON.
- **Column normalization** in `data_loader.py` is the single source of truth for dataset field names — update aliases there before changing service queries.
- **Pydantic schemas** in `schemas.py` define all request/response contracts; keep them in sync with service return dicts.
- **Tests** mock the FastAPI HTTP layer (`respx`) for tool tests and use a real in-memory graph for graph tests.

## FinOps Session Tracking (fix applied 2026-05-10)

Session tracking for the FinOps cost dashboard had an intermittent bug where the `X-Session-ID` header
sometimes arrived as `"unknown"` at FastAPI endpoints, causing orphaned or missing log entries.

**Root cause:** `tools.py` read session ID from a `ContextVar` (`SESSION_CTX`) that did not reliably
propagate across the async thread pool used by LangGraph's `ToolNode`.

**Fix:**
- `agent/tools.py` — added `_active_session_id` module variable and `_tool_call_wrapper` that
  intercepts every tool call, reads `session_id` from `request.runtime.config.configurable`, and
  sets `_active_session_id` before tool execution (resets to `"unknown"` in `finally`).
- `agent/graph.py` — `ToolNode` is now created with `wrap_tool_call=wrap_tool_call` to activate the interceptor.
- `app.py:193` — config now passes both `thread_id` and `session_id` in `configurable`.

**Important:** Do not use `SESSION_CTX` (`_SESSION_CTX.get()`) for the tool-call HTTP path —
all FinOps HTTP headers now route through `_active_session_id` set by the wrapper.
