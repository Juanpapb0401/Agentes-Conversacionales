import contextvars
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

# Propagates session_id from app.py through the entire call stack (same thread/context)
SESSION_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "finops_session_id", default="unknown"
)

# USD per 1M tokens — approximate public pricing as of 2025
PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash":      {"input": 0.15,  "output": 0.60},
    "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60},
    "gemini-1.5-flash":      {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":      {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gpt-4o-mini":           {"input": 0.15,  "output": 0.60},
    "gpt-4o":                {"input": 2.50,  "output": 10.00},
}
_DEFAULT_PRICE: Dict[str, float] = {"input": 0.15, "output": 0.60}

LOG_PATH = os.path.join("data", "finops_log.json")

_EMPTY_LOG: Dict[str, Any] = {
    "sessions": {},
    "global": {"total_cost_usd": 0.0, "total_calls": 0},
}


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICING.get(model, _DEFAULT_PRICE)
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000


def _load_log() -> Dict[str, Any]:
    if not os.path.exists(LOG_PATH):
        return {"sessions": {}, "global": {"total_cost_usd": 0.0, "total_calls": 0}}
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"sessions": {}, "global": {"total_cost_usd": 0.0, "total_calls": 0}}


def _save_log(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_call(
    session_id: str,
    service: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """Log one LLM call and return its cost in USD."""
    cost = calculate_cost(model, tokens_in, tokens_out)
    data = _load_log()

    if session_id not in data["sessions"]:
        data["sessions"][session_id] = {
            "total_cost_usd": 0.0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "calls": [],
        }

    session = data["sessions"][session_id]
    session["calls"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 8),
    })
    session["total_cost_usd"] = round(session["total_cost_usd"] + cost, 8)
    session["total_tokens_in"] += tokens_in
    session["total_tokens_out"] += tokens_out

    data["global"]["total_cost_usd"] = round(
        data["global"]["total_cost_usd"] + cost, 8
    )
    data["global"]["total_calls"] += 1

    _save_log(data)
    return cost


def get_session_total(session_id: str) -> Dict[str, Any]:
    session = _load_log()["sessions"].get(session_id, {})
    return {
        "total_cost_usd": session.get("total_cost_usd", 0.0),
        "total_tokens_in": session.get("total_tokens_in", 0),
        "total_tokens_out": session.get("total_tokens_out", 0),
        "calls_count": len(session.get("calls", [])),
    }


def get_global_stats() -> Dict[str, Any]:
    return _load_log()["global"]
