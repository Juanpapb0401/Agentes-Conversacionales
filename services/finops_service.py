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
    # Google Gemini
    "gemini-2.5-flash":                   {"input": 0.15,  "output": 0.60},
    "gemini-2.5-flash-preview-04-17":     {"input": 0.15,  "output": 0.60},
    "gemini-2.5-flash-lite-preview":      {"input": 0.075, "output": 0.30},
    "gemini-3.1-flash-lite-preview":      {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash":                   {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":                   {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-lite":              {"input": 0.075, "output": 0.30},
    # OpenAI
    "gpt-4o-mini":                        {"input": 0.15,  "output": 0.60},
    "gpt-4o":                             {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-11-20":                  {"input": 2.50,  "output": 10.00},
    # Anthropic Claude
    "claude-3-5-haiku-20241022":          {"input": 0.80,  "output": 4.00},
    "claude-3-5-sonnet-20241022":         {"input": 3.00,  "output": 15.00},
    "claude-3-haiku-20240307":            {"input": 0.25,  "output": 1.25},
    "claude-opus-4-5":                    {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001":          {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":                  {"input": 3.00,  "output": 15.00},
    # Groq (precios aproximados por 1M tokens)
    "llama-3.3-70b-versatile":            {"input": 0.59,  "output": 0.79},
    "llama-3.1-8b-instant":               {"input": 0.05,  "output": 0.08},
    "llama-3.1-70b-versatile":            {"input": 0.59,  "output": 0.79},
    "mixtral-8x7b-32768":                 {"input": 0.24,  "output": 0.24},
    "gemma2-9b-it":                       {"input": 0.20,  "output": 0.20},
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


# =============================================================================
# FinOps Analytics — funciones de análisis avanzado (puramente aditivas)
# =============================================================================

def get_all_sessions() -> Dict[str, Any]:
    """Devuelve todas las sesiones del log con sus datos completos."""
    return _load_log().get("sessions", {})


def get_session_calls(session_id: str) -> list:
    """Devuelve la lista de llamadas individuales de una sesión."""
    return _load_log()["sessions"].get(session_id, {}).get("calls", [])


def get_rich_report() -> Dict[str, Any]:
    """
    Genera un reporte analítico completo a partir de todos los datos históricos.
    Incluye estadísticas por servicio, por modelo, eficiencia de sesiones y
    listado plano de todas las llamadas con contexto de sesión.
    No modifica ningún dato existente.
    """
    data = _load_log()
    sessions = data.get("sessions", {})

    # ── Aplanar todas las llamadas con contexto de sesión ─────────────────────
    all_calls: list = []
    for sid, sess in sessions.items():
        for i, call in enumerate(sess.get("calls", [])):
            all_calls.append({**call, "session_id": sid, "call_seq": i + 1})

    # ── Estadísticas por servicio ─────────────────────────────────────────────
    service_stats: Dict[str, Any] = {}
    for call in all_calls:
        svc = call.get("service", "unknown")
        if svc not in service_stats:
            service_stats[svc] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        s = service_stats[svc]
        s["calls"]    += 1
        s["tokens_in"]  += call.get("tokens_in", 0)
        s["tokens_out"] += call.get("tokens_out", 0)
        s["cost_usd"]   = round(s["cost_usd"] + call.get("cost_usd", 0.0), 8)

    for svc, s in service_stats.items():
        if s["calls"] > 0:
            s["avg_tokens_in"]   = round(s["tokens_in"]  / s["calls"], 1)
            s["avg_tokens_out"]  = round(s["tokens_out"] / s["calls"], 1)
            s["avg_cost_usd"]    = round(s["cost_usd"]   / s["calls"], 8)
            s["efficiency_ratio"] = round(s["tokens_out"] / s["tokens_in"], 3) if s["tokens_in"] > 0 else 0.0

    # ── Estadísticas por modelo ───────────────────────────────────────────────
    model_stats: Dict[str, Any] = {}
    for call in all_calls:
        mdl = call.get("model", "unknown")
        if mdl not in model_stats:
            model_stats[mdl] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        m = model_stats[mdl]
        m["calls"]    += 1
        m["tokens_in"]  += call.get("tokens_in", 0)
        m["tokens_out"] += call.get("tokens_out", 0)
        m["cost_usd"]   = round(m["cost_usd"] + call.get("cost_usd", 0.0), 8)

    for mdl, m in model_stats.items():
        if m["calls"] > 0:
            m["avg_cost_per_call"] = round(m["cost_usd"] / m["calls"], 8)
            total_tokens = m["tokens_in"] + m["tokens_out"]
            m["cost_per_1k_tokens"] = round(m["cost_usd"] / total_tokens * 1000, 6) if total_tokens > 0 else 0.0

    # ── Eficiencia por sesión ─────────────────────────────────────────────────
    session_efficiency: Dict[str, Any] = {}
    for sid, sess in sessions.items():
        calls = sess.get("calls", [])
        n = len(calls)
        if n == 0:
            continue

        ti   = sess.get("total_tokens_in",  0)
        to_  = sess.get("total_tokens_out", 0)
        cost = sess.get("total_cost_usd",   0.0)

        eff_ratio = round(to_ / ti, 3) if ti > 0 else 0.0
        avg_cost  = round(cost / n, 8)

        # Duración y burn rate a partir de timestamps
        timestamps = sorted(c["timestamp"] for c in calls if c.get("timestamp"))
        burn_rate    = 0.0
        duration_min = 0.0
        if len(timestamps) >= 2:
            try:
                t0 = datetime.fromisoformat(timestamps[0])
                t1 = datetime.fromisoformat(timestamps[-1])
                secs = (t1 - t0).total_seconds()
                duration_min = round(secs / 60, 1)
                if secs > 0:
                    burn_rate = round(cost / (secs / 3600), 6)
            except ValueError:
                pass

        # Efficiency score 0–100:
        #   50 pts por ratio output/input (cap en 1.0)
        #   50 pts por bajo costo/llamada (ideal ≤ $0.000005)
        eff_score = min(eff_ratio, 1.0) * 50 + max(0.0, 50.0 - avg_cost * 10_000_000)
        eff_score = round(min(max(eff_score, 0.0), 100.0), 1)

        session_efficiency[sid] = {
            "total_cost_usd":    cost,
            "calls_count":       n,
            "efficiency_ratio":  eff_ratio,
            "avg_cost_per_call": avg_cost,
            "burn_rate_per_hour": burn_rate,
            "duration_min":      duration_min,
            "efficiency_score":  eff_score,
        }

    return {
        "global":             data.get("global", {"total_cost_usd": 0.0, "total_calls": 0}),
        "sessions_count":     len(sessions),
        "all_calls":          all_calls,
        "service_stats":      service_stats,
        "model_stats":        model_stats,
        "session_efficiency": session_efficiency,
    }


def get_cost_forecast(session_id: str, questions_remaining: int = 10) -> Dict[str, Any]:
    """
    Proyecta el costo futuro de cualquier sesión según su promedio actual por llamada.

    Args:
        session_id:          ID de la sesión a proyectar.
        questions_remaining: cuántas preguntas más planea hacer el usuario.

    Returns:
        Dict con costo actual, proyección adicional, total proyectado,
        presupuesto restante y si excederá el límite.
    """
    budget = float(os.environ.get("SESSION_BUDGET", "0.05"))
    sess   = _load_log()["sessions"].get(session_id, {})
    calls  = sess.get("calls", [])
    cost   = sess.get("total_cost_usd", 0.0)
    n      = len(calls)

    avg_per_call       = cost / n if n > 0 else 0.0
    projected_extra    = avg_per_call * questions_remaining
    projected_total    = cost + projected_extra
    budget_remaining   = max(0.0, budget - cost)
    calls_left_budget  = int(budget_remaining / avg_per_call) if avg_per_call > 0 else 999

    return {
        "current_cost_usd":          round(cost, 6),
        "avg_cost_per_call_usd":     round(avg_per_call, 8),
        "projected_additional_usd":  round(projected_extra, 6),
        "projected_total_usd":       round(projected_total, 6),
        "budget_usd":                budget,
        "budget_remaining_usd":      round(budget_remaining, 6),
        "calls_remaining_in_budget": calls_left_budget,
        "will_exceed_budget":        projected_total > budget,
        "questions_remaining":       questions_remaining,
    }
