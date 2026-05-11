import json
import os
import re
import urllib.error
import urllib.request
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

from services.finops_service import log_call as _finops_log

load_dotenv()

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()

MAX_PROMPT_CHARS = 12000


def analizar_sentimiento_llm(texto: str, idioma: str = "es", session_id: str = "unknown") -> Dict[str, Any]:
    if not texto or not texto.strip():
        return {
            "clima": "neutral",
            "score": 0.5,
            "justificacion": "No hay texto suficiente para evaluar."
        }

    system_prompt = (
        "Eres un analista de sentimientos para conversaciones digitales. "
        "Devuelve SOLO JSON valido y en espanol con claves: clima, score, justificacion. "
        "clima debe ser: positivo, negativo o neutral. score debe estar entre 0 y 1."
    )

    user_prompt = (
        f"Idioma esperado: {idioma}.\n"
        "Texto a analizar:\n"
        f"{texto}\n\n"
        "Responde solo JSON, sin markdown ni texto adicional."
    )

    raw = _llm_call(system_prompt, user_prompt, max_tokens=250, session_id=session_id, service="sentimiento")
    parsed = _safe_json_parse(raw)

    if not parsed:
        return {
            "clima": "neutral",
            "score": 0.5,
            "justificacion": "No fue posible interpretar la salida del modelo."
        }

    return {
        "clima": str(parsed.get("clima", "neutral")).lower(),
        "score": float(parsed.get("score", 0.5)),
        "justificacion": str(parsed.get("justificacion", ""))
    }


def resumir_conversacion_llm(
    textos: List[str],
    max_palabras: int = 120,
    idioma: str = "es",
    session_id: str = "unknown",
) -> Dict[str, Any]:
    textos = [t for t in textos if t and t.strip()]
    if not textos:
        return {
            "resumen": "No hay textos para resumir.",
            "temas_principales": [],
            "posturas_clave": [],
            "alcance_textos": 0
        }

    contenido = "\n---\n".join(textos)
    contenido = contenido[:MAX_PROMPT_CHARS]

    system_prompt = (
        "Eres un analista de conversaciones digitales. "
        "Debes sintetizar la conversacion con rigor y sin inventar datos. "
        "Devuelve SOLO JSON valido y en espanol con claves: resumen, temas_principales, posturas_clave. "
        "temas_principales y posturas_clave deben ser listas de strings."
    )

    user_prompt = (
        f"Idioma esperado: {idioma}.\n"
        f"Limite aproximado del resumen: {max_palabras} palabras.\n\n"
        "Textos de la conversacion:\n"
        f"{contenido}\n\n"
        "Responde solo JSON, sin markdown ni texto adicional."
    )

    raw = _llm_call(system_prompt, user_prompt, max_tokens=1024, session_id=session_id, service="resumen")
    parsed = _safe_json_parse(raw)

    if not parsed:
        return {
            "resumen": "No fue posible interpretar la salida del modelo.",
            "temas_principales": [],
            "posturas_clave": [],
            "alcance_textos": len(textos)
        }

    return {
        "resumen": str(parsed.get("resumen", "")),
        "temas_principales": list(parsed.get("temas_principales", [])),
        "posturas_clave": list(parsed.get("posturas_clave", [])),
        "alcance_textos": len(textos)
    }


def _llm_call(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    session_id: str = "unknown",
    service: str = "llm",
) -> str:
    provider = DEFAULT_PROVIDER
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, max_tokens=max_tokens, session_id=session_id, service=service)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, max_tokens=max_tokens, session_id=session_id, service=service)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, max_tokens=max_tokens, session_id=session_id, service=service)
    if provider == "groq":
        return _call_groq(system_prompt, user_prompt, max_tokens=max_tokens, session_id=session_id, service=service)

    raise RuntimeError("LLM_PROVIDER no esta configurado. Use: openai, gemini, anthropic o groq.")


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    session_id: str = "unknown",
    service: str = "llm",
    model: Optional[str] = None,
    json_mode: bool = True,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no esta configurado")

    model = model or DEFAULT_OPENAI_MODEL
    url = "https://api.openai.com/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            _finops_log(session_id, service, model,
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    session_id: str = "unknown",
    service: str = "llm",
    model: Optional[str] = None,
    json_mode: bool = True,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no esta configurado")

    model = model or DEFAULT_GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    gen_config: Dict[str, Any] = {
        "temperature": 0.2,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json" if json_mode else "text/plain",
    }
    # thinkingConfig only supported by Gemini 2.5+ models
    if model and "2.5" in model:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    payload: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": gen_config,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usageMetadata", {})
            _finops_log(session_id, service, model,
                        usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))
            candidates = data.get("candidates", [])
            if not candidates:
                return "{}" if json_mode else ""
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    session_id: str = "unknown",
    service: str = "llm",
    model: Optional[str] = None,
    json_mode: bool = True,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY no esta configurado")

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    # Cuando json_mode=True, pedimos explícitamente JSON en el system prompt
    sys_final = system_prompt
    if json_mode:
        sys_final = system_prompt + "\nResponde SOLO con JSON valido, sin markdown ni texto adicional."

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": sys_final,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            _finops_log(session_id, service, model,
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {detail}") from exc


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    session_id: str = "unknown",
    service: str = "llm",
    model: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    import httpx

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no esta configurado")

    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            _finops_log(session_id, service, model,
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Groq HTTP {exc.response.status_code}: {exc.response.text}") from exc


# =============================================================================
# API pública para comparación directa de modelos
# =============================================================================

def call_model_direct(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    max_tokens: int = 500,
    session_id: str = "unknown",
    service: str = "comparacion",
    json_mode: bool = False,
) -> str:
    """
    Llama a un modelo específico de cualquier proveedor soportado.
    Diseñado para la página de comparación multi-modelo.

    Proveedores soportados: openai, gemini, anthropic, groq
    """
    p = provider.strip().lower()
    if p == "openai":
        return _call_openai(system_prompt, user_prompt, max_tokens=max_tokens,
                            session_id=session_id, service=service, model=model, json_mode=json_mode)
    if p == "gemini":
        return _call_gemini(system_prompt, user_prompt, max_tokens=max_tokens,
                            session_id=session_id, service=service, model=model, json_mode=json_mode)
    if p == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, max_tokens=max_tokens,
                               session_id=session_id, service=service, model=model, json_mode=json_mode)
    if p == "groq":
        return _call_groq(system_prompt, user_prompt, max_tokens=max_tokens,
                          session_id=session_id, service=service, model=model, json_mode=False)
    raise RuntimeError(f"Proveedor '{provider}' no soportado. Use: openai, gemini, anthropic, groq")


def _safe_json_parse(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
