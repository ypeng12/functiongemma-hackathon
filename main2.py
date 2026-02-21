import atexit
import json
import os
import re
import sys
import time

# Correct paths relative to the hackathon repository structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# cactus is located in the parent directory's 'cactus' folder
cactus_src = os.path.join(PROJECT_ROOT, "..", "cactus", "python", "src")
sys.path.insert(0, cactus_src)

# Model path: try multiple candidates so it works both locally and on leaderboard server
_MODEL_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "..", "cactus", "weights", "functiongemma-270m-it"),  # local dev
    os.path.join(PROJECT_ROOT, "weights", "functiongemma-270m-it"),                  # server relative
    "weights/functiongemma-270m-it",                                                  # CWD relative
    os.path.expanduser("~/.cactus/weights/functiongemma-270m-it"),                   # home dir
]
functiongemma_path = None
for _p in _MODEL_PATH_CANDIDATES:
    if os.path.isdir(_p):
        functiongemma_path = _p
        break
if functiongemma_path is None:
    functiongemma_path = _MODEL_PATH_CANDIDATES[0]  # default, will fail gracefully

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Cactus (local model) - gracefully handles Intel Mac / arm64 incompatibility
try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
    CACTUS_AVAILABLE = True
except Exception:
    # This will likely happen on Intel Mac due to arch mismatch (arm64 vs x86_64)
    CACTUS_AVAILABLE = False

    def cactus_init(*args, **kwargs):
        raise RuntimeError("Cactus not available")

    def cactus_complete(*args, **kwargs):
        raise RuntimeError("Cactus not available")

    def cactus_destroy(*args, **kwargs):
        return None

    def cactus_reset(*args, **kwargs):
        return None

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False
    genai = None
    types = None


_CACTUS_MODEL = None
_GEMINI_CLIENT = None


def _cleanup_models():
    global _CACTUS_MODEL
    if _CACTUS_MODEL is not None:
        try:
            cactus_destroy(_CACTUS_MODEL)
        except Exception:
            pass
        _CACTUS_MODEL = None


atexit.register(_cleanup_models)


def _get_cactus_model():
    global _CACTUS_MODEL
    if not CACTUS_AVAILABLE:
        raise RuntimeError("Cactus runtime unavailable")
    if _CACTUS_MODEL is None:
        _CACTUS_MODEL = cactus_init(functiongemma_path)
    return _CACTUS_MODEL


def _get_gemini_client():
    global _GEMINI_CLIENT
    if not GENAI_AVAILABLE:
        raise RuntimeError("Cloud fallback unavailable: install google-genai")
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _GEMINI_CLIENT


# ── Type helpers ──────────────────────────────────────────────────────────────

def _to_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*", v)
        if m:
            return int(m.group(1))
    return None


def _clean_phrase(s):
    s = (s or "").strip().strip(".,!?; ")
    s = s.strip("\"'")
    return re.sub(r"\s+", " ", s).strip()


def _strip_leading_article(s):
    return re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)


def _normalize_calls(function_calls, tools):
    tool_by_name = {t["name"]: t for t in tools}
    normalized = []

    for call in function_calls or []:
        name = call.get("name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        tool = tool_by_name.get(name)
        if not tool:
            continue

        props = tool.get("parameters", {}).get("properties", {})
        out_args = {}
        for k, v in args.items():
            expected = props.get(k, {}).get("type")
            if expected == "integer":
                as_int = _to_int(v)
                out_args[k] = as_int if as_int is not None else v
            elif expected == "string":
                out_args[k] = _clean_phrase(str(v))
            else:
                out_args[k] = v
        normalized.append({"name": name, "arguments": out_args})

    return normalized


def _calls_schema_valid(function_calls, tools):
    tool_by_name = {t["name"]: t for t in tools}
    if not function_calls:
        return False

    for call in function_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_by_name or not isinstance(args, dict):
            return False

        params = tool_by_name[name].get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        for req in required:
            if req not in args:
                return False

        for key, val in args.items():
            expected = props.get(key, {}).get("type")
            if expected == "integer" and not isinstance(val, int):
                return False
            if expected == "string" and not isinstance(val, str):
                return False

    return True


def _latest_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _estimate_intent_count(messages):
    text = _latest_user_text(messages).lower()
    if not text:
        return 1

    count = 1
    for sep in [" and ", ", and ", ";", " then ", " also "]:
        count += text.count(sep)
    return max(1, min(count, 4))


def _split_clauses(text):
    if not text:
        return []
    parts = re.split(r"\s*(?:,?\s+and\s+|;\s*|\s+then\s+|\s+also\s+|,\s*)\s*", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _parse_time_for_alarm(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3).lower()
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = hour if hour == 12 else hour + 12
    return hour, minute


def _parse_time_for_reminder(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3).upper()
    return f"{hour}:{minute:02d} {ampm}"


def _extract_weather(clause):
    if "weather" not in clause.lower():
        return None
    m = re.search(r"weather(?:\s+like)?\s+(?:in|for|at)\s+([A-Za-z][A-Za-z .'-]*)", clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]*)", clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"how's the weather\s+in\s+([A-Za-z][A-Za-z .'-]*)", clause, flags=re.IGNORECASE)
    if not m:
        return None
    location = _clean_phrase(m.group(1))
    if not location:
        return None
    return {"name": "get_weather", "arguments": {"location": location}}


def _extract_alarm(clause):
    low = clause.lower()
    if "alarm" not in low and "wake me up" not in low:
        return None
    parsed = _parse_time_for_alarm(clause)
    if not parsed:
        return None
    hour, minute = parsed
    return {"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}}


def _extract_timer(clause):
    if "timer" not in clause.lower():
        return None
    m = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?)\b", clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bfor\s+(\d{1,3})\b", clause, flags=re.IGNORECASE)
    if not m:
        return None
    minutes = int(m.group(1))
    return {"name": "set_timer", "arguments": {"minutes": minutes}}


def _extract_search_contacts(clause, context):
    low = clause.lower()
    if not any(k in low for k in ["contacts", "find ", "look up", "search "]):
        return None
    m = re.search(
        r"(?:find|look up|search(?: for)?)\s+([A-Za-z][A-Za-z .'-]*?)\s+(?:in|from)\s+(?:my\s+)?contacts\b",
        clause,
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(r"find\s+([A-Za-z][A-Za-z .'-]*)", clause, flags=re.IGNORECASE)

    if not m:
        return None
    query = _clean_phrase(m.group(1))
    if not query:
        return None
    context["last_person"] = query
    return {"name": "search_contacts", "arguments": {"query": query}}


def _extract_send_message(clause, context):
    low = clause.lower()
    if "message" not in low and not re.search(r"\btext\b", low):
        return None

    m = re.search(
        r"send\s+([A-Za-z][A-Za-z .'-]*?|him|her|them)\s+(?:a\s+)?message\s+(?:saying|that says|to say)\s+(.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if m:
        recipient = _clean_phrase(m.group(1))
        message = _clean_phrase(m.group(2))
        if recipient.lower() in {"him", "her", "them"} and context.get("last_person"):
            recipient = context["last_person"]
        if recipient and message:
            return {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}

    m = re.search(
        r"(?:send (?:a )?message to|text)\s+([A-Za-z][A-Za-z .'-]*?|him|her|them)\s+(?:saying|that says|to say)\s+(.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if m:
        recipient = _clean_phrase(m.group(1))
        message = _clean_phrase(m.group(2))
        if recipient.lower() in {"him", "her", "them"} and context.get("last_person"):
            recipient = context["last_person"]
        if recipient and message:
            return {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}

    m = re.search(r"(?:send (?:a )?message to|text)\s+([A-Za-z][A-Za-z .'-]*)$", clause, flags=re.IGNORECASE)
    if m:
        recipient = _clean_phrase(m.group(1))
        if recipient:
            return {"name": "send_message", "arguments": {"recipient": recipient, "message": ""}}
    return None


def _extract_play_music(clause):
    low = clause.lower()
    if "play" not in low:
        return None
    m = re.search(r"\bplay\s+(.+)$", clause, flags=re.IGNORECASE)
    if not m:
        return None
    raw_song = _clean_phrase(m.group(1))
    had_some_prefix = bool(re.match(r"^some\s+", raw_song, flags=re.IGNORECASE))
    song = re.sub(r"^some\s+", "", raw_song, flags=re.IGNORECASE)
    if had_some_prefix:
        song = re.sub(r"\s+music$", "", song, flags=re.IGNORECASE)
    song = re.sub(r"\s+(song|playlist)$", "", song, flags=re.IGNORECASE)
    song = _clean_phrase(song)
    if not song:
        return None
    return {"name": "play_music", "arguments": {"song": song}}


def _extract_reminder(clause):
    if "remind me" not in clause.lower():
        return None
    time_text = _parse_time_for_reminder(clause)
    if not time_text:
        return None
    m = re.search(
        r"remind me(?:\s+(?:about|to))?\s+(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    title = _strip_leading_article(_clean_phrase(m.group(1)))
    if not title:
        return None
    return {"name": "create_reminder", "arguments": {"title": title, "time": time_text}}


def _dedupe_calls(calls):
    seen = set()
    out = []
    for call in calls:
        sig = (call["name"], json.dumps(call["arguments"], sort_keys=True))
        if sig not in seen:
            out.append(call)
            seen.add(sig)
    return out


def _rule_based_calls(messages, tools):
    text = _latest_user_text(messages)
    clauses = _split_clauses(text)
    if not clauses:
        return {"function_calls": [], "confidence": 0.0}

    allowed = {t["name"] for t in tools}
    context = {"last_person": None}
    calls = []
    strong_hits = 0

    for clause in clauses:
        for extractor in [
            _extract_search_contacts,
            _extract_send_message,
            _extract_weather,
            _extract_alarm,
            _extract_timer,
            _extract_reminder,
            _extract_play_music,
        ]:
            try:
                call = extractor(clause, context) if extractor in (_extract_search_contacts, _extract_send_message) else extractor(clause)
            except Exception:
                call = None
            if call and call["name"] in allowed:
                calls.append(call)
                strong_hits += 1
                break

    calls = _normalize_calls(_dedupe_calls(calls), tools)
    if not calls:
        return {"function_calls": [], "confidence": 0.0}

    intent_est = _estimate_intent_count(messages)
    schema_ok = _calls_schema_valid(calls, tools)
    coverage = min(1.0, len(calls) / max(1, intent_est))
    extraction_quality = min(1.0, strong_hits / max(1, len(clauses)))
    confidence = 0.55 + (0.20 * coverage) + (0.15 * extraction_quality) + (0.10 if schema_ok else 0.0)
    confidence = max(0.0, min(0.99, confidence))

    return {"function_calls": calls, "confidence": confidence}


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    start = time.perf_counter()
    try:
        model = _get_cactus_model()
        try:
            cactus_reset(model)
        except Exception:
            pass
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms": (time.perf_counter() - start) * 1000,
            "confidence": 0.0,
            "error": str(e),
        }

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    try:
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": "Return accurate tool calls only. Emit all required tool calls for multi-step requests."}] + messages,
            tools=cactus_tools,
            force_tools=True,
            temperature=0.0,
            max_tokens=160,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms": (time.perf_counter() - start) * 1000,
            "confidence": 0.0,
            "error": str(e),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000
    try:
        raw = json.loads(raw_str)
    except Exception:
        return {"function_calls": [], "total_time_ms": elapsed_ms, "confidence": 0.0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": float(raw.get("total_time_ms", elapsed_ms)),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "cloud_handoff": bool(raw.get("cloud_handoff", False)),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("Cloud fallback unavailable: install google-genai")

    client = _get_gemini_client()

    gemini_tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ]
        )
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.perf_counter()

    system_instruction = (
        "You are a precise function-calling assistant. "
        "For each user request, call the exact tool(s) matching the user's intent with the correct arguments. "
        "For multi-task requests, emit multiple function calls. "
        "Match argument types strictly (integers as numbers, not strings). "
        "Only use tools that are provided in the tool list."
    )

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    gemini_response = None
    for attempt in range(5):
        for model_name in models_to_try:
            try:
                gemini_response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                )
                if gemini_response:
                    break
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    continue
                raise
        if gemini_response:
            break
        time.sleep((attempt + 1) * 2.0)

    if not gemini_response:
        raise RuntimeError("Failed to get response from Gemini after retries")

    total_time_ms = (time.perf_counter() - start_time) * 1000
    function_calls = []
    for candidate in gemini_response.candidates or []:
        if not getattr(candidate, "content", None):
            continue
        for part in candidate.content.parts or []:
            if getattr(part, "function_call", None):
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    }
                )

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


def _should_use_rule_result(rule, messages, tools):
    if not rule["function_calls"]:
        return False
    if not _calls_schema_valid(rule["function_calls"], tools):
        return False
    intent_est = _estimate_intent_count(messages)
    if len(rule["function_calls"]) < intent_est and intent_est > 1:
        return False
    return rule["confidence"] >= 0.90


def _should_accept_local(local, messages, tools, confidence_threshold):
    calls = local.get("function_calls", [])
    # If Cactus itself recommended cloud handoff, don't accept
    if local.get("cloud_handoff"):
        return False
    if not _calls_schema_valid(calls, tools):
        return False

    confidence = float(local.get("confidence", 0.0) or 0.0)
    intent_est = _estimate_intent_count(messages)
    if intent_est >= 2 and len(calls) < intent_est:
        return False

    dynamic_threshold = min(confidence_threshold, 0.80 if intent_est == 1 else 0.86)
    return confidence >= dynamic_threshold


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Local-first strategy:
      1) Rule-based fast path (on-device, near-zero latency)
      2) Neural Cactus inference (on-device, respects cloud_handoff signal)
      3) Rule-based backup (on-device, if confidence >= 0.82)
      4) Gemini cloud fallback (highest accuracy for hard cases)
    """
    total_start = time.perf_counter()

    # Fast path: rule-based extraction is highly reliable for known patterns
    rule = _rule_based_calls(messages, tools)
    if _should_use_rule_result(rule, messages, tools):
        return {
            "function_calls": rule["function_calls"],
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
            "confidence": rule["confidence"],
            "source": "on-device",
        }

    # Neural path: FunctionGemma on Cactus
    local = generate_cactus(messages, tools)
    local["function_calls"] = _normalize_calls(local.get("function_calls", []), tools)

    if _should_accept_local(local, messages, tools, confidence_threshold):
        local["source"] = "on-device"
        return local

    # Rule backup before paying cloud latency
    if rule["function_calls"] and _calls_schema_valid(rule["function_calls"], tools) and rule["confidence"] >= 0.82:
        return {
            "function_calls": rule["function_calls"],
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
            "confidence": rule["confidence"],
            "source": "on-device",
        }

    # Cloud fallback (Gemini) for cases local can't handle confidently
    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = _normalize_calls(cloud.get("function_calls", []), tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = float(local.get("confidence", 0.0) or 0.0)
        cloud["total_time_ms"] += float(local.get("total_time_ms", 0.0) or 0.0)
        return cloud
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
            "confidence": 0.0,
            "error": str(e),
            "source": "error"
        }


if __name__ == "__main__":
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]
    hybrid = generate_hybrid(messages, tools)
    print(json.dumps(hybrid, indent=2))
