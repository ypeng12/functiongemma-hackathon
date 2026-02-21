import atexit
import json
import os
import re
import sys
import time

# Correct paths relative to the hackathon repository structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
cactus_src = os.path.join(PROJECT_ROOT, "..", "cactus", "python", "src")
sys.path.insert(0, cactus_src)

_MODEL_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "..", "cactus", "weights", "functiongemma-270m-it"),
    os.path.join(PROJECT_ROOT, "weights", "functiongemma-270m-it"),
    "weights/functiongemma-270m-it",
    os.path.expanduser("~/.cactus/weights/functiongemma-270m-it"),
]
functiongemma_path = None
for _p in _MODEL_PATH_CANDIDATES:
    if os.path.isdir(_p):
        functiongemma_path = _p
        break
if functiongemma_path is None:
    functiongemma_path = _MODEL_PATH_CANDIDATES[0]

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
    CACTUS_AVAILABLE = True
except Exception:
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


# ── Type helpers ──

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
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3).lower().replace(".", "")
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = hour if hour == 12 else hour + 12
    return hour, minute


def _parse_time_for_reminder(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3).upper().replace(".", "")
    return f"{hour}:{minute:02d} {ampm}"


# ── Improved extractors with broad triggers ──

_NOISE_WORDS = {'a', 'an', 'the', 'my', 'your', 'me', 'us', 'it', 'is', 'so', 'very', 'some', 'any', 'this', 'that', 'please', 'can', 'you', 'i'}


def _strip_location_noise(loc):
    loc = re.sub(
        r"\s+(?:right now|today|tomorrow|tonight|currently|please|now|outside|like|this week|this weekend)$",
        "", loc, flags=re.IGNORECASE
    ).strip()
    return loc


def _extract_weather(clause):
    low = clause.lower()
    triggers = [
        'weather', 'temperature', 'forecast', 'rain', 'raining',
        'snow', 'snowing', 'sunny', 'cloudy', 'cold', 'hot', 'warm',
        'humid', 'wind', 'windy', 'climate', 'degrees', 'chilly',
        'freezing', 'fog', 'foggy', 'stormy', 'hail', 'overcast',
        'drizzle', 'precipitation', 'heat',
    ]
    if not any(w in low for w in triggers):
        return None

    location = None
    patterns = [
        r'(?:weather|temperature|forecast|conditions|temp)\s*(?:like|going to be|gonna be)?\s*(?:in|at|for|of|near)\s+([A-Za-z][A-Za-z .\'-]+)',
        r"how'?s\s+(?:the\s+)?(?:weather|temperature)\s+(?:in|at|for|near)\s+([A-Za-z][A-Za-z .'\"-]+)",
        r'is\s+it\s+\w+\s+(?:in|at|near)\s+([A-Za-z][A-Za-z .\'-]+)',
        r'\b(?:in|at|for|near)\s+([A-Z][A-Za-z .\'-]+)',
        r'([A-Z][A-Za-z .\'-]+?)\s+(?:weather|temperature|forecast|conditions)',
        r'\bin\s+([A-Za-z][A-Za-z .\'-]+)',
    ]
    for pat in patterns:
        m = re.search(pat, clause, re.IGNORECASE)
        if m:
            loc = _strip_location_noise(_clean_phrase(m.group(1)))
            if loc and len(loc) >= 2 and loc.lower() not in _NOISE_WORDS:
                location = loc
                break

    if not location:
        return None
    return {"name": "get_weather", "arguments": {"location": location}}


def _extract_alarm(clause):
    low = clause.lower()
    if not any(t in low for t in ['alarm', 'wake me', 'wake up', 'wakeup']):
        return None
    parsed = _parse_time_for_alarm(clause)
    if not parsed:
        return None
    hour, minute = parsed
    return {"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}}


def _extract_timer(clause):
    low = clause.lower()
    if not any(t in low for t in ['timer', 'countdown', 'count down']):
        return None
    m = re.search(r'\b(\d{1,4})\s*(?:minutes?|mins?|min)\b', clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'\bfor\s+(\d{1,4})\b', clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'(\d{1,4})\s*[-]?\s*(?:minute|min)', clause, flags=re.IGNORECASE)
    if not m:
        return None
    return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}


def _extract_search_contacts(clause, context):
    low = clause.lower()
    has_contact_word = any(t in low for t in ['contact', 'contacts', 'look up', 'lookup', 'phone number', 'phone book', 'address book'])
    has_find = bool(re.search(r'\b(?:find|search)\b', low))

    if not has_contact_word and not has_find:
        return None
    # If only "find/search" without "contact", require a capitalized name
    if not has_contact_word and has_find:
        if not re.search(r'(?:find|search)\s+[A-Z]', clause):
            return None

    patterns = [
        r'(?:find|look\s*up|search(?:\s+for)?)\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:in|from)\s+(?:my\s+)?contacts?\b',
        r"(?:find|look\s*up|search(?:\s+for)?)\s+([A-Za-z][A-Za-z .']*?)(?:'s)?\s+(?:number|contact|info|phone|details)\b",
        r'(?:search|look\s+through)\s+(?:my\s+)?contacts?\s+(?:for|about)\s+([A-Za-z][A-Za-z .\'-]+)',
        r'(?:find|look\s*up|search(?:\s+for)?)\s+([A-Z][A-Za-z .\'-]*)',
        r"([A-Z][A-Za-z .']*?)\s+in\s+(?:my\s+)?contacts?\b",
        r'find\s+([A-Za-z][A-Za-z .\'-]*)',
    ]
    for pat in patterns:
        m = re.search(pat, clause, re.IGNORECASE)
        if m:
            query = _clean_phrase(m.group(1))
            if query and query.lower() not in _NOISE_WORDS:
                context["last_person"] = query
                return {"name": "search_contacts", "arguments": {"query": query}}
    return None


def _extract_send_message(clause, context):
    low = clause.lower()
    has_trigger = any(t in low for t in ['message', 'text ', ' text', '\btext\b', 'sms'])
    has_tell = bool(re.search(r'\btell\b', low))
    has_let_know = 'let ' in low and 'know' in low
    has_send = bool(re.search(r'\bsend\b', low))
    has_notify = 'notify' in low
    has_write = 'write to' in low

    if not (has_trigger or has_tell or has_let_know or (has_send and re.search(r'send\s+\w', low)) or has_notify or has_write):
        return None

    def _resolve_pronoun(name):
        if name.lower() in {"him", "her", "them"} and context.get("last_person"):
            return context["last_person"]
        return name

    patterns = [
        # "send [a] message to X saying/that Y"
        (r'(?:send\s+(?:a\s+)?message\s+to|text)\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:saying|that\s+says?|to\s+say|with\s+(?:the\s+)?message)\s+(.+)$', True),
        # "send X [a] message saying Y"
        (r'send\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:a\s+)?(?:message|text)\s+(?:saying|that\s+says?|to\s+say)\s+(.+)$', True),
        # "tell X [that] Y"
        (r'\btell\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:that\s+|to\s+say\s+)?(.+)$', True),
        # "let X know [that] Y"
        (r'\blet\s+([A-Za-z][A-Za-z .\'-]*?)\s+know\s+(?:that\s+)?(.+)$', True),
        # "notify X [that/about] Y"
        (r'\bnotify\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:that\s+|about\s+)?(.+)$', True),
        # "text X saying Y"
        (r'\btext\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:saying|that)\s+(.+)$', True),
        # "write to X saying Y"
        (r'\bwrite\s+to\s+([A-Za-z][A-Za-z .\'-]*?)\s+(?:saying|that)\s+(.+)$', True),
        # "send [a] message to X" (no message body)
        (r'(?:send\s+(?:a\s+)?message\s+to|text)\s+([A-Za-z][A-Za-z .\'-]*)$', False),
    ]
    for pat, has_msg in patterns:
        m = re.search(pat, clause, flags=re.IGNORECASE)
        if m:
            recipient = _clean_phrase(_resolve_pronoun(m.group(1)))
            message = _clean_phrase(m.group(2)) if has_msg else ""
            if recipient and recipient.lower() not in {'me', 'us', 'it', 'myself'}:
                return {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}
    return None


def _extract_play_music(clause):
    low = clause.lower()
    if not any(t in low for t in ['play', 'listen', 'put on', 'queue']):
        return None

    patterns = [
        r'\bplay\s+(.+)$',
        r'\blisten\s+to\s+(.+)$',
        r'\bput\s+on\s+(.+)$',
        r'\bqueue\s+(?:up\s+)?(.+)$',
    ]
    for pat in patterns:
        m = re.search(pat, clause, flags=re.IGNORECASE)
        if m:
            raw_song = _clean_phrase(m.group(1))
            had_some = bool(re.match(r'^some\s+', raw_song, flags=re.IGNORECASE))
            song = re.sub(r'^some\s+', '', raw_song, flags=re.IGNORECASE)
            if had_some:
                song = re.sub(r'\s+music$', '', song, flags=re.IGNORECASE)
            song = re.sub(r'\s+(song|playlist|track|album)$', '', song, flags=re.IGNORECASE)
            song = _clean_phrase(song)
            if song and song.lower() not in _NOISE_WORDS:
                return {"name": "play_music", "arguments": {"song": song}}
    return None


def _extract_reminder(clause):
    low = clause.lower()
    if not any(t in low for t in ['remind', 'reminder', "don't forget", 'do not forget', 'remember to']):
        return None
    time_text = _parse_time_for_reminder(clause)
    if not time_text:
        return None

    patterns = [
        r"(?:remind me|don'?t (?:let me )?forget|do not forget|remember)\s+(?:about|to)?\s*(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b",
        r"(?:set\s+(?:a\s+)?reminder\s+(?:for|about|to))\s+(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"(?:reminder)\s+(?:for|about|to)\s+(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    ]
    for pat in patterns:
        m = re.search(pat, clause, flags=re.IGNORECASE)
        if m:
            title = _strip_leading_article(_clean_phrase(m.group(1)))
            if title:
                return {"name": "create_reminder", "arguments": {"title": title, "time": time_text}}
    return None


# ── Generic tool matcher for unknown tools ──

def _generic_extract(clause, tool):
    """Try to match a clause to a tool based on tool name/description keywords."""
    low = clause.lower()
    name = tool["name"]
    desc = tool.get("description", "")

    # Build keywords from tool name and description
    name_words = set(name.replace("_", " ").split())
    desc_words = set(re.findall(r'[a-z]+', desc.lower()))
    # Remove very common words
    stopwords = {'a', 'an', 'the', 'for', 'to', 'of', 'in', 'is', 'and', 'or', 'by', 'with', 'on', 'at', 'get', 'set', 'create'}
    keywords = (name_words | desc_words) - stopwords

    # Score: how many keywords appear in the clause
    score = sum(1 for kw in keywords if kw in low)
    if score < 1:
        return None

    # Try to extract arguments
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])
    args = {}

    for param_name, param_info in props.items():
        ptype = param_info.get("type", "string")
        pdesc = param_info.get("description", "").lower()

        if ptype == "string":
            # Try "in/at/for/to WORDS" pattern for location-like params
            if any(loc_word in pdesc for loc_word in ['location', 'city', 'place', 'area']):
                m = re.search(r'\b(?:in|at|for|near|of)\s+([A-Z][A-Za-z .\'-]+)', clause)
                if m:
                    args[param_name] = _clean_phrase(m.group(1))
            # Try "to PERSON" for recipient-like params
            elif any(p_word in pdesc for p_word in ['person', 'recipient', 'contact', 'name', 'who']):
                m = re.search(r'\b(?:to|for)\s+([A-Z][A-Za-z .\'-]+)', clause)
                if m:
                    args[param_name] = _clean_phrase(m.group(1))
            # Generic: try to find a quoted string or a noun phrase
            else:
                m = re.search(r'"([^"]+)"', clause)
                if not m:
                    m = re.search(r"'([^']+)'", clause)
                if m:
                    args[param_name] = _clean_phrase(m.group(1))

        elif ptype == "integer":
            m = re.search(r'\b(\d+)\b', clause)
            if m:
                args[param_name] = int(m.group(1))

    # Check if we have all required args
    for req in required:
        if req not in args:
            return None

    return {"name": name, "arguments": args}


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
    tool_by_name = {t["name"]: t for t in tools}
    context = {"last_person": None}
    calls = []
    strong_hits = 0

    # Known extractors
    extractors = [
        _extract_search_contacts,
        _extract_send_message,
        _extract_weather,
        _extract_alarm,
        _extract_timer,
        _extract_reminder,
        _extract_play_music,
    ]

    for clause in clauses:
        matched = False
        for extractor in extractors:
            try:
                call = extractor(clause, context) if extractor in (_extract_search_contacts, _extract_send_message) else extractor(clause)
            except Exception:
                call = None
            if call and call["name"] in allowed:
                calls.append(call)
                strong_hits += 1
                matched = True
                break

        # Fallback: generic matcher for unmatched clauses
        if not matched:
            for tool in tools:
                if tool["name"] in allowed:
                    try:
                        call = _generic_extract(clause, tool)
                    except Exception:
                        call = None
                    if call:
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
            confidence_threshold=0.0,
            tool_rag_top_k=0,
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
    return rule["confidence"] >= 0.75


def _should_accept_local(local, messages, tools, _confidence_threshold):
    calls = local.get("function_calls", [])
    if local.get("cloud_handoff"):
        return False
    if not _calls_schema_valid(calls, tools):
        return False
    confidence = float(local.get("confidence", 0.0) or 0.0)
    intent_est = _estimate_intent_count(messages)
    if intent_est >= 2 and len(calls) < intent_est:
        return False
    return confidence >= (0.50 if intent_est == 1 else 0.60)


def generate_hybrid(messages, tools, confidence_threshold=0.70):
    """
    Local-first strategy:
      1) Rule-based fast path (on-device, near-zero latency)
      2) Neural Cactus inference (on-device, respects cloud_handoff signal)
      3) Rule-based backup (on-device, if confidence >= 0.70)
      4) Gemini cloud fallback (highest accuracy for hard cases)
    """
    total_start = time.perf_counter()

    # Step 1: Rule-based fast path (~0ms, on-device)
    rule = _rule_based_calls(messages, tools)
    if _should_use_rule_result(rule, messages, tools):
        return {
            "function_calls": rule["function_calls"],
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
            "confidence": rule["confidence"],
            "source": "on-device",
        }

    # Step 2: Neural on-device via Cactus (only when rule-based fails)
    local = generate_cactus(messages, tools)
    local["function_calls"] = _normalize_calls(local.get("function_calls", []), tools)

    if _should_accept_local(local, messages, tools, confidence_threshold):
        local["source"] = "on-device"
        return local

    # Rule backup before paying cloud latency
    if rule["function_calls"] and _calls_schema_valid(rule["function_calls"], tools) and rule["confidence"] >= 0.70:
        return {
            "function_calls": rule["function_calls"],
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
            "confidence": rule["confidence"],
            "source": "on-device",
        }

    # Cloud fallback (Gemini)
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
