import sys, os
# Get the absolute path to the parent directory (Hackathongoogle)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cactus_src = os.path.join(base_path, "python/src")
functiongemma_path = os.path.join(base_path, "weights/functiongemma-270m-it")

import sys, os
# Get the absolute path to the parent directory (Hackathongoogle)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cactus_src = os.path.join(base_path, "python/src")
functiongemma_path = os.path.join(base_path, "weights/functiongemma-270m-it")
sys.path.insert(0, cactus_src)

import json, os, time
from dotenv import load_dotenv
load_dotenv()

# Gracefully handle Cactus import on unsupported architectures (e.g. Intel Mac)
try:
    from cactus import cactus_init, cactus_complete, cactus_destroy
    CACTUS_AVAILABLE = True
except (OSError, ImportError, Exception):
    CACTUS_AVAILABLE = False
    def cactus_init(*args, **kwargs): raise RuntimeError("Cactus not available on this architecture")
    def cactus_complete(*args, **kwargs): pass
    def cactus_destroy(*args, **kwargs): pass

from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    try:
        model = cactus_init(functiongemma_path)
    except (OSError, ImportError, Exception) as e:
        # Graceful fallback for unsupported architectures (e.g. Intel Mac)
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "error": str(e)
        }

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
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
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()
    
    models_to_try = ["gemini-flash-latest"]
    gemini_response = None
    
    for attempt in range(5): # Up to 5 retries
        for model_name in models_to_try:
            try:
                gemini_response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(tools=gemini_tools),
                )
                if gemini_response:
                    break
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    print(f"  Model {model_name} busy or quota hit. Trying next...")
                    continue
                raise e
        
        if gemini_response:
            break
            
        wait_time = (attempt + 1) * 0.1
        print(f"  All models hit limits. Waiting {wait_time}s before retry {attempt+1}/5...")
        time.sleep(wait_time)

    if not gemini_response:
        raise RuntimeError("Failed to get response from any Gemini model after multiple retries.")

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    local = generate_cactus(messages, tools)

    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)