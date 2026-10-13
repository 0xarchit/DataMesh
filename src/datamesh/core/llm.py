import base64
import json
import os
import time

from openai import OpenAI

BASE_URL = os.environ.get("DATAMESH_LLM_BASE", "http://localhost:8080/v1")
MODEL_ENV = os.environ.get("DATAMESH_LLM_MODEL", "")
TIMEOUT = float(os.environ.get("DATAMESH_LLM_TIMEOUT", "300"))
MAX_TOKENS = int(os.environ.get("DATAMESH_LLM_MAX_TOKENS", "1536"))
THINKING = os.environ.get("DATAMESH_LLM_THINKING", "0") == "1"

_cached_model = MODEL_ENV


def client():
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("DATAMESH_LLM_KEY", "local"),
        timeout=TIMEOUT,
    )


def _extra():
    if THINKING:
        return {}
    return {
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }


def available(timeout=3.0):
    import requests

    try:
        r = requests.get(BASE_URL.rstrip("/") + "/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def model_name():
    global _cached_model
    if _cached_model:
        return _cached_model
    import requests

    try:
        data = requests.get(BASE_URL.rstrip("/") + "/models", timeout=5).json()
        items = data.get("data") or data.get("models") or []
        if items:
            _cached_model = items[0].get("id") or items[0].get("name") or "local"
    except Exception:
        pass
    if not _cached_model:
        _cached_model = "local"
    return _cached_model


def _drain(resp, on_token):
    parts = []
    for chunk in resp:
        if not getattr(chunk, "choices", None):
            continue
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            parts.append(text)
            on_token(text)
    return "".join(parts)


def chat(prompt, system=None, temperature=0.0, max_tokens=None, on_token=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or MAX_TOKENS,
        stream=on_token is not None,
        **_extra(),
    )
    if on_token is None:
        return resp.choices[0].message.content or ""
    return _drain(resp, on_token)


def chat_measured(prompt, system=None, temperature=0.0, max_tokens=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    started = time.time()
    resp = client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or MAX_TOKENS,
        stream=False,
        **_extra(),
    )
    elapsed = time.time() - started
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage is None:
        tokens = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "estimated": True}
    else:
        tokens = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "estimated": False,
        }
    return text, tokens, elapsed


def vision_extract(image_path, on_token=None):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    instruction = (
        "Extract the receipt as strict JSON with keys vendor, receipt_no, date, amount. "
        "amount must be an integer number of INR with no separators or symbols. "
        "Return only JSON, no explanation."
    )
    resp = client().chat.completions.create(
        model=model_name(),
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        stream=on_token is not None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                ],
            }
        ],
        **_extra(),
    )
    if on_token is None:
        return parse_json(resp.choices[0].message.content or "")
    return parse_json(_drain(resp, on_token))


def parse_json(text):
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}
