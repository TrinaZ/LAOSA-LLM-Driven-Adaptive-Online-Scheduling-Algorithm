"""
llm_adapter.py — Multi-provider LLM adapter for LAOSA
Supports: DeepSeek (cheapest), OpenAI, Anthropic Claude, ZhipuAI

Priority: DEEPSEEK_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY > simulated

Usage:
  export DEEPSEEK_API_KEY=sk-xxx    # Option 1: DeepSeek (recommended, cheapest)
  export OPENAI_API_KEY=sk-xxx      # Option 2: OpenAI
  export ANTHROPIC_API_KEY=sk-xxx   # Option 3: Anthropic
  # If none set, uses simulated responses
"""
import os
import json
import http.client


def detect_provider():
    """Auto-detect which LLM provider is available."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek", os.environ["DEEPSEEK_API_KEY"]
    elif os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ["OPENAI_API_KEY"]
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    elif os.environ.get("ZHIPU_API_KEY"):
        return "zhipu", os.environ["ZHIPU_API_KEY"]
    else:
        return "simulated", ""


PROVIDER_CONFIG = {
    "deepseek": {
        "endpoint": "api.deepseek.com",
        "model": "deepseek-chat",       # DeepSeek-V3, very cheap
        "path": "/v1/chat/completions",
    },
    "openai": {
        "endpoint": "api.openai.com",
        "model": "gpt-4o-mini",          # Cheapest OpenAI model
        "path": "/v1/chat/completions",
    },
    "zhipu": {
        "endpoint": "open.bigmodel.cn",
        "model": "glm-4-flash",          # Free tier available
        "path": "/api/paas/v4/chat/completions",
    },
}


def call_llm(prompt: str, provider: str = None, api_key: str = None,
             max_tokens: int = 1024) -> str:
    """
    Call LLM API. Auto-detects provider if not specified.
    Returns generated text or None on failure.
    """
    if provider is None:
        provider, api_key = detect_provider()

    if provider == "simulated":
        return None  # Caller handles simulated fallback

    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, max_tokens)

    # OpenAI-compatible API (DeepSeek, OpenAI, Zhipu all use this format)
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        print(f"  [Unknown provider: {provider}]")
        return None

    return _call_openai_compatible(
        prompt, api_key,
        endpoint=config["endpoint"],
        model=config["model"],
        path=config["path"],
        max_tokens=max_tokens,
    )


def _call_openai_compatible(prompt, api_key, endpoint, model, path, max_tokens):
    """Call OpenAI-compatible API (works for DeepSeek, OpenAI, Zhipu)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    import time
    for attempt in range(3):
        try:
            conn = http.client.HTTPSConnection(endpoint, timeout=60)
            conn.request("POST", path, payload, headers)
            res = conn.getresponse()
            raw_body = res.read()
            data = json.loads(raw_body)
            
            # 检查是否是错误响应
            if "error" in data and "choices" not in data:
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                err_type = err.get("type", "") if isinstance(err, dict) else ""
                print(f"  [API error: {err_type}: {err_msg[:100]}]")
                # 余额不足：直接放弃（不用重试）
                if "balance" in err_msg.lower() or "insufficient" in err_msg.lower():
                    print("  [余额不足，请充值: https://platform.deepseek.com/top_up]")
                    return None
                # rate limit：等待后重试
                if "rate" in err_msg.lower() or res.status == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None
            
            text = data["choices"][0]["message"]["content"]
            conn.close()
            return _extract_code(text)
        except KeyError as e:
            # 打印完整响应帮助诊断
            try:
                print(f"  [API attempt {attempt+1} failed: '{e}' key missing]")
                print(f"  [Full response: {json.dumps(data)[:300]}]")
            except Exception:
                print(f"  [API attempt {attempt+1} failed: {e}]")
            continue
        except Exception as e:
            print(f"  [API attempt {attempt+1} failed: {type(e).__name__}: {e}]")
            continue
    return None


def _call_anthropic(prompt, api_key, max_tokens):
    """Call Anthropic Claude API."""
    payload = json.dumps({
        "model": "claude-sonnet-4-5"  # current model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            conn = http.client.HTTPSConnection("api.anthropic.com", timeout=60)
            conn.request("POST", "/v1/messages", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read())
            text = data["content"][0]["text"]
            conn.close()
            return _extract_code(text)
        except Exception as e:
            print(f"  [Anthropic attempt {attempt+1} failed: {type(e).__name__}: {e}]")
            continue
    return None


def _extract_code(raw_text: str) -> str:
    """Extract Python function code from LLM response."""
    if "```python" in raw_text:
        code = raw_text.split("```python")[1].split("```")[0]
    elif "```" in raw_text and "def priority" in raw_text:
        code = raw_text.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.split("```")[0]
    elif "def priority" in raw_text:
        start = raw_text.index("def priority")
        code = raw_text[start:]
        # Find end of function
        lines = code.split("\n")
        func_lines = [lines[0]]
        for line in lines[1:]:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                break
            func_lines.append(line)
        code = "\n".join(func_lines)
    else:
        code = raw_text

    return code.strip()


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    provider, key = detect_provider()
    print(f"Detected provider: {provider}")
    if provider != "simulated":
        print(f"API key: {key[:10]}...")
        result = call_llm("Write a Python function that returns 42. Only return the code.")
        print(f"Response: {result[:200] if result else 'FAILED'}")
    else:
        print("No API key found. Set one of:")
        print("  export DEEPSEEK_API_KEY=sk-xxx  (recommended, cheapest)")
        print("  export OPENAI_API_KEY=sk-xxx")
        print("  export ANTHROPIC_API_KEY=sk-xxx")
