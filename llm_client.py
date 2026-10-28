"""One function, ask(), that talks to any of the three tiers."""
import os
import threading
import time
from openai import OpenAI, RateLimitError
import config

# Create the client once and reuse it for all calls.
# timeout: give up on a hung request after 60s (the SDK default is 10 minutes).
# max_retries=0: the SDK's hidden retries are off; ask() does its own 3 tries.
client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY,
                timeout=60.0, max_retries=0)

# Rate limiter: calls start at least 60/MAX_RPM seconds apart (NVIDIA free tier = 40 requests/min)
MAX_RPM = int(os.getenv("MAX_RPM", "40"))
_rate_lock = threading.Lock()
_next_slot = 0.0


def _wait_for_slot():
    """Block until this call may start; safe to use from several threads."""
    global _next_slot
    with _rate_lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + 60.0 / MAX_RPM
    time.sleep(wait)


def ask(tier, prompt, max_tokens=512):
    """Send `prompt` to the model of `tier` and return a result dictionary.

    Returns: tier, answer, in_tokens, out_tokens, latency_ms, cost
    Retries up to 3 times if a call fails (longer waits after a 429 rate-limit error).
    """
    if tier not in config.MODELS:
        raise ValueError(f"Unknown tier '{tier}'. Use one of {config.TIERS}")

    spec = config.MODELS[tier]

    last_error = None
    for attempt in range(3):
        try:
            _wait_for_slot()  # waiting here is NOT counted in latency
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=spec["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,          # same question -> same answer
                max_tokens=max_tokens,  # cap on the reply length
                # Turn OFF the model's hidden "thinking" so all tiers answer directly
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            latency_ms = (time.perf_counter() - start) * 1000

            in_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
            # cost = tokens / 1000 * price per 1K, for input and output separately
            cost = (in_tokens / 1000) * spec["in_cost"] + (out_tokens / 1000) * spec["out_cost"]

            return {
                "tier": tier,
                "answer": (response.choices[0].message.content or "").strip(),
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "latency_ms": round(latency_ms, 1),
                "cost": cost,
            }
        except RateLimitError as e:  # HTTP 429: the server wants us to slow down a lot
            last_error = e
            if attempt < 2:
                time.sleep(15 * (attempt + 1))  # wait 15s, then 30s
        except Exception as e:  # network error, timeout, bad key...
            last_error = e
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s

    raise RuntimeError(f"[{tier}] failed after 3 tries: {last_error}")


if __name__ == "__main__":
    # Quick test: ask the same question to all three tiers
    question = "What is 17 * 23? Reply with just the number."
    for t in config.TIERS:
        try:
            r = ask(t, question)
            print(f"{t:6} | {r['latency_ms']:8.1f} ms | in={r['in_tokens']:3} out={r['out_tokens']:3} "
                  f"| cost=${r['cost']:.6f} | answer: {r['answer']!r}")
        except Exception as e:
            print(f"{t:6} | ERROR: {e}")
