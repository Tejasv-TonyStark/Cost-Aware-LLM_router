"""FastAPI service: route a question to the cheapest suitable tier, then answer it.

Run:  uvicorn app.main:app --port 8000      (from the llm-router folder)

Security:
- POST /ask requires the header  X-API-Key: <APP_API_KEY from .env>  (constant-time compare)
- Per-client rate limit (RATE_LIMIT_PER_MIN requests per minute)
- Daily budget cap on LLM calls and simulated cost (DAILY_CALL_LIMIT, DAILY_COST_LIMIT)
- Audit log of every request in logs/audit.jsonl (question hash only, never the text or any key)
- Question length limit, generic error messages (details go to the server log only)
- Security headers on every response; no CORS (only the Streamlit server calls this API)
"""
import datetime as dt
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from llm_client import ask
from routers import ml_router, policy, rule_router

log = logging.getLogger("router-api")
logging.basicConfig(level=logging.INFO)

# ---- security settings (from .env) ----
API_KEY = os.getenv("APP_API_KEY", "")
if len(API_KEY) < 16:
    raise RuntimeError("Set APP_API_KEY in .env to a random string of at least 16 characters")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
MAX_QUESTION_CHARS = 2000
DAILY_CALL_LIMIT = int(os.getenv("DAILY_CALL_LIMIT", "500"))        # LLM calls per UTC day
DAILY_COST_LIMIT = float(os.getenv("DAILY_COST_LIMIT", "1.00"))     # simulated $ per UTC day
AUDIT_LOG = Path(os.getenv("AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl")))

app = FastAPI(title="Cost-Aware LLM Router")

# ---- load everything once at startup ----
RULES = rule_router.load_rules()
ML_MODEL = ml_router.load_model()
POLICY = policy.load_policy() if policy.POLICY_PATH.exists() else None
try:
    from routers import laya_router
    LAYA = laya_router.load()
    LAYA_ERROR = None
except Exception as e:  # Laya is optional: the API still works with rule / ml
    LAYA, LAYA_ERROR = None, type(e).__name__
    log.warning("Laya not loaded: %s", e)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header)):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, "missing or invalid API key")


_hits = defaultdict(deque)  # client ip -> timestamps of recent requests


def rate_limit(request: Request):
    now = time.monotonic()
    q = _hits[request.client.host if request.client else "unknown"]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "rate limit exceeded, try again in a minute", headers={"Retry-After": "60"})
    q.append(now)


class Budget:
    """Daily spending guard. Counters are rebuilt from today's audit log, so a restart does not reset them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.day, self.calls, self.cost = self._today(), 0, 0.0
        if AUDIT_LOG.exists():
            for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("ts", "").startswith(self.day) and rec.get("status") == 200:
                    self.calls += 1
                    self.cost += rec.get("cost", 0.0)

    @staticmethod
    def _today():
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    def _roll(self):
        if self._today() != self.day:  # new day: start counting from zero
            self.day, self.calls, self.cost = self._today(), 0, 0.0

    def check(self):
        with self.lock:
            self._roll()
            if self.calls >= DAILY_CALL_LIMIT or self.cost >= DAILY_COST_LIMIT:
                raise HTTPException(429, "daily budget reached, try again tomorrow (UTC)")

    def add(self, cost):
        with self.lock:
            self._roll()
            self.calls += 1
            self.cost += cost

    def usage(self):
        with self.lock:
            self._roll()
            return {"day": self.day, "calls": self.calls, "call_limit": DAILY_CALL_LIMIT,
                    "cost": round(self.cost, 6), "cost_limit": DAILY_COST_LIMIT}


BUDGET = Budget()
_audit_lock = threading.Lock()


def audit(request: Request, **fields):
    """Append one JSON line per request. Never stores the question text or any key."""
    rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "client": request.client.host if request.client else "unknown", **fields}
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _audit_lock, open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:  # logging must never break the API
        log.error("audit log write failed: %s", e)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    router: Literal["rule", "ml", "laya"] = "rule"


@app.get("/")
def root():
    return {"service": "Cost-Aware LLM Router API", "health": "/health", "docs": "/docs",
            "ask": "POST /ask with header X-API-Key", "ui": "Streamlit app on port 8501"}


@app.get("/health")
def health():
    return {"status": "ok", "tiers": config.TIERS,
            "routers": {"rule": True, "ml": True, "laya": LAYA is not None},
            "laya_variant": "finetuned" if LAYA is not None and (laya_router.CHECKPOINT_DIR / "rl_agent_config.json").exists()
            else ("zero_shot" if LAYA is not None else None),
            "confidence_policy": POLICY}


@app.get("/usage", dependencies=[Depends(require_api_key)])
def usage():
    return BUDGET.usage()


@app.post("/ask", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ask_endpoint(req: AskRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")
    q_meta = {"router": req.router, "q_sha256": hashlib.sha256(question.encode()).hexdigest()[:16],
              "q_chars": len(question)}
    try:
        BUDGET.check()
    except HTTPException:
        audit(request, **q_meta, status=429, reason="daily budget")
        raise

    probs = None
    t0 = time.perf_counter()
    if req.router == "rule":
        tier = rule_router.route(question, RULES)
    elif req.router == "ml":
        tier, probs = ml_router.route(question, ML_MODEL)
    else:
        if LAYA is None:
            raise HTTPException(503, "Laya router is not available on this server")
        tier, probs = laya_router.route(question, LAYA)
        if POLICY:  # bump one tier up when Laya is unsure
            tier = policy.apply(tier, max(probs.values()), POLICY["threshold"])
    router_ms = (time.perf_counter() - t0) * 1000

    try:
        r = ask(tier, question)
    except Exception as e:
        log.error("LLM call to %s tier failed: %s", tier, e)  # details stay in the server log
        audit(request, **q_meta, tier=tier, router_ms=round(router_ms, 3), status=502)
        raise HTTPException(502, f"the {tier} model is unavailable right now, please retry")

    BUDGET.add(r["cost"])
    audit(request, **q_meta, tier=tier, in_tokens=r["in_tokens"], out_tokens=r["out_tokens"],
          cost=r["cost"], router_ms=round(router_ms, 3), model_ms=r["latency_ms"], status=200)
    return {"answer": r["answer"], "chosen_tier": tier, "router_used": req.router,
            "router_ms": round(router_ms, 3), "model_latency_ms": r["latency_ms"],
            "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
            "cost": r["cost"], "probabilities": probs}
