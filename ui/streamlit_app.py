"""Streamlit frontend: ask questions through the FastAPI router, and explore the results.

Run:  streamlit run ui/streamlit_app.py      (the API must be running)

The API key stays on the Streamlit server (read from .env / environment);
the browser never sees it.
"""
import hmac
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("APP_API_KEY", "")
UI_PASSWORD = os.getenv("UI_PASSWORD", "")  # empty = no login (fine on your own PC)
RES, DATA = ROOT / "results", ROOT / "data"

# ---- design tokens (one place for every color): dark "tech" theme with violet accents ----
INK, INK_2, MUTED = "#ece9ff", "#b9b4d8", "#8a85a8"
GRID, AXIS, SURFACE = "#221f40", "#3b3670", "rgba(255,255,255,0.04)"
VIOLET, VIOLET_2, CYAN = "#8b5cf6", "#c4b5fd", "#22d3ee"
BASELINE = "#6b6785"
# Router families use the palette's dark-mode slots (validated for dark surfaces)
FAMILY_COLORS = {"rule": "#3987e5", "ml": "#d95926", "laya": "#199e70"}
TIER_COLORS = {"small": "#3987e5", "medium": "#199e70", "large": "#d95926"}
SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ROUTERS = {"rule": "Rule-based", "ml": "ML classifier", "laya": "Laya"}
NICE = {"always_small": "Always small", "always_large": "Always large", "random": "Random",
        "rule_based": "Rule-based", "ml_classifier": "ML classifier", "laya_zero_shot": "Laya zero-shot",
        "laya_finetuned": "Laya fine-tuned", "laya_finetuned+confidence": "Laya fine-tuned + confidence",
        "laya_zero_shot+confidence": "Laya zero-shot + confidence", "oracle": "Oracle (perfect)"}

st.set_page_config(page_title="Cost-Aware LLM Router", page_icon="🔀", layout="wide")

st.markdown(f"""
<style>
  /* page: deep navy with soft violet / blue glows */
  .stApp {{ background:
      radial-gradient(1100px 520px at 8% -8%, rgba(139,92,246,.28), transparent 60%),
      radial-gradient(900px 480px at 100% 0%, rgba(37,99,235,.20), transparent 60%),
      radial-gradient(800px 600px at 50% 110%, rgba(168,85,247,.12), transparent 60%), #0b0a1a; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.0rem; max-width: 1240px; }}
  h1, h2, h3, h4, h5 {{ letter-spacing: -0.01em; color: {INK}; }}

  /* hero: violet -> indigo -> blue gradient with a faint tech grid */
  .hero {{ position: relative; overflow: hidden; padding: 1.7rem 1.9rem; border-radius: 20px; margin-bottom: 1.3rem;
           background:
             linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px) 0 0 / 28px 28px,
             linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px) 0 0 / 28px 28px,
             linear-gradient(125deg, #2e1065 0%, #5b21b6 38%, #7c3aed 62%, #2563eb 100%);
           border: 1px solid rgba(196,181,253,.25); box-shadow: 0 20px 60px -20px rgba(124,58,237,.55); }}
  .hero .tag {{ display: inline-block; font-size: .75rem; letter-spacing: .14em; text-transform: uppercase;
                color: #e9d5ff; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
                padding: .2rem .65rem; border-radius: 999px; margin-bottom: .7rem; }}
  .hero h1 {{ font-size: 2.35rem; margin: 0 0 .35rem 0; padding: 0; font-weight: 750;
              background: linear-gradient(90deg, #ffffff 0%, #e9d5ff 55%, #a5f3fc 100%);
              -webkit-background-clip: text; background-clip: text; color: transparent; }}
  .hero p {{ color: #ddd6fe; font-size: 1.03rem; margin: 0; max-width: 860px; }}
  .hero-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin-top: 1.2rem; }}
  .hero-stat {{ background: rgba(11,10,26,.35); border: 1px solid rgba(255,255,255,.14); border-radius: 14px;
                padding: .8rem 1rem; backdrop-filter: blur(6px); }}
  .hero-stat .label {{ color: #c4b5fd; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; }}
  .hero-stat .value {{ color: #fff; font-size: 1.65rem; font-weight: 750; line-height: 1.2; margin-top: .15rem; }}
  .hero-stat .sub {{ color: #ddd6fe; font-size: .82rem; margin-top: .1rem; }}

  /* glass cards with gradient numbers */
  .kpi {{ background: {SURFACE}; border: 1px solid rgba(167,139,250,.18); border-radius: 16px;
          padding: 1rem 1.15rem; height: 100%; backdrop-filter: blur(6px);
          box-shadow: inset 0 1px 0 rgba(255,255,255,.05); }}
  .kpi .label {{ color: {MUTED}; font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; }}
  .kpi .value {{ font-size: 1.8rem; font-weight: 750; line-height: 1.25; margin-top: .2rem;
                 background: linear-gradient(90deg, {VIOLET_2}, {CYAN}); -webkit-background-clip: text;
                 background-clip: text; color: transparent; }}
  .kpi .sub {{ color: {INK_2}; font-size: .85rem; margin-top: .15rem; }}
  .badge {{ display: inline-block; padding: .22rem .75rem; border-radius: 999px; color: #fff;
            font-weight: 700; font-size: .85rem; letter-spacing: .06em; box-shadow: 0 0 18px -4px currentColor; }}
  .muted {{ color: {MUTED}; font-size: .88rem; }}
  .step {{ background: {SURFACE}; border: 1px solid rgba(167,139,250,.18); border-radius: 16px; padding: 1.1rem 1.2rem; }}
  .step .num {{ display: inline-flex; width: 1.9rem; height: 1.9rem; border-radius: 10px; align-items: center;
                justify-content: center; font-weight: 750; color: #fff; margin-bottom: .55rem;
                background: linear-gradient(135deg, #7c3aed, #2563eb); }}
  .step b {{ color: {INK}; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .6rem; margin: .6rem 0 .4rem; }}
  .stat {{ background: rgba(139,92,246,.08); border: 1px solid rgba(139,92,246,.16); border-radius: 12px;
           padding: .55rem .75rem; }}
  .stat .label {{ color: {MUTED}; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }}
  .stat .value {{ color: {INK}; font-size: 1.1rem; font-weight: 650; margin-top: .1rem; white-space: nowrap;
                  font-variant-numeric: tabular-nums; }}

  /* "did you know" fact cards */
  .facts {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem; }}
  .fact {{ position: relative; background: linear-gradient(160deg, rgba(139,92,246,.14), rgba(37,99,235,.06));
           border: 1px solid rgba(167,139,250,.22); border-radius: 16px; padding: 1rem 1.1rem 1.05rem; }}
  .fact .icon {{ font-size: 1.35rem; }}
  .fact .big {{ font-size: 1.45rem; font-weight: 750; margin: .25rem 0 .15rem;
                background: linear-gradient(90deg, #f5d0fe, {VIOLET_2}, {CYAN}); -webkit-background-clip: text;
                background-clip: text; color: transparent; }}
  .fact .text {{ color: {INK_2}; font-size: .9rem; line-height: 1.45; }}
  @media (max-width: 900px) {{ .facts, .hero-stats {{ grid-template-columns: 1fr; }} }}

  /* tabs */
  [data-testid="stTab"] p {{ font-size: 1.02rem; font-weight: 650; }}
  [data-testid="stTab"] {{ padding-top: .45rem; padding-bottom: .45rem; }}
</style>
""", unsafe_allow_html=True)


# ---- optional login (enabled when UI_PASSWORD is set) ----
@st.cache_resource
def login_guard():
    """Shared across all browser sessions, so reloading the page does not reset the lockout."""
    return {"failures": [], "lock": threading.Lock()}


def require_login():
    if not UI_PASSWORD or st.session_state.get("authed"):
        return
    guard = login_guard()
    now = time.time()
    with guard["lock"]:
        guard["failures"] = [t for t in guard["failures"] if now - t < 300]  # 5-minute window
        locked = len(guard["failures"]) >= 5
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="hero" style="text-align:center"><div class="tag">restricted</div>'
                    '<h1>Cost-Aware LLM Router</h1><p>Enter the access password to continue.</p></div>',
                    unsafe_allow_html=True)
        if locked:
            st.error("Too many wrong attempts. Login is locked for a few minutes.")
            st.stop()
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                if hmac.compare_digest(pw.encode(), UI_PASSWORD.encode()):
                    st.session_state["authed"] = True
                    st.rerun()
                with guard["lock"]:
                    guard["failures"].append(now)
                st.error("Wrong password.")
    st.stop()


# ---- helpers ----
def family(strategy):
    for fam in FAMILY_COLORS:
        if strategy.startswith(fam):
            return fam
    return "baseline"


def color_of(strategy):
    return FAMILY_COLORS.get(family(strategy), BASELINE)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def badge(tier):
    return f'<span class="badge" style="background:{TIER_COLORS.get(tier, MUTED)}">{tier.upper()}</span>'


def style(fig, height=380, legend=True, top=8):
    """One chart style for the whole app: quiet grid, system font, no plot background.

    automargin makes Plotly reserve room for tick labels (it is off by default).
    """
    fig.update_layout(
        height=height, font=dict(family=SANS, size=12, color=INK_2), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=16, t=top + (28 if legend else 0), b=8),
        barcornerradius=4, bargap=0.3,
        hoverlabel=dict(bgcolor="#1c1a33", font=dict(family=SANS, color=INK), bordercolor="#4c3fa6"),
        showlegend=legend, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="",
                                       traceorder="normal", font=dict(color=INK_2)))
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True,
                tickfont=dict(color=INK_2), title_font=dict(color=INK_2))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def stats(items):
    """A compact row of small stat cards (smaller than st.metric so long values fit)."""
    cells = "".join(f'<div class="stat"><div class="label">{k}</div><div class="value">{v}</div></div>'
                    for k, v in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def clean_answer(text):
    """Make model answers display safely: Streamlit treats $...$ as LaTeX, which garbles money amounts.

    Common LaTeX bits the models emit are turned into plain symbols, then every $ is escaped.
    """
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", text)
    for latex, plain in {r"\times": "×", r"\div": "÷", r"\cdot": "·", r"\approx": "≈", r"\le": "≤", r"\ge": "≥",
                         r"\%": "%", r"\$": "$", r"\,": " ", r"\(": "", r"\)": "", r"\[": "", r"\]": ""}.items():
        text = text.replace(latex, plain)
    return text.replace("$", r"\$")


def show(fig):
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})


@st.cache_data(ttl=60)
def load_results():
    comp = pd.read_csv(RES / "comparison.csv")
    labels = pd.read_csv(DATA / "labels.csv")
    preds = {}
    for name, f in [("rule_based", "rule_preds.csv"), ("ml_classifier", "ml_preds.csv")]:
        if (RES / f).exists():
            preds[name] = pd.read_csv(RES / f)
    if (RES / "laya_preds.csv").exists():
        lp = pd.read_csv(RES / "laya_preds.csv")
        for v in lp.variant.unique():
            preds[v] = lp[(lp.variant == v) & (lp.split == "test")]
    return comp, labels, preds


def api_usage():
    try:
        r = requests.get(f"{API_URL}/usage", headers={"X-API-Key": API_KEY}, timeout=5)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=10)
def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def call_api(question, router):
    """Returns (result dict, error message)."""
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question, "router": router},
                          headers={"X-API-Key": API_KEY}, timeout=180)
    except requests.RequestException:
        return None, f"Could not reach the API at {API_URL}."
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text[:200]
        return None, f"HTTP {r.status_code}: {detail}"
    return r.json(), None


@st.cache_data(ttl=60)
def avg_out_tokens():
    """Average reply length per tier from the raw model runs (None if the files are missing)."""
    out = {}
    for t in ["small", "medium", "large"]:
        f = RES / f"raw_{t}.jsonl"
        if f.exists():
            out[t] = pd.read_json(f, lines=True).out_tokens.mean()
    return out


def interesting_facts():
    """Facts computed from the real result files: (icon, big number, explanation)."""
    facts = []
    src = labels.groupby("source")[[f"correct_{t}" for t in tiers]].mean()
    if "triviaqa" in src.index and len(tiers) >= 2:
        s, l = 100 * src.loc["triviaqa", f"correct_{tiers[0]}"], 100 * src.loc["triviaqa", f"correct_{tiers[-1]}"]
        facts.append(("🧠", f"{s:.0f}% vs {l:.0f}%", f"Trivia is the great divider: the small model answers {s:.0f}% of "
                      f"trivia questions, the large one {l:.0f}%. Facts need parameters; reasoning needs less."))
    if "gsm8k" in src.index:
        s = 100 * src.loc["gsm8k", f"correct_{tiers[0]}"]
        facts.append(("🧮", f"{s:.0f}%", f"of grade-school math word problems are solved by the ~4B small model, so "
                      "most math does not need the 120B model at all."))
    lat_s, lat_l = by.loc[f"always_{tiers[0]}"].avg_latency_ms, by.loc[f"always_{tiers[-1]}"].avg_latency_ms
    if lat_l < lat_s:
        facts.append(("⚡", f"{lat_l:,.0f} ms vs {lat_s:,.0f} ms", "Speed paradox: on NVIDIA's hosted GPUs the large "
                      "model answered faster than the small one. Routing here saves money, not time."))
    o, l = by.loc["oracle"], by.loc[f"always_{tiers[-1]}"]
    if o.accuracy > l.accuracy:
        facts.append(("🎯", f"{100 * o.accuracy:.1f}% at {o.cost_pct_of_large:.0f}% cost", "A perfect router would be "
                      "cheaper AND more accurate than always using the large model, because on some questions only "
                      "the small model got it right."))
    if {"laya_zero_shot", "laya_finetuned"} <= set(by.index):
        z, f = 100 * by.loc["laya_zero_shot"].accuracy, 100 * by.loc["laya_finetuned"].accuracy
        facts.append(("🔧", f"{z:.1f}% → {f:.1f}%", "Fine-tuning Laya on our own training labels lifted its final "
                      f"answer accuracy by {f - z:.1f} points at about the same cost."))
    if {"ml_classifier", "laya_finetuned"} <= set(by.index) and by.loc["ml_classifier"].router_ms > 0:
        ratio = by.loc["laya_finetuned"].router_ms / by.loc["ml_classifier"].router_ms
        facts.append(("🏎️", f"{ratio:,.0f}× faster", "The TF-IDF classifier decides about "
                      f"{ratio:,.0f}× faster than fine-tuned Laya on CPU, with a statistically tied accuracy."))
    toks = avg_out_tokens()
    if tiers[0] in toks and tiers[-1] in toks and toks[tiers[0]] > toks[tiers[-1]]:
        facts.append(("💬", f"{toks[tiers[0]]:.0f} vs {toks[tiers[-1]]:.0f} tokens", "The small model is chattier: "
                      "its average reply is longer than the large model's, which eats into its cost advantage."))
    unsolved = int((~labels[[f"correct_{t}" for t in tiers]].any(axis=1)).sum())
    facts.append(("🧩", f"{unsolved} of {len(labels)}", "questions were answered wrongly by every model tier. "
                  "No router can fix those; they are the ceiling of this setup."))
    return facts


def show_facts(facts):
    cards = "".join(f'<div class="fact"><div class="icon">{i}</div><div class="big">{b}</div>'
                    f'<div class="text">{t}</div></div>' for i, b, t in facts)
    st.markdown(f'<div class="facts">{cards}</div>', unsafe_allow_html=True)


require_login()  # stops here until the password is entered (only when UI_PASSWORD is set)

# ---- data for the header ----
have_results = (RES / "comparison.csv").exists() and (DATA / "labels.csv").exists()
if have_results:
    comp, labels, preds = load_results()
    test = labels[labels.split == "test"]
    tiers = [c.removeprefix("pct_") for c in comp.columns if c.startswith("pct_")]
    by = comp.set_index("strategy")
    routers_only = comp[comp.strategy.map(family) != "baseline"]
    best = routers_only.loc[routers_only.accuracy.idxmax()]
    large_row = by.loc[f"always_{tiers[-1]}"]
    saved_per_m = (large_row.total_cost - best.total_cost) / len(test) * 1_000_000
    hero_stats = [
        ("Saved per 1M questions", f"${saved_per_m:,.0f}", f"{100 - best.cost_pct_of_large:.0f}% cheaper than always large"),
        ("Accuracy kept", f"{100 * best.accuracy:.1f}%", f"vs {100 * large_row.accuracy:.1f}% always using the large model"),
        ("Best router", NICE.get(best.strategy, best.strategy), f"decides in {best.router_ms:.1f} ms per question"),
    ]
else:
    hero_stats = [("Results", "—", "run the evaluation scripts first")]

cells = "".join(f'<div class="hero-stat"><div class="label">{a}</div><div class="value">{b}</div>'
                f'<div class="sub">{c}</div></div>' for a, b, c in hero_stats)
st.markdown(f"""
<div class="hero">
  <div class="tag">LLM cost optimization · 3 routers compared</div>
  <h1>Cost-Aware LLM Router</h1>
  <p>Every question goes to the cheapest model that can still answer it. Three ways of deciding go head to head:
     hand-written rules, a TF-IDF classifier, and the Laya decision model.</p>
  <div class="hero-stats">{cells}</div>
</div>""", unsafe_allow_html=True)

tab_ask, tab_dash, tab_how = st.tabs(["💬  Ask", "📊  Dashboard", "🧭  How it works"])


# =====================================================================
# Ask tab
# =====================================================================
with tab_ask:
    health = api_health()
    if health:
        laya = f"Laya {health['laya_variant'].replace('_', '-')}" if health["routers"]["laya"] else "Laya unavailable"
        st.markdown(f'<span class="muted">🟢 API online · tiers: {" / ".join(health["tiers"])} · {laya}</span>',
                    unsafe_allow_html=True)
    else:
        st.error(f"API not reachable at {API_URL}. Start it with:  uvicorn app.main:app --port 8000")

    left, right = st.columns([5, 7], gap="large")
    usage = api_usage() if health else None

    with left:
        examples = {
            "Easy fact": "What is the capital of France?",
            "Rare fact": "At which sport could you win the Victor Barna Award?",
            "Math": "Pens cost 3 for $4. Maria buys 27 pens and pays with a $50 bill. How much change does she get?",
            "Specialist": "Let R be a ring and U, V two-sided ideals of R. Is U + V always an ideal of R?",
        }
        pick = st.pills("Try an example", list(examples), selection_mode="single", key="example")
        if pick and st.session_state.get("last_pick") != pick:
            st.session_state["question"] = examples[pick]
            st.session_state["last_pick"] = pick
        question = st.text_area("Your question", key="question", height=130, max_chars=2000,
                                placeholder="Ask anything: a fact, a math word problem, a multiple-choice question...")
        router = st.segmented_control("Router", list(ROUTERS), format_func=ROUTERS.get, default="ml", key="router")
        compare = st.toggle("Compare all three routers on this question", key="compare")
        go_clicked = st.button("Route & answer", type="primary", width="stretch",
                               disabled=health is None or not (question or "").strip())
        if usage:
            frac = max(usage["calls"] / usage["call_limit"], usage["cost"] / usage["cost_limit"])
            st.progress(min(frac, 1.0), text=f"Today's budget: {usage['calls']}/{usage['call_limit']} calls · "
                                             f"\\${usage['cost']:.4f} / \\${usage['cost_limit']:.2f} simulated")

    with right:
        if go_clicked:
            chosen = list(ROUTERS) if compare else [router or "ml"]
            results = []
            with st.spinner("Routing and answering..."):
                for r_name in chosen:
                    out, err = call_api(question, r_name)
                    results.append((r_name, out, err))
                    if out:
                        st.session_state.setdefault("history", []).insert(0, {
                            "question": question[:70] + ("…" if len(question) > 70 else ""),
                            "router": ROUTERS[r_name], "tier": out["chosen_tier"],
                            "router ms": round(out["router_ms"], 1), "model ms": round(out["model_latency_ms"]),
                            "cost ($)": out["cost"]})
            st.session_state["last_results"] = results

        results = st.session_state.get("last_results")
        if not results:
            with st.container(border=True):
                st.markdown("#### How to use")
                st.markdown("1. Pick an example or type a question.\n"
                            "2. Choose a router, or turn on **compare** to see all three.\n"
                            "3. The router picks **small** or **large** before any model runs; "
                            "only that one model answers.")
        else:
            for r_name, out, err in results:
                with st.container(border=True):
                    if err:
                        st.markdown(f"**{ROUTERS[r_name]}**")
                        st.error(err)
                        continue
                    st.markdown(f"**{ROUTERS[r_name]}** chose &nbsp;{badge(out['chosen_tier'])}", unsafe_allow_html=True)
                    stats([("Router time", f"{out['router_ms']:.1f} ms"),
                           ("Model latency", f"{out['model_latency_ms']:,.0f} ms"),
                           ("Tokens in → out", f"{out['in_tokens']} → {out['out_tokens']}"),
                           ("Cost (simulated)", f"${out['cost']:.6f}")])
                    if out["probabilities"]:
                        p = out["probabilities"]
                        fig = go.Figure([go.Bar(x=[100 * p[t]], y=["confidence"], orientation="h", name=t,
                                                marker_color=TIER_COLORS.get(t, MUTED),
                                                text=f"{t} {100 * p[t]:.0f}%", textposition="inside",
                                                insidetextanchor="middle", textfont=dict(color="white"),
                                                hovertemplate=f"{t}: %{{x:.1f}}%<extra></extra>") for t in p])
                        style(fig, height=46, legend=False)
                        fig.update_layout(barmode="stack", bargap=0, margin=dict(l=0, r=0, t=4, b=4))
                        fig.update_xaxes(range=[0, 100], visible=False)
                        fig.update_yaxes(visible=False)
                        show(fig)
                    with st.expander("Answer", expanded=len(results) == 1):
                        st.markdown(clean_answer(out["answer"]))

    if st.session_state.get("history"):
        st.markdown("##### This session")
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, hide_index=True, width="stretch",
                     column_config={"cost ($)": st.column_config.NumberColumn(format="%.6f")})


# =====================================================================
# Dashboard tab
# =====================================================================
with tab_dash:
    if not have_results:
        st.warning("No results yet. Run  python scripts/evaluate.py  first.")
    else:
        large = by.loc["always_large"]
        small = by.loc["always_small"]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Best router", NICE.get(best.strategy, best.strategy), f"{len(test)} test questions")
        with k2:
            kpi("Accuracy", f"{100 * best.accuracy:.1f}%",
                f"{100 * (best.accuracy - large.accuracy):+.1f} pts vs always large")
        with k3:
            kpi("Cost", f"{best.cost_pct_of_large:.0f}%", "of always using the large model")
        with k4:
            gap = (best.accuracy - small.accuracy) / max(large.accuracy - small.accuracy, 1e-9)
            kpi("Gap recovered", f"{100 * gap:.0f}%", "of the small→large accuracy gap")
        st.write("")
        st.markdown("##### ✨ Did you know?")
        show_facts(interesting_facts()[:6])
        st.write("")

        # Merge rows that are identical to another (e.g. +confidence that bumped nothing) so labels don't overlap
        plot = comp.copy()
        plot["name"] = plot.strategy.map(lambda s: NICE.get(s, s))
        key_cols = ["accuracy", "cost_pct_of_large", "pct_" + tiers[-1]]
        dup = plot.duplicated(subset=key_cols, keep="first")
        for i in plot[dup].index:
            same = (plot[key_cols] == plot.loc[i, key_cols]).all(axis=1) & ~dup
            plot.at[plot[same].index[0], "name"] += "  (= + confidence)"
        plot = plot[~dup]

        t_over, t_save, t_break, t_type, t_conf, t_data = st.tabs(
            ["Overview", "Savings", "Cost, routing & latency", "By question type", "Confusion matrices", "Data"])

        with t_save:
            st.markdown("##### How much does routing save? &nbsp;<span class='muted'>· projected from the test set</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            volume = c1.number_input("Questions per month", min_value=1_000, max_value=1_000_000_000,
                                     value=1_000_000, step=100_000, format="%d")
            scale = c2.slider("Price scale (1× = our simulated prices)", 0.1, 50.0, 1.0, 0.1,
                              help="Our prices are simulated (large ≈ 20× small). Scale them to match real API prices; "
                                   "the percentages stay the same, only the dollar amounts change.")
            n_test = len(test)
            per_q_large = large.total_cost / n_test * scale
            sv = comp[comp.strategy != "always_large"].copy()
            sv["monthly_cost"] = sv.total_cost / n_test * scale * volume
            sv["saved"] = per_q_large * volume - sv.monthly_cost
            sv["saved_pct"] = 100 * sv.saved / (per_q_large * volume)
            sv["extra_wrong"] = (large.accuracy - sv.accuracy) * volume
            sv = sv[sv.strategy.map(family) != "baseline"].sort_values("saved")

            best_row = sv.set_index("strategy").loc[best.strategy]
            k1, k2, k3 = st.columns(3)
            with k1:
                kpi("Always large", f"${per_q_large * volume:,.0f}", "per month, no routing")
            with k2:
                kpi(f"With {NICE.get(best.strategy, best.strategy)}", f"${best_row.monthly_cost:,.0f}",
                    f"saves ${best_row.saved:,.0f} ({best_row.saved_pct:.0f}%) per month")
            with k3:
                kpi("Accuracy given up", f"{100 * (large.accuracy - best.accuracy):.1f} pts",
                    f"≈ {best_row.extra_wrong:,.0f} more wrong answers per month")
            st.write("")
            names = [NICE.get(s, s) for s in sv.strategy]
            fig = go.Figure(go.Bar(
                x=sv.saved, y=names, orientation="h", marker_color=[color_of(s) for s in sv.strategy],
                text=[f"${v:,.0f}  ({p:.0f}%)  ·  {100 * a:.1f}% accuracy"
                      for v, p, a in zip(sv.saved, sv.saved_pct, sv.accuracy)],
                textposition="outside", cliponaxis=False,
                customdata=sv[["monthly_cost", "extra_wrong"]].values,
                hovertemplate="<b>%{y}</b><br>saved $%{x:,.0f} per month<br>costs $%{customdata[0]:,.0f}"
                              "<br>%{customdata[1]:,.0f} more wrong answers than always large<extra></extra>"))
            fig.update_xaxes(title="Saved per month vs always large ($)", tickprefix="$",
                             range=[0, sv.saved.max() * 1.55])
            show(style(fig, height=360, legend=False))
            st.caption("Savings and accuracy trade off: the rule router saves the most only because it sends almost "
                       "everything to the small model. The oracle shows a perfect router could save ~77% while being "
                       "slightly *more* accurate than always large.")

        with t_over:
            st.markdown("##### Cost vs accuracy &nbsp;<span class='muted'>· up and to the left is better · "
                        "gray = reference baselines</span>", unsafe_allow_html=True)
            # Labels go to the right of each point, except where two points sit close together
            label_pos = {"always_small": "bottom center", "rule_based": "middle right"}
            fig = go.Figure()
            for fam, grp in plot.groupby(plot.strategy.map(family)):
                fig.add_trace(go.Scatter(
                    x=grp.cost_pct_of_large, y=100 * grp.accuracy, mode="markers+text", name=fam,
                    text=grp.name, textposition=[label_pos.get(s, "middle right") for s in grp.strategy],
                    textfont=dict(size=12, color=INK_2),
                    marker=dict(size=14 if fam != "baseline" else 11, color=FAMILY_COLORS.get(fam, BASELINE),
                                line=dict(width=2, color="#0b0a1a")),
                    customdata=grp[["routing_accuracy", "router_ms"]].values,
                    hovertemplate="<b>%{text}</b><br>accuracy %{y:.1f}%<br>cost %{x:.1f}% of always large"
                                  "<br>routing accuracy %{customdata[0]:.1%}<br>router %{customdata[1]:.2f} ms<extra></extra>"))
            fig.update_xaxes(title="Cost (% of always large)", range=[-2, 125], ticksuffix="%")
            fig.update_yaxes(title="Final answer accuracy", ticksuffix="%",
                             range=[100 * plot.accuracy.min() - 3, 100 * plot.accuracy.max() + 2])
            show(style(fig, height=500, legend=False))
            st.caption("Learned routers (ML, fine-tuned Laya) sit close to the oracle's cost but well above the rules. "
                       "Costs use simulated prices; only their ratio matters.")

        with t_break:
            order = comp.sort_values("accuracy").strategy.tolist()
            d = comp.set_index("strategy").loc[order]
            names = [NICE.get(s, s) for s in order]
            colors = [color_of(s) for s in order]
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### Final answer accuracy")
                fig = go.Figure(go.Bar(x=100 * d.accuracy, y=names, orientation="h", marker_color=colors,
                                       text=[f"{100 * v:.1f}%" for v in d.accuracy], textposition="outside",
                                       hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
                fig.update_xaxes(range=[0, 108], ticksuffix="%")
                show(style(fig, legend=False))
            with c2:
                st.markdown("##### Cost (% of always large)")
                fig = go.Figure(go.Bar(x=d.cost_pct_of_large, y=names, orientation="h", marker_color=colors,
                                       text=[f"{v:.1f}%" for v in d.cost_pct_of_large], textposition="outside",
                                       hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
                fig.update_xaxes(range=[0, 118], ticksuffix="%")
                show(style(fig, legend=False))
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### Where questions are sent")
                fig = go.Figure([go.Bar(name=t, y=names, x=d[f"pct_{t}"], orientation="h",
                                        marker_color=TIER_COLORS.get(t, MUTED),
                                        hovertemplate=f"%{{y}}: %{{x:.1f}}% to {t}<extra></extra>") for t in tiers])
                fig.update_layout(barmode="stack")
                fig.update_xaxes(range=[0, 100], ticksuffix="%")
                show(style(fig))
            with c2:
                st.markdown("##### Latency per question &nbsp;<span class='muted'>· model + router overhead</span>",
                            unsafe_allow_html=True)
                fig = go.Figure([
                    go.Bar(name="model", y=names, x=d.avg_latency_ms - d.router_ms, orientation="h", marker_color=BASELINE,
                           hovertemplate="%{y}: model %{x:.0f} ms<extra></extra>"),
                    go.Bar(name="router", y=names, x=d.router_ms, orientation="h", marker_color="#e34948",
                           hovertemplate="%{y}: router %{x:.1f} ms<extra></extra>")])
                fig.update_layout(barmode="stack")
                fig.update_xaxes(title="ms")
                show(style(fig))
            st.caption("Routing saves cost, not time, here: the hosted large model was faster than the small one, "
                       "and Laya adds CPU time per decision.")

        with t_type:
            rows = []
            for t in tiers:
                for src, g in test.groupby("source"):
                    rows.append({"strategy": f"always_{t}", "source": src, "acc": 100 * g[f"correct_{t}"].mean()})
            for name, p in preds.items():
                m = test.merge(p[["id", "pred_tier"]], on="id")
                ok = [row[f"correct_{tier}"] for (_, row), tier in zip(m.iterrows(), m.pred_tier)]
                m = m.assign(ok=ok)
                for src, g in m.groupby("source"):
                    rows.append({"strategy": name, "source": src, "acc": 100 * g.ok.mean()})
            by_src = pd.DataFrame(rows)
            sources = sorted(by_src.source.unique())
            st.markdown("##### Accuracy by question type")
            fig = make_subplots(rows=1, cols=len(sources), subplot_titles=[s.upper() for s in sources],
                                shared_yaxes=True, horizontal_spacing=0.04)
            strat_order = [s for s in comp.sort_values("accuracy").strategy if s in set(by_src.strategy)]
            for j, src in enumerate(sources, start=1):
                g = by_src[by_src.source == src].set_index("strategy").reindex(strat_order).dropna()
                fig.add_trace(go.Bar(x=g.acc, y=[NICE.get(s, s) for s in g.index], orientation="h",
                                     marker_color=[color_of(s) for s in g.index],
                                     text=[f"{v:.0f}%" for v in g.acc], textposition="outside",
                                     hovertemplate=f"{src}: " + "%{y} %{x:.1f}%<extra></extra>"), row=1, col=j)
                fig.update_xaxes(range=[0, 118], ticksuffix="%", row=1, col=j)
            fig.update_annotations(font=dict(size=12, color=INK))
            show(style(fig, height=440, legend=False, top=36))
            st.caption("TriviaQA separates the routers: the small model knows far fewer facts, and only routers "
                       "that learned from our labels can tell which facts it misses.")

        with t_conf:
            names = [n for n in ["rule_based", "ml_classifier", "laya_zero_shot", "laya_finetuned"] if n in preds]
            st.markdown("##### Confusion matrices &nbsp;<span class='muted'>· rows = true cheapest tier, "
                        "columns = router's choice</span>", unsafe_allow_html=True)
            fig = make_subplots(rows=1, cols=len(names), subplot_titles=[NICE.get(n, n) for n in names],
                                horizontal_spacing=0.06)
            zmax = 0
            mats = []
            for n in names:
                m = test.merge(preds[n][["id", "pred_tier"]], on="id")
                cm = pd.crosstab(m.label, m.pred_tier).reindex(index=tiers, columns=tiers, fill_value=0)
                mats.append(cm)
                zmax = max(zmax, cm.values.max())
            for j, cm in enumerate(mats, start=1):
                fig.add_trace(go.Heatmap(
                    z=cm.values, x=[f"→ {t}" for t in tiers], y=tiers, zmin=0, zmax=zmax, showscale=False,
                    colorscale=[[0, "#17152e"], [0.35, "#3b2f86"], [1, "#8b5cf6"]], xgap=3, ygap=3,
                    text=cm.values, texttemplate="%{text}", textfont=dict(size=15, color="#ffffff"),
                    hovertemplate="true %{y}, chose %{x}: %{z} questions<extra></extra>"), row=1, col=j)
                fig.update_yaxes(autorange="reversed", row=1, col=j)
            fig.update_annotations(font=dict(size=12, color=INK))
            fig.update_xaxes(showgrid=False, side="bottom")
            fig.update_yaxes(showgrid=False)
            show(style(fig, height=320, legend=False, top=36))
            st.caption("Bottom-left cell = hard questions wrongly sent to small (accuracy loss). "
                       "Top-right = easy questions sent to large (wasted cost).")

        with t_data:
            table = comp.assign(strategy=comp.strategy.map(lambda s: NICE.get(s, s)))
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="percent"),
                "routing_accuracy": st.column_config.NumberColumn("routing acc.", format="percent"),
                "total_cost": st.column_config.NumberColumn("total cost ($)", format="%.4f"),
                "cost_pct_of_large": st.column_config.NumberColumn("cost % of large", format="%.1f%%"),
                "avg_latency_ms": st.column_config.NumberColumn("avg latency (ms)", format="%.0f"),
                "router_ms": st.column_config.NumberColumn("router (ms)", format="%.2f"),
                **{f"pct_{t}": st.column_config.NumberColumn(f"% to {t}", format="%.1f%%") for t in tiers}})
            st.download_button("Download comparison.csv", (RES / "comparison.csv").read_bytes(),
                               file_name="comparison.csv", mime="text/csv")


# =====================================================================
# How it works tab
# =====================================================================
with tab_how:
    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("Measure", "All 900 questions (math, multiple choice, trivia) were answered by every model tier and "
                    "graded against known answers."),
        ("Label & learn", "Each question's label is the cheapest tier that answered correctly. Routers learn "
                          "from the 630 training questions only."),
        ("Route", "For a new question the router predicts the tier first, then only that model is called. "
                  "Savings come from easy questions going to the small model."),
    ]
    for n, (col, (title, body)) in enumerate(zip([c1, c2, c3], steps), start=1):
        col.markdown(f'<div class="step"><div class="num">{n}</div><br><b>{title}</b><br>'
                     f'<span class="muted">{body}</span></div>', unsafe_allow_html=True)
    st.write("")
    if have_results:
        st.markdown("##### ✨ Interesting facts from the experiment")
        show_facts(interesting_facts())
        st.write("")
    with st.container(border=True):
        st.markdown("""
**The three routers**
- **Rule-based:** 7 hand-written features (length, numbers, math words, ...) with weights grid-searched on train.
- **ML classifier:** TF-IDF word features + logistic regression, 5-fold CV on train.
- **Laya:** a 0.4B decision model asked *"small_model or large_model?"*. Zero-shot, and fine-tuned on our labels.

**Fair comparison:** same 270 test questions for every router, never used for tuning; results are simulated
from saved answers, so every router sees identical model outputs.

**Security:** the API needs a key (kept on this server, never sent to your browser), rate-limits clients,
validates input, and returns generic errors.
""")
