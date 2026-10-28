"""Run every question through one tier and save results (resumable).

Usage: python scripts/run_models.py --tier small [--limit N] [--workers 4]
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_client import ask

parser = argparse.ArgumentParser()
parser.add_argument("--tier", required=True, choices=["small", "medium", "large"])
parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
parser.add_argument("--workers", type=int, default=4, help="parallel requests (keep low for rate limits)")
args = parser.parse_args()

questions = [json.loads(l) for l in open(ROOT / "data" / "questions.jsonl", encoding="utf-8")]
if args.limit:
    questions = questions[: args.limit]

out_path = ROOT / "results" / f"raw_{args.tier}.jsonl"
done = set()
if out_path.exists():  # resume: skip ids already saved
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
todo = [q for q in questions if q["id"] not in done]
print(f"{args.tier}: {len(done)} already done, {len(todo)} to run")

if todo:
    ask(args.tier, "Say OK.", max_tokens=5)  # warm-up call (not saved)

lock = threading.Lock()
failures = 0
with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
    futures = {pool.submit(ask, args.tier, q["prompt"]): q for q in todo}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        q = futures[fut]
        try:
            r = fut.result()
        except Exception as e:  # keep going; failed ids are retried on the next run
            failures += 1
            tqdm.write(f"FAILED {q['id']}: {e}")
            continue
        with lock:
            f.write(json.dumps({"id": q["id"], **r}, ensure_ascii=False) + "\n")
            f.flush()

rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
print(f"Saved {len(rows)} rows to {out_path} ({failures} failed this run)")
print(f"Total cost: ${sum(r['cost'] for r in rows):.4f}")
print(f"Avg latency: {sum(r['latency_ms'] for r in rows) / len(rows):.0f} ms")
