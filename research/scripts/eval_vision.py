"""Evaluate a vision checkpoint the way it is used: `laya.load(...).predict(..., images=...)`.

Per source and per tower patch budget: accuracy, ECE, median latency per question, and
coverage at a fixed precision -- the share of rows an auto-labelling pipeline could accept
without review at that precision, taking the most confident answers first. Text-only rows
(`typed`) are also run on `--baseline` (the text checkpoint the vision one started from), which
is the before/after on text the plan asks for.

    python research/scripts/eval_vision.py --ckpt ckpt/joint --data data/vision/eval \\
        --patches 256 1024 2048 --out research/results/eval_vision.json
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

import laya  # noqa: E402
from laya.common import ece_score  # noqa: E402


def probs(answer, question):
    """The answer's distribution in the row's option order (the target's order)."""
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    if answer["type"] == "choice":
        crit = question["criteria"]
        return [answer["probabilities"][str(k)] for k in (crit if isinstance(crit, list) else list(crit))]
    return [answer["probabilities"][str(i)] for i in range(len(answer["probabilities"]))]


def coverage_at(conf, correct, precision):
    """Largest share of rows, most confident first, whose accuracy is still >= precision."""
    order = np.argsort(-np.asarray(conf))
    hits = np.cumsum(np.asarray(correct, dtype=float)[order])
    ok = np.nonzero(hits / np.arange(1, len(order) + 1) >= precision)[0]
    return float((ok[-1] + 1) / len(order)) if len(ok) else 0.0


def evaluate(agent, rows, patches, precision):
    by_source = defaultdict(lambda: {"conf": [], "correct": [], "ms": []})
    for r in rows:
        kw = {"images": [str(Path(r["root"]) / p) for p in r["images"]], "image_patches": patches} if r["images"] else {}
        t = time.perf_counter()
        answer = agent.predict(r["state"], {"q": r["question"]}, **kw)["answers"]["q"]
        ms = (time.perf_counter() - t) * 1000
        p = probs(answer, r["question"])
        s = by_source[r["source"]]
        s["conf"].append(max(p))
        s["correct"].append(int(np.argmax(p) == np.argmax(r["target"])))
        s["ms"].append(ms)
    return {src: {"n": len(s["conf"]), "accuracy": round(float(np.mean(s["correct"])), 4),
                  "ece": round(ece_score(np.array(s["conf"]), np.array(s["correct"], dtype=float)), 4),
                  "coverage_at_%g" % precision: round(coverage_at(s["conf"], s["correct"], precision), 4),
                  "median_ms": round(statistics.median(s["ms"]), 1)}
            for src, s in sorted(by_source.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--patches", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--baseline", default="convaiinnovations/laya", help="text checkpoint for the text before/after")
    ap.add_argument("--precision", type=float, default=0.98)
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for d in args.data:
        rows += [dict(json.loads(line), root=d) for line in open(Path(d) / "data.jsonl", encoding="utf-8")]
    rows = rows[: args.limit] if args.limit else rows
    image_rows = [r for r in rows if r["images"]]
    text_rows = [r for r in rows if not r["images"]]

    agent = laya.load(args.ckpt, device=args.device)
    report = {"ckpt": args.ckpt, "by_patches": {}, "text": {}}
    for P in args.patches:
        report["by_patches"][P] = evaluate(agent, image_rows, P, args.precision)
        print("P=%d" % P)
        for src, m in report["by_patches"][P].items():
            print("  %-12s %s" % (src, m))
    if text_rows:
        report["text"]["vision"] = evaluate(agent, text_rows, None, args.precision)
        if args.baseline:
            report["text"]["baseline"] = evaluate(laya.load(args.baseline, device=args.device), text_rows, None,
                                                  args.precision)
        for name, res in report["text"].items():
            for src, m in res.items():
                print("text %-8s %-8s %s" % (name, src, m))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
