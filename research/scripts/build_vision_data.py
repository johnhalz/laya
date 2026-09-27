"""Build the vision training/eval data for `train_vision.py` from public datasets.

Writes JSONL, one question per row:

    {"source": "vqav2", "images": ["images/vqav2/123.jpg"], "state": "...",
     "question": {"type": "noul", "instructions": "..."}, "target": [0.2, 0.8]}

`question` is the public question format `Agent.predict` takes; `target` is a (possibly soft)
distribution in option order: choice -> criteria order, noul -> [false, true]. Image paths are
relative to the JSONL's directory, and images are written as the source's original bytes, never
re-encoded or resized, so the processor sees the real resolution.

Sources (streamed; `--per-source` caps each):
    coco       lmms-lab/COCO-Caption2017  noul  true caption vs a hard negative from another image
    vqav2      lmms-lab/VQAv2             noul  yes/no questions, soft target = annotator agreement
    aokvqa     HuggingFaceM4/A-OKVQA      choice 4-way, world knowledge
    scienceqa  derek-thomas/ScienceQA     choice image + text context (the hint)
    textvqa    lmms-lab/textvqa           choice answer vs OCR tokens seen in the same image
    screenqa   bevaya/RICO-ScreenQA       choice app screenshots (1080x1920+); distractors = answers
                                                 to other questions on the same screen
    nlvr2      lmms-lab/NLVR2             noul  two images + a statement (multi-image)
    typed      LocalLLaMA/typed-decisions all   text-only replay, soft gold probabilities
    screens    (generated here)           noul  1920x1080 billing dashboards with 13px text; the
                                                 answer depends on reading it (the 1080p eval;
                                                 in the default sources only with --split eval)
Opt-in, check the license for your use first: chartqa (HuggingFaceM4/ChartQA, GPL-3.0),
hateful_memes (neuralcatcher/hateful_memes, Facebook's dataset license).

    pip install datasets
    python research/scripts/build_vision_data.py --out data/vision/train --split train
    python research/scripts/build_vision_data.py --out data/vision/eval --split eval --per-source 500
"""
import argparse
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SPLITS = {  # (train split, eval split) per source
    # COCO test and VQAv2 test/testdev have no public labels, so both use their labelled split and
    # hold out every 10th image (`_held_out`) for eval.
    "coco": ("val", "val"), "vqav2": ("validation", "validation"), "aokvqa": ("train", "validation"),
    "scienceqa": ("train", "validation"), "textvqa": ("train", "validation"),
    "screenqa": ("train", "validation"), "nlvr2": ("unbalanced_dev", "balanced_test_public"),
    "typed": ("train", "test"), "chartqa": ("train", "val"), "hateful_memes": ("train", "validation"),
}
# `screens` is eval-only by default: training on its one template would inflate the 1080p score.
DEFAULT_SOURCES = ["coco", "vqav2", "aokvqa", "scienceqa", "textvqa", "screenqa", "nlvr2", "typed"]
_WORD = re.compile(r"[a-z]+")


class Writer:
    def __init__(self, out: Path, split: str):
        self.out, self.split, self.n = out, split, Counter()
        (out / "images").mkdir(parents=True, exist_ok=True)
        self.f = open(out / "data.jsonl", "a", encoding="utf-8")

    def image(self, source: str, key, img) -> str:
        """Save an image (datasets' undecoded {"bytes", "path"} dict or a PIL image) as-is."""
        d = self.out / "images" / source
        d.mkdir(exist_ok=True)
        if isinstance(img, dict):
            data = img["bytes"]
            ext = Path(img.get("path") or "").suffix or _sniff(data)
        else:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            data, ext = buf.getvalue(), ".png"
        path = d / ("%s%s" % (re.sub(r"[^\w.-]", "_", str(key)), ext))
        if not path.exists():
            path.write_bytes(data)
        return str(path.relative_to(self.out))

    def row(self, source, images, state, question, target):
        s = sum(target)
        self.f.write(json.dumps({"source": source, "images": images, "state": state, "question": question,
                                 "target": [t / s for t in target]}, ensure_ascii=False) + "\n")
        self.n[source] += 1


def _sniff(data: bytes) -> str:
    return ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"


def _stream(repo, split, image_cols=(), n=1000, config=None):
    """Stream a split with image columns left undecoded. Scans at most 50n + 1000 rows, so a
    filter that matches nothing stops instead of pulling the whole dataset over the network."""
    import itertools

    from datasets import Image, load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    for c in image_cols:
        ds = ds.cast_column(c, Image(decode=False))
    return itertools.islice(ds, 50 * n + 1000)


def _held_out(key) -> bool:
    return int(key) % 10 == 0


def _choice(rng, correct, pool, k=4):
    """A 4-way choice question's criteria and one-hot target: `correct` plus distractors from `pool`."""
    distract = [p for p in dict.fromkeys(pool) if p and p.lower() != correct.lower()]
    rng.shuffle(distract)
    opts = distract[: k - 1] + [correct]
    rng.shuffle(opts)
    return opts, [1.0 if o == correct else 0.0 for o in opts]


def coco(w, split, n, rng):
    rows = []
    for ex in _stream("lmms-lab/COCO-Caption2017", split, ["image"], n):
        if _held_out(ex["id"]) != (w.split == "eval"):
            continue
        rows.append(ex)
        if len(rows) >= n:
            break
    words = [set(_WORD.findall(ex["answer"][0].lower())) for ex in rows]
    for i, ex in enumerate(rows):
        path = w.image("coco", ex["id"], ex["image"])
        # Hard negative: the caption of another image sharing the most words, from a random pool.
        pool = rng.sample(range(len(rows)), min(32, len(rows)))
        j = max((p for p in pool if p != i), key=lambda p: len(words[p] & words[i]), default=None)
        q = {"type": "noul", "instructions": "Does the caption describe image 1?"}
        w.row("coco", [path], 'Caption: "%s"' % ex["answer"][0].strip(), q, [0.0, 1.0])
        if j is not None:
            w.row("coco", [path], 'Caption: "%s"' % rows[j]["answer"][0].strip(), q, [1.0, 0.0])


def vqav2(w, split, n, rng):
    for ex in _stream("lmms-lab/VQAv2", split, ["image"], n):
        if ex["answer_type"] != "yes/no" or not ex["answers"] or _held_out(ex["image_id"]) != (w.split == "eval"):
            continue
        yes = sum(a["answer"] == "yes" for a in ex["answers"]) / len(ex["answers"])
        path = w.image("vqav2", ex["image_id"], ex["image"])
        w.row("vqav2", [path], "", {"type": "noul", "instructions": ex["question"]}, [1 - yes + 1e-3, yes + 1e-3])
        if w.n["vqav2"] >= n:
            break


def aokvqa(w, split, n, rng):
    for ex in _stream("HuggingFaceM4/A-OKVQA", split, ["image"], n):
        if ex["correct_choice_idx"] is None:
            continue
        path = w.image("aokvqa", ex["question_id"], ex["image"])
        target = [1.0 if i == ex["correct_choice_idx"] else 0.0 for i in range(len(ex["choices"]))]
        w.row("aokvqa", [path], "", {"type": "choice", "instructions": ex["question"],
                                     "criteria": list(ex["choices"])}, target)
        if w.n["aokvqa"] >= n:
            break


def scienceqa(w, split, n, rng):
    for i, ex in enumerate(_stream("derek-thomas/ScienceQA", split, ["image"], n)):
        if not ex["image"] or len(set(ex["choices"])) != len(ex["choices"]):
            continue
        path = w.image("scienceqa", "%s_%d" % (split, i), ex["image"])
        target = [1.0 if j == ex["answer"] else 0.0 for j in range(len(ex["choices"]))]
        w.row("scienceqa", [path], ex["hint"] or "", {"type": "choice", "instructions": ex["question"],
                                                      "criteria": list(ex["choices"])}, target)
        if w.n["scienceqa"] >= n:
            break


def textvqa(w, split, n, rng):
    for ex in _stream("lmms-lab/textvqa", split, ["image"], n):
        answer = Counter(a.strip() for a in ex["answers"]).most_common(1)[0][0]
        pool = [t for t in ex["ocr_tokens"] if t.lower() not in answer.lower()]
        if len(set(pool)) < 3:
            continue
        opts, target = _choice(rng, answer, pool)
        path = w.image("textvqa", ex["image_id"], ex["image"])
        w.row("textvqa", [path], "", {"type": "choice", "instructions": ex["question"], "criteria": opts}, target)
        if w.n["textvqa"] >= n:
            break


def _screen_answer(ex):
    """The on-screen text that answers a ScreenQA question (its first annotation), or None."""
    if not ex["ground_truth"]:
        return None                                             # unanswerable on this screen
    gt = ex["ground_truth"][0]
    return " ".join(e["text"] for e in gt["ui_elements"] if e.get("text")) or gt["full_answer"]


def screenqa(w, split, n, rng):
    by_screen = {}
    for ex in _stream("bevaya/RICO-ScreenQA", split, ["image"], 3 * n):
        by_screen.setdefault(ex["screen_id"], []).append(ex)
        if sum(map(len, by_screen.values())) >= 3 * n:
            break
    all_answers = [a for exs in by_screen.values() for a in map(_screen_answer, exs) if a]
    for sid, exs in by_screen.items():
        for ex in exs:
            answer = _screen_answer(ex)
            if not answer:
                continue
            # Answers to other questions on the same screen are text visible in it: hard distractors.
            pool = [a for a in map(_screen_answer, exs) if a] + rng.sample(all_answers, min(3, len(all_answers)))
            opts, target = _choice(rng, answer, pool)
            if len(opts) < 2:
                continue
            path = w.image("screenqa", sid, ex["image"])
            w.row("screenqa", [path], "", {"type": "choice", "instructions": ex["question"],
                                           "criteria": opts}, target)
            if w.n["screenqa"] >= n:
                return


def nlvr2(w, split, n, rng):
    for ex in _stream("lmms-lab/NLVR2", split, ["left_image", "right_image"], n):
        left = w.image("nlvr2", ex["identifier"] + "-L", ex["left_image"])
        right = w.image("nlvr2", ex["identifier"] + "-R", ex["right_image"])
        truth = str(ex["answer"]).lower() == "true"
        q = {"type": "noul", "instructions": "Is this statement true of image 1 and image 2?"}
        w.row("nlvr2", [left, right], ex["question"], q, [0.0, 1.0] if truth else [1.0, 0.0])
        if w.n["nlvr2"] >= n:
            break


def typed(w, split, n, rng):
    for ex in _stream("LocalLLaMA/typed-decisions", split, n=n, config="all"):
        questions, gold = json.loads(ex["questions"]), json.loads(ex["gold"])
        for qid, q in questions.items():
            probs = gold[qid]["probabilities"]
            if q["type"] == "choice":
                keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else q["criteria"]
            elif q["type"] == "noul":
                keys = ["false", "true"]
            else:
                keys = [str(i) for i in range(len(q["criteria"]))]
            w.row("typed", [], ex["state"], q, [float(probs[k]) + 1e-4 for k in keys])
        if w.n["typed"] >= n:
            break


def chartqa(w, split, n, rng):
    rows = []
    for i, ex in enumerate(_stream("HuggingFaceM4/ChartQA", split, ["image"], n)):
        rows.append((i, ex))
        if len(rows) >= n:
            break
    pool = [ex["label"][0] for _, ex in rows]
    for i, ex in rows:
        answer = ex["label"][0]
        try:                                                    # numeric: nearby wrong numbers
            v = float(answer)
            near = ["%g" % (v * f) for f in (0.5, 0.8, 1.25, 2.0)] + ["%g" % (v + d) for d in (-1, 1)]
        except ValueError:
            near = []
        opts, target = _choice(rng, answer, near + rng.sample(pool, min(6, len(pool))))
        path = w.image("chartqa", "%s_%d" % (split, i), ex["image"])
        w.row("chartqa", [path], "", {"type": "choice", "instructions": ex["query"], "criteria": opts}, target)


def hateful_memes(w, split, n, rng):
    from huggingface_hub import hf_hub_download
    name = {"train": "train.jsonl", "validation": "dev_seen.jsonl"}[split]
    lines = Path(hf_hub_download("neuralcatcher/hateful_memes", name, repo_type="dataset")).read_text().splitlines()
    for line in lines[:n]:
        ex = json.loads(line)
        local = hf_hub_download("neuralcatcher/hateful_memes", ex["img"], repo_type="dataset")
        path = w.image("hateful_memes", ex["id"], {"bytes": Path(local).read_bytes(), "path": local})
        q = {"type": "noul", "instructions": "Is this meme hateful?"}
        w.row("hateful_memes", [path], 'Text on the meme: "%s"' % ex["text"], q,
              [0.0, 1.0] if ex["label"] else [1.0, 0.0])


def screens(w, split, n, rng):
    """1920x1080 billing dashboards drawn with 13px text. Whether a charge is duplicated is only
    visible by reading the rows, so this checks what the patch budget keeps legible."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=13)
    big = ImageFont.load_default(size=22)
    items = ["Pro plan", "Team seats", "Storage add-on", "Support tier", "API overage", "Backup service"]
    months = ["January", "February", "March", "April", "May", "June"]
    seed_base = 0 if split == "train" else 10 ** 6
    for i in range(n):
        r = random.Random(seed_base + i)
        img = Image.new("RGB", (1920, 1080), (246, 247, 249))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 240, 1080], fill=(32, 36, 44))
        for k, label in enumerate(["Dashboard", "Invoices", "Customers", "Reports", "Settings"]):
            d.text((24, 90 + 36 * k), label, font=font, fill=(200, 204, 210))
        d.text((280, 40), "Invoices - Acme Corp", font=big, fill=(20, 20, 20))
        rows = [(r.choice(months), r.choice(items), "%.2f" % r.uniform(20, 900)) for _ in range(r.randint(18, 30))]
        dup = r.random() < 0.5
        if dup:                                                 # the same line item billed twice
            rows.insert(r.randrange(len(rows)), rows[r.randrange(len(rows))])
        else:                                                   # make sure no accidental duplicate
            rows = list(dict.fromkeys(rows))
        y = 110
        for c, head in zip((280, 520, 900, 1100), ("Month", "Item", "Amount (USD)", "Status")):
            d.text((c, y), head, font=font, fill=(90, 90, 90))
        for month, item, amount in rows:
            y += 30
            for c, text in zip((280, 520, 900, 1100), (month, item, amount, r.choice(["Paid", "Due"]))):
                d.text((c, y), text, font=font, fill=(30, 30, 30))
        path = w.image("screens", "%s_%05d" % (split, i), img)
        q = {"type": "noul", "instructions": "Does the invoice list in image 1 show the same charge twice "
                                             "(same month, item and amount)?"}
        w.row("screens", [path], "Customer ticket: I think I was billed twice for something.", q,
              [0.0, 1.0] if dup else [1.0, 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("train", "eval"), default="train")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--per-source", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.sources = args.sources or DEFAULT_SOURCES + (["screens"] if args.split == "eval" else [])
    w = Writer(Path(args.out), args.split)
    rng = random.Random(args.seed)
    for source in args.sources:
        split = "train" if source == "screens" and args.split == "train" else (
            "eval" if source == "screens" else SPLITS[source][args.split == "eval"])
        print("%s (%s split)..." % (source, split), file=sys.stderr, flush=True)
        globals()[source](w, split, args.per_source, rng)
        w.f.flush()
        print("  %d rows" % w.n[source], file=sys.stderr, flush=True)
    w.f.close()
    print(json.dumps(dict(w.n)))


if __name__ == "__main__":
    main()
