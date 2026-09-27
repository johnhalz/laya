"""Step 0 of the vision plan: what an image costs per call, measured before any training.

Latency does not depend on the weights, so this builds the real shapes -- the text encoder
(ModernBERT-large by default, the `laya` checkpoint's) plus the SigLIP2 NaFlex tower -- with
random weights, and times one call the way the agent runs it: image processor, one forward over
every question row, image placeholders filled from the tower. Only configs and the tokenizer are
downloaded, never weights.

Two sweeps, because the knobs cost in different places:
  * tower patches P (256/1024/2048) at a fixed token cap: tower time, paid once per image
  * token cap per image (64..512) at P=2048: encoder time, paid again in every question row
Each is run with 1 and 4 images of 1920x1080 and with 1 and 10 questions, next to the same
questions on text alone.

    python research/scripts/bench_vision.py --device mps --out research/results/bench_vision_m2pro_mps.json
    python research/scripts/bench_vision.py --device cpu --reps 2 --questions 1 10
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from laya.agent import MPS_AMP_MIN_ROWS_DEFAULT  # noqa: E402
from laya.common import build_model, build_sequence, collate_items, image_token_counts, load_image_processor  # noqa: E402

STATE = ("Ticket 4411 from the finance team: the customer says the invoice page shows the March charge "
         "twice and the export button returns an error. They attached a screenshot of the billing "
         "dashboard and ask whether this is a billing bug or a display problem. Priority unknown. ") * 2
QUESTION = {"t": "noul", "ins": "Does the screenshot show a duplicated charge on the invoice?", "crit": None}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device, reps):
    fn()                                                        # warm-up: kernels, caches
    sync(device)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    ap.add_argument("--vision", default="google/siglip2-base-patch16-naflex")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--images", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--patches", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--caps", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--out")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    image_token_id = tok.convert_tokens_to_ids("[unused0]")
    cfg = {"encoder": args.encoder, "head_layers": 2, "act_costs": {"act": 0},
           "vision": {"encoder": args.vision, "max_num_patches": 1024, "max_image_tokens": 128,
                      "image_token_id": image_token_id}}
    model = build_model(cfg, pretrained=False)
    with torch.no_grad():                                       # random weights; timing ignores values
        for p in model.parameters():
            p.normal_(0, 0.02)
    model.to(device).eval()
    processor = load_image_processor(args.vision)
    rng = np.random.default_rng(0)
    frame = Image.fromarray(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))

    def run(n_questions, n_images, patches, cap):
        model.max_image_tokens = cap
        vision, counts = None, None
        t0 = time.perf_counter()
        if n_images:
            out = processor([frame] * n_images, max_num_patches=patches, return_tensors="pt")
            vision = {k: out[k] for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
            counts = image_token_counts(out["spatial_shapes"], cap)
        prep_ms = (time.perf_counter() - t0) * 1000
        items = []
        for _ in range(n_questions):
            ids, markers = build_sequence(tok, STATE, QUESTION, max_len=8192, head_max_len=192,
                                          image_tokens=counts, image_token_id=image_token_id)
            items.append({"ids": ids, "markers": markers, "qtype": 2, **({"vision": vision} if vision else {})})
        b = collate_items([items], tok.pad_token_id)
        kw = {}
        if vision:
            kw = {k: b[k].to(device) for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
            kw["image_rows"] = b["image_rows"]
        args_ = [b[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        # The agent's autocast policy: always on CUDA, on MPS only from MPS_AMP_MIN_ROWS rows.
        amp = device.type == "cuda" or (device.type == "mps" and n_questions >= MPS_AMP_MIN_ROWS_DEFAULT)

        def forward():
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                model(*args_, **kw)

        def tower():
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                model.encode_images(kw["pixel_values"], kw["pixel_attention_mask"], kw["spatial_shapes"])

        row = {"questions": n_questions, "amp_fp16": amp, "images": n_images, "patches": patches if n_images else 0,
               "cap": cap if n_images else 0, "tokens_per_row": int(b["input_ids"].shape[1]),
               "image_tokens_per_row": sum(counts) if counts else 0,
               "prep_ms": round(prep_ms, 1),
               "tower_ms": round(timed(tower, device, args.reps), 1) if n_images else 0.0,
               "total_ms": round(timed(forward, device, args.reps) + prep_ms, 1)}
        print("q=%(questions)2d img=%(images)d P=%(patches)4d cap=%(cap)3d tokens/row=%(tokens_per_row)4d "
              "prep=%(prep_ms)6.1f tower=%(tower_ms)7.1f total=%(total_ms)7.1f ms" % row, flush=True)
        return row

    rows = []
    for q in args.questions:
        rows.append(run(q, 0, 0, 0))                            # text-only baseline
        for n in args.images:
            for P in args.patches:
                rows.append(run(q, n, P, 128))
            for cap in args.caps:
                if cap != 128 or 2048 not in args.patches:
                    rows.append(run(q, n, 2048, cap))

    if args.out:
        meta = {"encoder": args.encoder, "vision": args.vision, "device": str(device),
                "platform": platform.platform(), "processor": platform.processor(),
                "torch": torch.__version__, "transformers": transformers.__version__, "reps": args.reps,
                "image": "1920x1080 random RGB", "weights": "random (latency only)"}
        Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
