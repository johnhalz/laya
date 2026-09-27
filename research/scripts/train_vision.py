"""Train a Laya vision checkpoint: stage 1 (align) and stage 2 (joint) of the vision plan, with
stage 3 (temperature refit) at the end of every run.

  align  trains only the image projector (`img_proj`) on image rows; the encoder, head and
         SigLIP2 tower are frozen, gradients flow through the frozen encoder to the projector.
  joint  also trains the text encoder (low LR) and the head; the tower stays frozen. Each batch
         mixes in text-only rows (`--replay`, default 40%) so the text skills don't regress.

Every batch samples the tower's patch budget from `--patches`, so the checkpoint works across
budgets instead of only the one it was trained at. The loss is the notebook's: an RLCD
policy-gradient term on a strictly proper reward plus soft cross-entropy
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

    # stage 1 from the English checkpoint + SigLIP2 NaFlex
    python research/scripts/train_vision.py --data data/vision/train --stage align --sources coco \\
        --out ckpt/align
    # stage 2 from the stage-1 checkpoint
    python research/scripts/train_vision.py --data data/vision/train --stage joint --init ckpt/align \\
        --out ckpt/joint

Data comes from build_vision_data.py. A held-out slice (`--calib`) never reaches a training batch;
per-type temperatures are fitted on it and accuracy/ECE per source are printed.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import (  # noqa: E402
    QTYPES, build_model, build_sequence, clamp_temperature, collate_items, ece_score, image_token_counts,
    load_image_processor, open_images, proper_reward,
)


def load_rows(dirs, sources=None):
    rows = []
    for d in dirs:
        for line in open(Path(d) / "data.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if not sources or r["source"] in sources:
                r["root"] = str(d)
                rows.append(r)
    return rows


def load_start(args):
    """(cfg, tokenizer, model, processor) from --init (a vision checkpoint) or --base (text-only)."""
    src = args.init or args.base
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*",
                                                     "encoder/*", "vision/*"])
    src = Path(src)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(src / "tokenizer")
    weights = load_file(str(src / "model.safetensors"))
    if cfg.get("vision"):
        model = build_model(cfg, encoder_dir=str(src / "encoder"), vision_dir=str(src / "vision"))
        model.load_state_dict(weights, strict=True)
        processor = load_image_processor(str(src / "vision"))
    else:
        # A text checkpoint: add the tower and a fresh projector, keep every text weight.
        unused = [t for t in tok.get_vocab() if re.fullmatch(r"[\[<]unused0[\]>]", t)]
        if not unused:
            raise SystemExit("no unused vocab token to use as the image placeholder in %s" % src)
        cfg["vision"] = {"encoder": args.vision, "max_num_patches": 1024,
                         "max_image_tokens": args.max_image_tokens,
                         "image_token_id": tok.convert_tokens_to_ids(unused[0])}
        model = build_model(cfg, encoder_dir=str(src / "encoder"))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        stray = [k for k in missing if not k.startswith(("vision.", "img_proj."))]
        if stray or unexpected:
            raise SystemExit("base checkpoint does not match: missing %s, unexpected %s" % (stray[:5], unexpected[:5]))
        processor = load_image_processor(args.vision)
    cfg["max_len"] = args.max_len
    return cfg, tok, model, processor


def encode(row, tok, processor, cfg, patches):
    """One training item: the sequence Agent.predict would build, plus its target."""
    q = row["question"]
    Agent._check_question("q", q)
    q = Agent._to_internal(q)
    vis, vision, counts = cfg["vision"], None, None
    if row["images"]:
        out = processor(open_images([Path(row["root"]) / p for p in row["images"]]),
                        max_num_patches=patches, return_tensors="pt")
        vision = {k: out[k] for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
        counts = image_token_counts(out["spatial_shapes"], vis["max_image_tokens"])
    ids, markers = build_sequence(tok, row["state"], q, cfg["max_len"], cfg.get("head_max_len", 192),
                                  image_tokens=counts, image_token_id=vis["image_token_id"])
    if len(markers) != len(row["target"]):
        raise ValueError("options exceed head_max_len")
    item = {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": row["target"], "source": row["source"]}
    if vision:
        item["vision"] = vision
    return item


def batch_of(rows, tok, processor, cfg, patches):
    items = []
    for r in rows:
        try:
            items.append(encode(r, tok, processor, cfg, patches))
        except ValueError as e:                               # too many images/options for max_len
            print("  skip %s row: %s" % (r["source"], e), flush=True)
    return collate_items([[it] for it in items], tok.pad_token_id) if items else None


def forward(model, b, device):
    kw = {}
    if "pixel_values" in b:
        kw = {k: b[k].to(device) for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
        kw["image_rows"] = b["image_rows"]
    return model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                 b["marker_mask"].to(device), b["qtype"].to(device), **kw)


def rlcd_loss(logits, act, b, device, sigma, group=4):
    """The notebook's objective: policy gradient on a proper reward + soft cross-entropy."""
    logits = logits.float()
    mask = b["marker_mask"].to(device)
    target = b["target"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return -(adv * logp).mean() + ce + 0.0 * act.sum()


def fit_one_temp(sel):
    """Per-type temperature on held-out logits (the notebook's fit), clamped like the runtime."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z, T = torch.full((len(sel), kmax), -1e4), torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)], T[i, :len(t)] = torch.tensor(z), torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return clamp_temperature(log_t.exp().item())


@torch.no_grad()
def calibrate(model, rows, tok, processor, cfg, device, amp, patches):
    model.eval()
    preds = []
    for i in range(0, len(rows), 8):
        b = batch_of(rows[i:i + 8], tok, processor, cfg, patches)
        if b is None:
            continue
        with amp():
            logits, _ = forward(model, b, device)
        logits = logits.float().cpu().numpy()
        for r, meta in enumerate(b["meta"]):
            k = int(b["marker_mask"][r].sum())
            preds.append((meta["qtype"], meta["source"], logits[r, :k], b["target"][r, :k].numpy()))
    temps = [fit_one_temp([(z, t) for q, _, z, t in preds if q == qt]) for qt in range(3)]
    report = {}
    for source in sorted({s for _, s, _, _ in preds}):
        conf, correct = [], []
        for q, s, z, t in preds:
            if s == source:
                p = np.exp(z / temps[q] - (z / temps[q]).max())
                p /= p.sum()
                conf.append(p.max())
                correct.append(p.argmax() == t.argmax())
        report[source] = {"n": len(conf), "accuracy": round(float(np.mean(correct)), 4),
                          "ece": round(ece_score(np.array(conf), np.array(correct, dtype=float)), 4)}
    return temps, report


def save(model, cfg, tok, processor, out: Path, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    model.encoder.config.save_pretrained(out / "encoder")
    model.vision.config.save_pretrained(out / "vision")
    processor.save_pretrained(out / "vision")
    tok.save_pretrained(out / "tokenizer")
    (out / "rl_agent_config.json").write_text(json.dumps({**cfg, **(extra or {})}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="directories written by build_vision_data.py")
    ap.add_argument("--sources", nargs="+", help="only these sources (align: coco)")
    ap.add_argument("--stage", choices=("align", "joint"), required=True)
    ap.add_argument("--base", default="convaiinnovations/laya", help="text checkpoint to start from")
    ap.add_argument("--init", help="vision checkpoint from an earlier stage (replaces --base)")
    ap.add_argument("--vision", default="google/siglip2-base-patch16-naflex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-image-tokens", type=int, default=128)
    ap.add_argument("--patches", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--replay", type=float, default=0.4, help="joint: share of text-only rows per batch")
    ap.add_argument("--calib", type=int, default=400)
    ap.add_argument("--limit", type=int, help="use at most this many rows (smoke runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg, tok, model, processor = load_start(args)

    rows = load_rows(args.data, args.sources)
    random.shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    n_calib = min(args.calib, len(rows) // 10)
    calib, train = rows[:n_calib], rows[n_calib:]
    image_rows = [r for r in train if r["images"]]
    text_rows = [r for r in train if not r["images"]] if args.stage == "joint" else []
    if not image_rows:
        raise SystemExit("no image rows to train on")

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("img_proj.") or (
            args.stage == "joint" and not name.startswith("vision."))
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("img_proj.")],
               "lr": args.lr_proj if args.stage == "align" else args.lr_head}]
    if args.stage == "joint":
        groups += [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")],
                    "lr": args.lr_encoder},
                   {"params": [p for n, p in model.named_parameters() if p.requires_grad
                               and not n.startswith(("encoder.", "img_proj."))], "lr": args.lr_head}]
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)

    n_text = round(args.batch * args.replay) if text_rows else 0
    n_img = args.batch - n_text
    steps = (len(image_rows) // n_img) * args.epochs
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps // args.accum), eta_min=1e-6)
    use_amp = device.type == "cuda"
    bf16 = use_amp and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not bf16)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16, enabled=use_amp)

    print("%s: %d image rows + %d text rows (%d held out) | %d steps of %d image + %d text rows | %s"
          % (args.stage, len(image_rows), len(text_rows), n_calib, steps, n_img, n_text, device), flush=True)
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        random.shuffle(image_rows)
        model.train()
        model.vision.eval()                                     # frozen: keep it deterministic
        for i in range(0, len(image_rows) - n_img + 1, n_img):
            chunk = image_rows[i:i + n_img] + (random.sample(text_rows, n_text) if n_text else [])
            b = batch_of(chunk, tok, processor, cfg, random.choice(args.patches))
            if b is None:
                continue
            sigma = 0.4 + (0.1 - 0.4) * step / max(1, steps - 1)
            with amp():
                logits, act = forward(model, b, device)
            loss = rlcd_loss(logits, act, b, device, sigma) / args.accum
            scaler.scale(loss).backward()
            step += 1
            if step % args.accum == 0 or step == steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == steps:
                print("  epoch %d step %d/%d loss %.4f %.1fs" % (epoch + 1, step, steps, loss.item() * args.accum,
                                                                  time.time() - t0), flush=True)
        save(model, cfg, tok, processor, Path(args.out) / "checkpoint_latest", {"stage": args.stage})

    temps, report = calibrate(model, calib, tok, processor, cfg, device, amp, max(args.patches))
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    for source, r in report.items():
        print("  %-12s n=%4d accuracy=%.3f ece=%.3f" % (source, r["n"], r["accuracy"], r["ece"]))
    cfg.pop("temperature_by_options", None)                     # this fit is per type
    save(model, cfg, tok, processor, Path(args.out),
         {"temperature": temps, "stage": args.stage, "fine_tuned": True, "calibration": report})
    print("saved", args.out)


if __name__ == "__main__":
    main()
