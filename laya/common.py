"""Core model architecture, token sequence construction, and confidence estimation for laya."""
import io
import json
import math
import os
import threading
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}

# A fast tokenizer is not read-only: `truncation=True` / `padding=True` make it call
# `enable_truncation` / `enable_padding`, which mutates the shared Rust object. One tokenizer is
# parsed per checkpoint directory and shared by every Agent that wants it, so concurrent
# `predict()` calls -- on one Agent or on two sharing the cache -- raced and raised
# `RuntimeError: Already borrowed`. Serialise encoding instead: it is a small fraction of a call
# next to the forward pass, and this keeps the cache's single parse.
_TOKENIZE_LOCK = threading.RLock()


def encode_text(tok, text, **kwargs):
    """Tokenize `text` while holding the lock a shared fast tokenizer needs."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    image_tokens: Optional[List[int]] = None,
    image_token_id: Optional[int] = None,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `image_tokens` (one placeholder count per image) puts an image block before the state:
    ` image 1:` followed by that many `image_token_id` placeholders, then ` image 2:` and so on.
    The block sits before the state so truncation only ever cuts text; a block that does not fit
    raises. `image_token_id` is also scrubbed from the text, so user text cannot forge a placeholder.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = encode_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    if image_token_id is not None:
        head_ids = [i for i in head_ids if i != image_token_id]
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = encode_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        if image_token_id is not None:
            opt_tokens = [i for i in opt_tokens if i != image_token_id]
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    if image_tokens:
        block = []
        for k, n in enumerate(image_tokens):
            block += encode_text(tok, " image %d:" % (k + 1), add_special_tokens=False)["input_ids"]
            block += [image_token_id] * n
        if len(block) > max_len - len(ids) - 1:
            raise ValueError(
                "%d images need %d tokens but only %d of max_len=%d are left after the question; "
                "lower image_patches, pass fewer images or raise max_len"
                % (len(image_tokens), len(block), max(0, max_len - len(ids) - 1), max_len))
        ids += block
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(tok, serialize_state(state).replace(mask_tok, " "),
                                add_special_tokens=False)["input_ids"]
    if image_token_id is not None:
        state_ids = [i for i in state_ids if i != image_token_id]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def open_images(images: List) -> List:
    """RGB PIL images from PIL images, file paths or encoded bytes."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("images need Pillow: pip install 'laya[vision]'") from e
    out = []
    for i, im in enumerate(images):
        if isinstance(im, (bytes, bytearray)):
            im = Image.open(io.BytesIO(im))
        elif isinstance(im, (str, os.PathLike)):
            im = Image.open(im)
        elif not isinstance(im, Image.Image):
            raise TypeError("image %d: expected a PIL image, a file path or bytes, got %s" % (i, type(im).__name__))
        out.append(im.convert("RGB"))
    return out


def load_image_processor(path: str):
    """The SigLIP2 NaFlex image processor saved at `path` (a checkpoint's `vision/` or a Hub id).

    Not AutoImageProcessor: on transformers 5 that requires torchvision, and the PIL processor
    does the same job without it.
    """
    import importlib.util

    import transformers
    cls = transformers.Siglip2ImageProcessor
    if importlib.util.find_spec("torchvision") is None:
        cls = getattr(transformers, "Siglip2ImageProcessorPil", cls)
    return cls.from_pretrained(path)


def image_grid(h: int, w: int, max_tokens: int):
    """Token grid for one image's `h` x `w` patch grid: the grid itself when it fits `max_tokens`,
    else average-pooled down to at most `max_tokens` cells with the aspect ratio kept.

    The tower sees the full patch budget once per image, but every question row repeats the
    image's tokens, so the cap keeps the per-question cost fixed while the resolution grows.
    """
    if h * w <= max_tokens:
        return h, w
    s = math.sqrt(max_tokens / (h * w))
    th = max(1, min(h, int(h * s)))
    return th, max(1, min(w, int(w * s), max_tokens // th))


def image_token_counts(spatial_shapes, max_tokens: int) -> List[int]:
    """Placeholder count per image, from the processor's `spatial_shapes` ([n, 2] of h, w)."""
    return [a * b for a, b in (image_grid(int(h), int(w), max_tokens) for h, w in spatial_shapes.tolist())]


def _pool_matrix(n_out: int, n_in: int, like: torch.Tensor) -> torch.Tensor:
    """[n_out, n_in] averaging matrix with adaptive_avg_pool's windows.

    A matmul instead of F.adaptive_avg_pool2d because MPS only implements that op when the input
    size divides evenly by the output size, and a NaFlex grid rarely does.
    """
    m = torch.zeros(n_out, n_in, dtype=like.dtype, device=like.device)
    for i in range(n_out):
        a, b = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
        m[i, a:b] = 1.0 / (b - a)
    return m


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False, vision: Optional[nn.Module] = None, max_image_tokens: int = 0,
                 image_token_id: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        # Optional vision tower (a SigLIP2 NaFlex vision model). Without one nothing below is
        # registered, so a text-only checkpoint's state dict and forward are unchanged.
        self.vision = vision
        self.max_image_tokens = max_image_tokens
        self.image_token_id = image_token_id
        self.img_proj = None
        d = encoder.config.hidden_size
        # `no_init` means the caller is about to load every parameter from a checkpoint, so the
        # head's initial values are pure overhead -- and not just time. transformers'
        # `no_init_weights()` patches the torch.nn.init functions, but something inside
        # nn.TransformerEncoderLayer draws from the RNG outside them, so building it still
        # advanced the global generator and made `load()` a visible side effect. On the meta
        # device no initialisation kernel runs at all; the layers are materialised empty and the
        # load fills them.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
            if vision is not None:
                dv = vision.config.hidden_size
                self.img_proj = nn.Sequential(nn.Linear(dv, d), nn.GELU(), nn.Linear(d, d))
        if no_init:
            # Only the modules created above are on the meta device; the encoder is already real
            # and may hold non-persistent buffers (RoPE frequencies) that to_empty would wipe.
            for module in (self.head, self.type_emb, self.scorer, self.act_head, self.img_proj):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")
        self.head_checkpointing = False

    def encode_images(self, pixel_values, pixel_attention_mask, spatial_shapes) -> List[torch.Tensor]:
        """One [tokens, d] tensor per image: tower patches pooled to `image_grid`, then projected."""
        x = self.vision(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                        spatial_shapes=spatial_shapes).last_hidden_state
        out = []
        for feats, (h, w) in zip(x, spatial_shapes.tolist()):
            th, tw = image_grid(h, w, self.max_image_tokens)
            grid = feats[: h * w].reshape(h, w, -1)     # NaFlex patches are row-major, padding last
            if (th, tw) != (h, w):
                grid = torch.einsum("ih,hwd,jw->ijd", _pool_matrix(th, h, grid), grid, _pool_matrix(tw, w, grid))
            out.append(self.img_proj(grid.reshape(th * tw, -1)))
        return out

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False,
                pixel_values=None, pixel_attention_mask=None, spatial_shapes=None, image_rows=None):
        if pixel_values is None:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            # Image placeholders get the projected image features; every other position keeps
            # its token embedding, so the rest of the forward is the text path unchanged.
            # `image_rows[r]` lists the images (indices into pixel_values) of row r, in order.
            feats = self.encode_images(pixel_values, pixel_attention_mask, spatial_shapes)
            emb = self.encoder.get_input_embeddings()(input_ids)
            slots = input_ids == self.image_token_id
            per_row = [torch.cat([feats[j] for j in imgs]) for imgs in image_rows if imgs]
            for r, n in enumerate(slots.sum(-1).tolist()):
                want = sum(feats[j].size(0) for j in image_rows[r])
                if n != want:
                    raise ValueError("row %d has %d image placeholders but its images give %d features"
                                     % (r, n, want))
            # Out of place, so autograd reaches the projector through a frozen embedding too.
            # masked_scatter fills the slots in row-major order: row by row, images in order.
            if per_row:
                emb = emb.masked_scatter(slots.unsqueeze(-1), torch.cat(per_row).to(emb.dtype))
            h = self.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _apply_rope_config(ecfg) -> None:
    """Carry transformers>=5 per-layer RoPE settings over to the attributes 4.x reads.

    A checkpoint re-saved by transformers 5 stores RoPE as
    `rope_parameters = {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}}`.
    transformers 4.x does not know that key, so it keeps its own defaults (global 160000,
    local 10000) and any checkpoint whose sliding-attention theta differs silently runs the
    wrong RoPE base -- mmBERT is exactly that case, both of its thetas are 160000. Map the
    values onto `global_rope_theta` / `local_rope_theta`, which 4.x does read. On
    transformers 5 this is a no-op beyond re-setting the same numbers.
    """
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _no_init_weights():
    """`no_init_weights` lives in different modules across transformers versions."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def _build_vision(vis: Dict, vision_dir: Optional[str], pretrained: bool):
    """The SigLIP2 NaFlex vision tower, without its pooling head (the decision head replaces it).

    Returns (tower, fresh): `fresh` is True when the weights came from the Hub base model, which
    means this is a training run and the projector next to it is new too.
    """
    from transformers import AutoConfig, AutoModel

    fresh = pretrained and not (vision_dir and os.path.exists(vision_dir))
    vcfg = AutoConfig.from_pretrained(vis["encoder"] if fresh else (vision_dir or vis["encoder"]))
    vcfg = getattr(vcfg, "vision_config", vcfg)     # a full SigLIP2 config nests the vision part
    vcfg.vision_use_head = False
    if fresh:
        from transformers import Siglip2VisionModel
        return Siglip2VisionModel.from_pretrained(vis["encoder"], config=vcfg, attn_implementation="sdpa"), True
    with _no_init_weights():
        return AutoModel.from_config(vcfg, attn_implementation="sdpa"), False


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained: bool = True,
                revision: Optional[str] = None, vision_dir: Optional[str] = None) -> DecisionModel:
    """Build the decision model described by `cfg`.

    With `pretrained=False`, or when `encoder_dir` holds a saved encoder config, nothing is
    downloaded and **no parameter is initialised**: the caller is expected to load a checkpoint
    into the result with `load_state_dict(..., strict=True)` immediately. Skipping initialisation
    keeps `load()` from spending time on, or consuming RNG for, weights it is about to overwrite.

    When `cfg` has a `"vision"` block the model also gets a vision tower: from `vision_dir` (the
    checkpoint's `vision/`) without initialising it, or, with `pretrained=True` and no
    `vision_dir`, from the Hub base model named in `cfg["vision"]["encoder"]` -- the training
    case, where the image projector is initialised because no checkpoint has one yet.
    """
    from transformers import AutoConfig, AutoModel

    head_layers, n_act = cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    vis = cfg.get("vision")
    vkw, fresh = {}, False
    if vis:
        tower, fresh = _build_vision(vis, vision_dir, pretrained)
        vkw = {"vision": tower, "max_image_tokens": int(vis["max_image_tokens"]),
               "image_token_id": int(vis["image_token_id"])}
    if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        _apply_rope_config(ecfg)
        with _no_init_weights():
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        model = DecisionModel(enc, head_layers, n_act, no_init=True, **vkw)
        if fresh:
            for m in model.img_proj:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        return model
    # Training-time Hub load of the base encoder; allow pinning it like the checkpoints.
    kw = {"attn_implementation": "sdpa"}
    if revision:
        kw["revision"] = revision
    enc = AutoModel.from_pretrained(cfg["encoder"], **kw)
    return DecisionModel(enc, head_layers, n_act, **vkw)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. It is therefore the one confidence with the property the README's
    gating section relies on: of the answers returned at confidence c, about c of them are right.

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.
    """
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published as
# 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
# needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            if len(it["target"]) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(it["target"]), k))
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target", "vision")} for it in items],
    }
    if target is not None:
        res["target"] = target
    # Images belong to a state, so every row of a group shares its group's `vision` tensors
    # (the image processor output). Each set is stacked once and rows point at their images.
    shared, image_rows, n = [], [], 0
    for group in batch:
        vision = next((it["vision"] for it in group if it.get("vision")), None)
        idx = list(range(n, n + len(vision["spatial_shapes"]))) if vision else []
        if vision:
            shared.append(vision)
            n += len(idx)
        image_rows += [idx] * len(group)
    if shared:
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
            res[key] = torch.cat([v[key] for v in shared])
        res["image_rows"] = image_rows
    return res
