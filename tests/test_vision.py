"""Vision path: image placeholders, pooling, routing and errors on a tiny offline checkpoint.

Builds a random BERT + SigLIP2 checkpoint on disk (no downloads) and loads it through the real
`laya.load`, so the processor, sequence layout, collation and forward are all exercised.

Run: python tests/test_vision.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import AutoModel, BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig  # noqa: E402

from laya import Router, load  # noqa: E402
from laya.common import (  # noqa: E402
    DecisionModel, _pool_matrix, build_sequence, collate_items, image_grid, image_token_counts,
)

IMG = 5  # the placeholder id ("[unused0]")
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]", "image", "1", "2", "3", "4", ":",
         "hello", "is", "there", "a", "cat", "true", "false", "question", "noul", "choice", "yes", "no"]
QUESTIONS = {"cat": {"type": "noul", "instructions": "is there a cat"},
             "pick": {"type": "choice", "instructions": "is there a cat", "criteria": ["yes", "no"]}}


def _checkpoint(root: Path, vision: bool) -> str:
    torch.manual_seed(0)
    ecfg = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=1, intermediate_size=32)
    ecfg.save_pretrained(root / "encoder")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tok.backend_tokenizer.pre_tokenizer = Whitespace()
    tok.save_pretrained(root / "tokenizer")
    cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
           "max_len": 256, "head_max_len": 96}
    kw = {}
    if vision:
        vcfg = Siglip2VisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                   num_attention_heads=1, patch_size=4, num_patches=16, vision_use_head=False)
        vcfg.save_pretrained(root / "vision")
        Siglip2ImageProcessor(patch_size=4, max_num_patches=64).save_pretrained(root / "vision")
        cfg["vision"] = {"encoder": "unused/offline", "max_num_patches": 64, "max_image_tokens": 8,
                         "image_token_id": IMG}
        kw = {"vision": AutoModel.from_config(vcfg), "max_image_tokens": 8, "image_token_id": IMG}
    model = DecisionModel(BertModel(ecfg), head_layers=1, **kw)
    save_file(model.state_dict(), str(root / "model.safetensors"))
    (root / "rl_agent_config.json").write_text(json.dumps(cfg))
    return str(root)


def _image(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


TMP = tempfile.TemporaryDirectory()
VISION_DIR = _checkpoint(Path(TMP.name) / "vision", vision=True)
TEXT_DIR = _checkpoint(Path(TMP.name) / "text", vision=False)
with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
    AGENT = load(VISION_DIR, device="cpu")
    TEXT_AGENT = load(TEXT_DIR, device="cpu")


def _encoded(images, state="hello", **kw):
    ids = list(QUESTIONS)
    internal = {q: AGENT._to_internal(QUESTIONS[q]) for q in ids}
    return AGENT._encode_state(state, ids, internal, images=images, **kw)


def test_image_grid_caps_tokens_and_keeps_aspect():
    assert image_grid(4, 6, 64) == (4, 6)                      # fits: no pooling
    th, tw = image_grid(34, 60, 128)                           # 16:9 at 2048 patches
    assert th * tw <= 128 and abs(tw / th - 60 / 34) < 0.25, (th, tw)
    th, tw = image_grid(1, 4000, 128)                          # extreme aspect stays under the cap
    assert th * tw <= 128, (th, tw)


def test_pool_matrix_matches_adaptive_avg_pool():
    x = torch.randn(1, 3, 7, 11)
    want = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    got = torch.einsum("ih,chw,jw->cij", _pool_matrix(3, 7, x), x[0], _pool_matrix(4, 11, x))
    assert torch.allclose(got, want[0], atol=1e-6)


def test_1080p_keeps_aspect_and_counts_match():
    items = _encoded([_image(1920, 1080, 0)])
    h, w = items[0]["vision"]["spatial_shapes"][0].tolist()
    assert 1.5 < w / h < 2.0, (h, w)                           # 16:9, not squashed to a square
    want = image_token_counts(items[0]["vision"]["spatial_shapes"], 8)
    for it in items:
        assert it["ids"].count(IMG) == sum(want)


def test_two_images_get_labelled_spans_in_order():
    images = [_image(64, 64, 1), _image(1920, 1080, 2)]
    items = _encoded(images)
    counts = image_token_counts(items[0]["vision"]["spatial_shapes"], 8)
    assert counts[0] != counts[1], counts
    ids = items[0]["ids"]
    tid = AGENT.tok.convert_tokens_to_ids
    label = lambda k: [tid("image"), tid(str(k)), tid(":")]  # noqa: E731
    start1 = next(i for i in range(len(ids)) if ids[i:i + 3] == label(1))
    assert ids[start1 + 3:start1 + 3 + counts[0]] == [IMG] * counts[0]
    start2 = start1 + 3 + counts[0]
    assert ids[start2:start2 + 3] == label(2)
    assert ids[start2 + 3:start2 + 3 + counts[1]] == [IMG] * counts[1]


def test_placeholders_receive_projected_features():
    images = [_image(64, 64, 1), _image(1920, 1080, 2)]
    b = collate_items([_encoded(images)], AGENT.tok.pad_token_id)
    model = AGENT.model
    seen = {}
    hook = model.encoder.register_forward_pre_hook(
        lambda m, a, kw: seen.update(emb=kw["inputs_embeds"].clone()), with_kwargs=True)
    try:
        with torch.no_grad():
            model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                  pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"],
                  spatial_shapes=b["spatial_shapes"], image_rows=b["image_rows"])
            feats = torch.cat(model.encode_images(b["pixel_values"], b["pixel_attention_mask"],
                                                  b["spatial_shapes"]))
            text = model.encoder.get_input_embeddings()(b["input_ids"])
    finally:
        hook.remove()
    slots = b["input_ids"] == IMG
    for r in range(len(b["image_rows"])):
        assert torch.allclose(seen["emb"][r][slots[r]], feats)
        assert torch.equal(seen["emb"][r][~slots[r]], text[r][~slots[r]])


def test_text_rows_identical_with_and_without_tower():
    b = collate_items([_encoded(None)], AGENT.tok.pad_token_id)
    text_only = DecisionModel(BertModel(AGENT.model.encoder.config), head_layers=1).eval()
    text_only.load_state_dict({k: v for k, v in AGENT.model.state_dict().items()
                               if not k.startswith(("vision.", "img_proj."))}, strict=True)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.no_grad():
        a, b2 = AGENT.model(*args), text_only(*args)
    assert torch.equal(a[0], b2[0]) and torch.equal(a[1], b2[1])


def test_different_image_changes_answer():
    p1 = AGENT.predict("hello", QUESTIONS, images=[_image(64, 64, 3)])["answers"]["cat"]["noul"]
    p2 = AGENT.predict("hello", QUESTIONS, images=[Image.new("RGB", (64, 64))])["answers"]["cat"]["noul"]
    assert p1 != p2, (p1, p2)


def test_images_accept_paths_and_bytes():
    img = _image(32, 32, 4)
    path = Path(TMP.name) / "img.png"
    img.save(path)
    a = AGENT.predict("hello", QUESTIONS, images=[img])["answers"]["cat"]["noul"]
    assert AGENT.predict("hello", QUESTIONS, images=[str(path)])["answers"]["cat"]["noul"] == a
    assert AGENT.predict("hello", QUESTIONS, images=[path.read_bytes()])["answers"]["cat"]["noul"] == a


def test_long_text_truncated_image_block_survives():
    items = _encoded([_image(1920, 1080, 5)], state="hello " * 1000)
    n = image_token_counts(items[0]["vision"]["spatial_shapes"], 8)[0]
    for it in items:
        assert len(it["ids"]) == 256 and it["ids"].count(IMG) == n


def test_forged_placeholder_in_text_is_scrubbed():
    items = _encoded([_image(64, 64, 6)], state="[unused0] hello [unused0]")
    n = image_token_counts(items[0]["vision"]["spatial_shapes"], 8)[0]
    assert all(it["ids"].count(IMG) == n for it in items)
    AGENT.predict("[unused0] hello", {"q": {"type": "noul", "instructions": "[unused0] cat"}},
                  images=[_image(64, 64, 6)])


def test_images_overflowing_max_len_raise():
    try:
        AGENT.predict("hello", QUESTIONS, images=[_image(64, 64, i) for i in range(4)], max_len=40)
    except ValueError as e:
        assert "images need" in str(e), e
    else:
        raise AssertionError("expected ValueError")


def test_image_patches_override_changes_grid():
    small = _encoded([_image(1920, 1080, 7)], image_patches=16)[0]["vision"]["spatial_shapes"]
    big = _encoded([_image(1920, 1080, 7)])[0]["vision"]["spatial_shapes"]
    assert small.prod() <= 16 < big.prod()


def test_predict_batch_mixes_image_and_text_states():
    out = AGENT.predict_batch(["hello", "hello", "hello"], QUESTIONS,
                              images=[[_image(64, 64, 8)], None, [_image(1920, 1080, 9), _image(64, 64, 8)]])
    single = AGENT.predict("hello", QUESTIONS)["answers"]["cat"]["noul"]
    assert abs(out[1]["answers"]["cat"]["noul"] - single) < 1e-4
    try:
        AGENT.predict_batch(["hello", "hello"], QUESTIONS, images=[[_image(8, 8, 0)]])
    except ValueError as e:
        assert "per state" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_images_on_text_only_agent_raise():
    try:
        TEXT_AGENT.predict("hello", QUESTIONS, images=[_image(8, 8, 0)])
    except ValueError as e:
        assert "no vision encoder" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_router_routes_images_to_vision():
    try:
        Router().route("hello", images=[_image(8, 8, 0)])
    except ValueError as e:
        assert "vision" in str(e)
    else:
        raise AssertionError("expected ValueError without a registered vision checkpoint")
    router = Router(models={"vision": VISION_DIR}, device="cpu")
    decision = router.route("hello", images=[_image(8, 8, 0)])
    assert decision.model == "vision" and "image" in decision.reason
    assert router.route("hello").model == "english"            # text still routes as before
    router.attach("vision", AGENT)
    out = router.predict("hello", QUESTIONS, images=[_image(64, 64, 3)])
    assert out["routing"]["model"] == "vision" and "cat" in out["answers"]
    try:
        router.predict_batch([{"state": "hello", "questions": QUESTIONS, "images": [_image(8, 8, 0)]}])
    except ValueError as e:
        assert "Router.predict" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "-", type(e).__name__, e)
    print("\n%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
