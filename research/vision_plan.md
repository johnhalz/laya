# Laya vision: images as part of the decision context

## Context
Laya is a text-only decision model: a ModernBERT/mmBERT encoder reads
`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`, and a small
transformer head scores each `[MASK]` marker (`build_sequence` and `DecisionModel` in
`laya/common.py`). The goal is a checkpoint that also takes images as context. That means one
question, like "does this screenshot and ticket show a billing bug?", gets answered from the
pixels and the text together, in the same single forward pass, with the same typed, calibrated
outputs.

Decisions already made: build a **native vision encoder** (not a caption bridge). A GPU is
available but **there is no dataset yet**, so the plan includes building one from public sets.
Everything stays local (AGENTS.md): the vision tower and processor ship inside the checkpoint
directory.

**Target inputs: 1080p images now, video later.** A 1920×1080 screenshot has to stay readable:
UI text there is about 12–14px tall. A fixed 224×224 tower can't do that. It squashes 16:9 into
a square (8.6× horizontally, 4.8× vertically), which leaves text 1–3px tall, and an 8×8 pool then
gives each token a 240×135px area of the original. The plan therefore uses a variable-resolution
tower that keeps the aspect ratio. Video is not built in v1, but the v1 format must not block it
(see [Video](#video-after-v1)).

**It has to stay a System 1 model: fast and cheap per decision.** Text-only calls must not change
at all, and an image must cost a bounded, measured multiple of a text call (see
[Cost](#cost-the-per-question-multiplier)).

## Status (2026-09-27)
Implemented on the `feat/vision` branch of the fork: the runtime (agent, router, hooks), the data
builder, the training and eval scripts, and the step-0 latency benchmark with measured numbers
on an M2 Pro (MPS and CPU). **Not done:** building the full dataset, the training runs, and the
post-training numbers. Those need a CUDA GPU and are the next step.

## Approach: LLaVA-style soft tokens inside the state segment

```
image ─► SigLIP2 NaFlex tower (frozen, native aspect, ≤P patches, once per image)
      ─► avg-pool to ≤ max_image_tokens (aspect kept) ─► MLP projector (dv→d)
                                                                    │
[CLS] <type> ins [SEP] [MASK] opt… [SEP] image 1: <img>×n1 image 2: <img>×n2 … state_text [SEP]
                                                   ▲ placeholder ids; embeddings replaced before encoder
```
- The tower is `google/siglip2-base-patch16-naflex`, without its pooling head. Its processor
  resizes each image to the largest grid that keeps the aspect ratio and fits `max_num_patches`
  (P). For 16:9 that gives:

  | P | grid (w×h patches) | effective input |
  |---|---|---|
  | 256 | ~21×12 | 336×192 |
  | 1024 (default) | ~42×24 | 672×384 |
  | 2048 | ~60×34 | 960×544, where UI text starts to be legible |

- **High resolution goes to the tower, and a small token count goes to the encoder.** The tower
  runs once per image, but every question row repeats the image's tokens. So each image's patch
  grid is average-pooled down to at most `max_image_tokens` (default 128), keeping its aspect
  ratio (`image_grid`). The pooling is a matmul, not `adaptive_avg_pool2d`, because MPS only
  supports that op for sizes that divide evenly. `image_patches=` overrides P per call.
  `max_image_tokens` is a checkpoint setting, because the model is trained at it.
- **Each image block is labelled with plain text**, ` image 1:`, ` image 2:`, before its
  placeholders. That marks where one image ends, keeps them in order (which is frame order for
  video), and lets instructions say "image 2". It costs about 3 tokens per image, and no special
  separator token is needed.
- The encoder is called with `inputs_embeds` instead of `input_ids`, and the placeholder
  positions are filled with `masked_scatter`. That is out of place, so gradients reach the
  projector through a frozen embedding during training. The head, marker scoring and act head are
  unchanged.
- Images go **before** the state text, so text truncation never cuts an image. If the images
  alone overflow `max_len`, a `ValueError` names the token counts and suggests lowering
  `image_patches`, passing fewer images or raising `max_len`.
- The placeholder is an existing unused vocab id (`[unused0]` on ModernBERT), stored in the
  config, so there is no vocab resize. It is scrubbed from the question, option and state text,
  so user text cannot forge a placeholder. The forward also checks that each row's placeholder
  count matches its image features.
- Text-only checkpoints have no `vision` key, which leaves `self.vision=None` and the forward
  path bit-identical to today (`tests/test_vision.py` asserts this).
- **Context length.** The shipped `laya` (English) is ModernBERT-large with `max_len=512` and
  `head_max_len=192`, which leaves about 300 tokens for state. The vision checkpoint trains and
  ships at `max_len: 2048`, which fits roughly 12 images at the 128-token cap with short text.
  ModernBERT supports 8,192 tokens. The README's "4,000 tokens in about 1.7 s on an Apple GPU" is
  for `laya-multilingual` (mmBERT-base), not the large English encoder, so it isn't the number to
  plan with here. The benchmark below is.

### Checkpoint format
`rl_agent_config.json` gains
`"vision": {"encoder": "google/siglip2-base-patch16-naflex", "max_num_patches": 1024, "max_image_tokens": 128, "image_token_id": <id>}`
and `"max_len": 2048`. The checkpoint dir gains `vision/` (vision config + image processor
config). `vision.*` and `img_proj.*` weights go into the existing `model.safetensors`, and the
strict load keeps working. Loading uses the SigLIP2 processor class directly rather than
`AutoImageProcessor`, which requires torchvision on transformers 5.

## Cost: the per-question multiplier
Step 0 was measured before any training. Latency doesn't depend on the weights, so
`research/scripts/bench_vision.py` builds the real shapes (ModernBERT-large + SigLIP2-base) with
random weights and times full calls: preprocessing, tower, and one forward over every question
row. The input is one or four 1920×1080 images. Results are in `research/results/bench_vision_m2pro_*.json`.

Apple M2 Pro GPU (MPS, fp16 from 5 rows like the agent), total ms and multiple of the same call on text:

| questions | images | P | token cap | tokens/row | total | × text |
|---|---|---|---|---|---|---|
| 1 | 0 | – | – | 142 | 45 ms | 1.0 |
| 1 | 1 | 256 | 128 | 257 | 109 ms | 2.4 |
| 1 | 1 | 2048 | 128 | 265 | 264 ms | 5.8 |
| 10 | 0 | – | – | 142 | 334 ms | 1.0 |
| 10 | 1 | 256 | 128 | 257 | 630 ms | 1.9 |
| 10 | 1 | 2048 | 64 | 205 | 644 ms | 1.9 |
| 10 | 1 | 2048 | 128 | 265 | 774 ms | 2.3 |
| 10 | 1 | 2048 | 512 | 655 | 1,682 ms | 5.0 |
| 10 | 4 | 2048 | 128 | 634 | 2,082 ms | 6.2 |
| 10 | 4 | 2048 | 512 | 2,194 | 6,796 ms | 20.3 |

CPU (M2 Pro, fp32), 1 question: text 170 ms, one image at P=256 287 ms (1.7×), at P=2048 775 ms
(4.6×). At 10 questions: text 678 ms, one image at P=256 1,369 ms (2.0×). The full CPU sweep is in
`research/results/bench_vision_m2pro_cpu.json`.

What this settles:
- **The token cap matters much more than the tower resolution once there are several
  questions.** At 10 questions and P=2048, going from a 64 to a 512 token cap moves the call from
  1.9× to 5.0×, while P=256 → 2048 at a 128 cap only moves it 1.9× → 2.3×. The original design
  (2×2 pool, about 510 tokens per image at P=2048) is the 5× / 20× row. The cap is the fix, and its
  default is 128 until the accuracy-by-cap numbers say otherwise.
- **The tower dominates with one question on CPU** (about 640 ms at P=2048). A CPU deployment
  should default to P=256–1024 and ask for 2048 only for fine text.
- **Acceptance limit (proposed):** 1 image + 10 questions at the default settings ≤ 2.5× the text
  call on a GPU. That holds on MPS (2.0× at P=1024). Confirm it on a T4 after training.
- **Backbone:** mmBERT-base is about 2× faster per row (README T4 numbers), and the per-row part
  is what grows with questions. Train on the English large checkpoint first. Try mmBERT-base
  (`--encoder jhu-clsp/mmBERT-base` in the benchmark, `--base` the multilingual checkpoint in
  training) if the English accuracy holds up.

## Code changes (done)

**`laya/common.py`**
- `build_sequence(…, image_tokens=None, image_token_id=None)`: inserts the labelled image block
  after the options `[SEP]`, subtracts it from `room`, raises if it doesn't fit, and scrubs the
  placeholder id from text. The return signature is unchanged.
- `image_grid` / `image_token_counts`: the pooled grid per image. The sequence builder and the
  forward both use them, so their counts can't disagree.
- `DecisionModel(…, vision=None, max_image_tokens=0, image_token_id=None)`: `img_proj` is built in
  the existing meta-device / `no_init` block. `encode_images` runs the tower, pools and projects.
  `forward(…, pixel_values, pixel_attention_mask, spatial_shapes, image_rows)` fills the
  placeholders.
- `collate_items`: a state's processor tensors are stacked once, and `image_rows` maps each
  question row to its images.
- `build_model(cfg, …, vision_dir=None)`: builds the tower from the checkpoint's `vision/` without
  initialising it, or from the Hub base when training (and then initialises `img_proj`).
- `open_images` and `load_image_processor`, shared by the agent and the scripts.

**`laya/agent.py`**: `"vision/*"` is added to the download patterns. The tower's prefixes are
checked in `_verify_compatibility`. `images=` / `image_patches=` are added to `system_one` /
`predict` / `predict_batch` (one image list per state). The processor runs once per state in
`_encode_state`. Images on a text checkpoint raise `ValueError`. `fast=True` or `accelerate()` on a
vision checkpoint raises.

**`laya/router.py`**: `"vision"` can be registered with `Router(models={"vision": …})`.
`route(…, images=)` and `predict(…, images=, image_patches=)` route to it when neither `model` nor
`task` names another checkpoint, and the reason says so. If none is registered the call raises
and says how to register one. `Router.predict_batch` with images raises for now.

**`laya/hooks.py`**: `PredictContext.images` / `image_patches`, so hooks can inspect, redact or
budget them. **`tests/test_hooks_api.py`** has the new contract.

**`pyproject.toml`**: `vision = ["pillow>=9.0", "transformers>=4.49.0"]`.

**Docs**: `docs/vision.md` (in `docs/.nav.yml`) and a README section.

## Training
**Data** (`research/scripts/build_vision_data.py`) writes JSONL rows of
`{source, images, state, question, target}`, one question per row. `question` uses the public
format and `target` is a soft distribution in option order. Images are written as the source's
original bytes, never resized. Each source's scan is capped, so a filter that matches nothing
can't stream a whole dataset.

| Source | Becomes | Why |
|---|---|---|
| COCO captions (val, every 10th image held out): true caption vs. the most word-overlapping caption of another image | `noul` | cheap alignment signal, stage 1 |
| VQAv2 yes/no (validation, every 10th image held out; testdev has no labels) | `noul`, soft target from annotator agreement | grounded binary questions |
| A-OKVQA multiple choice | `choice` | 4-way, needs world knowledge |
| ScienceQA (image subset) | `choice` with the hint as text context | exercises **image + text jointly** |
| TextVQA: majority answer vs. OCR tokens from the same image | `choice` | the answer is in small text |
| RICO-ScreenQA (1080×1920 app screenshots): answer vs. answers to other questions on the same screen | `choice` | screenshots, distractors that are also visible on screen |
| NLVR2: two images + a statement | `noul` | real multi-image data, so `image 1`/`image 2` are learned |
| `LocalLLaMA/typed-decisions` (no image) | all types, soft gold | text replay, so the text skills don't regress |
| Generated 1920×1080 billing dashboards with 13px text | `noul` | **eval only**: the answer depends on reading small text at 1080p |
| Opt-in: ChartQA (GPL-3.0), Hateful Memes (Facebook's license) | `choice`, `noul` | check the license for your use first |

**Stages** (`research/scripts/train_vision.py`)
1. Alignment (`--stage align --sources coco`): only `img_proj` trains. Everything else is frozen,
   and gradients reach the projector through the frozen encoder.
2. Joint (`--stage joint --init <stage 1>`): the text encoder (low LR, same split as the
   notebook), the head and `img_proj` train. SigLIP2 stays frozen. Each batch is 40% text-only
   replay, and the patch budget is sampled per batch from {256, 1024, 2048}.
3. Every run holds out a calibration slice that never reaches a training batch. It fits
   per-type temperatures on that slice, reports accuracy/ECE per source, and saves a loadable
   checkpoint.

The loss is the notebook's: an RLCD policy gradient on a strictly proper reward plus soft
cross-entropy. Gradient checkpointing is on, with bf16/fp16 autocast on CUDA.

## Verification
- **`tests/test_vision.py`** (15 checks, offline, a tiny BERT + SigLIP2 checkpoint loaded through
  `laya.load`):
  - Sequence layout: a 1080p image keeps its 16:9 grid, and the placeholder count matches
    `image_token_counts`. Two images of different sizes get labelled spans in order. Long text is
    truncated while the image block survives. Forged placeholders in user text are scrubbed.
  - Model: placeholders receive exactly the projected features, and every other position keeps
    its token embedding. Text rows are bit-identical with and without the tower. A different
    image changes the answer. The pooling matrix matches `adaptive_avg_pool2d`.
  - API: paths, bytes and PIL images give the same answer. `image_patches` changes the grid.
    `predict_batch` can mix image and text states, and a text state's answer is unchanged.
  - Errors: overflow raises, images on a text checkpoint raise, and the router routes to
    `"vision"` or raises when none is registered.
- All existing suites pass, apart from the environment-gated ones (`test_fast`,
  `test_local_e2e`, `test_mcp_local_e2e`, `test_serve`). Both CI gates and
  `zensical build --strict` are clean.
- The scripts were smoke-run end to end on the M2 Pro: the builder on every source, then both
  training stages on a tiny model. That confirmed align changes only `img_proj`, joint leaves the
  tower untouched, and the result loads with `laya.load`. The eval also ran.
- **After training, report** (`research/scripts/eval_vision.py`):
  - held-out accuracy and ECE per question type on each vision source
  - **accuracy by patch budget (256/1024/2048)** on the generated 1080p screens, which shows
    whether the extra resolution is used; repeat for token caps 64/128/256 to pick the default
  - **coverage at 98% precision** per source, the auto-labelling number: the share of rows
    that could be accepted without review
  - **before/after on text**: typed-decisions test rows on the vision checkpoint vs. `--baseline`
    (the text checkpoint it started from)
  - the benchmark above rerun on a T4, checked against the acceptance limit

## Video (after v1)
This is not built in v1. It stays open because v1 already provides:
- Ordered frames in `images=` with `image k:` labels, so a clip is just a list of frames and
  needs no new public kwarg.
- A per-frame budget through `image_patches` and the token cap. For example, 16 frames at a
  64-token cap is about 1,070 tokens, which fits `max_len=2048`.
- Long clips can use the same approach as `predict_long`: overlapping windows of frames,
  combined with the existing per-type rules (`noul` takes the max over windows, which fits "did X
  happen anywhere in the clip"; `choice` and `score` take the most confident window). Windows
  can't see each other, so comparing frames across windows isn't possible.

Add when video is built: frame sampling and decoding (without a hard ffmpeg dependency in
core), caching tower features across overlapping windows, and a video eval set.

## Out of scope for v1 (add when a trained checkpoint exists)
ONNX export, the TileLang fast path, laya-ts, image upload in `serve.py` / MCP, `predict_long`
and `Router.predict_batch` with images, and video (see above). Also inputs that need more than
2048 patches, such as full-page 300dpi documents or multi-page PDFs, which need tiling on top of
NaFlex. A learned resampler (K query tokens cross-attending over all patches) replaces average
pooling if the accuracy-by-cap numbers show pooling losing small text.
