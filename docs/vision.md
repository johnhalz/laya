# Images

A vision checkpoint answers the same typed questions from images and text together, in the same
single forward pass, with the same calibrated outputs. Text-only checkpoints are unchanged: their
forward pass, outputs and speed are exactly what they were.

!!! note "No published vision checkpoint yet"
    The runtime, the data builder and the training scripts are here; a trained checkpoint is not.
    Build one with the [training scripts](#training-a-vision-checkpoint), then load it by path.

## Installation

```bash
pip install "laya[vision]"   # Pillow, and transformers >= 4.49 for SigLIP2
```

## Usage

```python
import laya

agent = laya.load("path/to/vision-checkpoint")

result = agent.predict(
    "Ticket 4411: the customer says March was charged twice.",
    {"duplicate": {"type": "noul",
                   "instructions": "Does the invoice list in image 1 show the same charge twice?"}},
    images=["invoice.png"],          # PIL images, file paths or encoded bytes
)
result["answers"]["duplicate"]["noul"]
```

Several images, or the frames of a clip, go in one list, in order. The model sees each one labelled
`image 1:`, `image 2:` and so on, so instructions can refer to them by number:

```python
agent.predict(state, {"changed": {"type": "noul",
                                  "instructions": "Is the error dialog in image 2 missing from image 1?"}},
              images=[before_png, after_png])
```

`predict_batch` takes one list of images per state (`None` for a state without any):

```python
agent.predict_batch([state_a, state_b], questions, images=[[img_a], None])
```

A `Router` sends a request with images to the checkpoint registered as `"vision"`:

```python
router = laya.Router(models={"vision": "path/to/vision-checkpoint"})
router.predict(state, questions, images=[img])["routing"]["reason"]
# '1 image(s) given; only the vision checkpoint reads images'
```

Passing `images=` to a checkpoint without a vision tower raises `ValueError`; images are never
silently ignored. Hooks see them as `ctx.images` (one list per state) and can rewrite or redact
them in `on_predict_start`.

## Resolution and cost

Two numbers set what an image costs, and they cost in different places:

* **`image_patches`**: how many 16px patches the vision tower gets per image (checkpoint default in
  `rl_agent_config.json`, override per call). The image is resized to the largest grid that keeps
  its aspect ratio and fits, so a 1920x1080 screenshot is never squashed into a square. 2048
  patches is about 960x544, where 13-14px UI text stays legible. The tower runs **once per image**.
* **`max_image_tokens`** (checkpoint setting): the tower's patches are average-pooled down to at
  most this many tokens per image. These tokens are repeated in **every question row**, so this is
  the number that grows with the number of questions.

Measured with random weights at the real shapes (ModernBERT-large, the `laya` encoder, plus SigLIP2
base) on an Apple M2 Pro GPU, one 1920x1080 image, full call including preprocessing
([`research/scripts/bench_vision.py`](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_vision.py)):

| questions | patches | tokens per image | time | vs. the same questions on text |
|---|---|---|---|---|
| 1 | 256 | 128 | 109 ms | 2.4x |
| 1 | 2048 | 128 | 264 ms | 5.8x |
| 10 | 256 | 128 | 630 ms | 1.9x |
| 10 | 2048 | 64 | 644 ms | 1.9x |
| 10 | 2048 | 128 | 774 ms | 2.3x |
| 10 | 2048 | 512 | 1,682 ms | 5.0x |

With many questions the token cap matters far more than the patch budget, which is why high
resolution goes to the tower and the cap stays small. Ask for more patches only when the answer
depends on fine detail such as small text.

## Limits

* Images plus text must fit `max_len`. Images come before the text, so long text is truncated but an
  image never is; if the images alone do not fit, the call raises and names the token counts.
* Not supported yet: `fast=True` (the TileLang path only takes token ids), ONNX export,
  `predict_long`, `Router.predict_batch`, and image upload in `laya-serve` / MCP.
* Video: pass sampled frames as an ordered image list. There is no frame sampling or video decoding
  in laya, and a clip longer than the budget has to be split by the caller.

## Training a vision checkpoint

All three scripts live in `research/scripts/` and need `pip install datasets` on top of
`laya[vision]`:

```bash
# data: public sets, one question per JSONL row, images at their original resolution
python research/scripts/build_vision_data.py --out data/vision/train --split train
python research/scripts/build_vision_data.py --out data/vision/eval --split eval --per-source 500

# stage 1: train only the image projector, from the English checkpoint + SigLIP2 NaFlex
python research/scripts/train_vision.py --data data/vision/train --stage align --sources coco --out ckpt/align
# stage 2: also the text encoder and head, with 40% text-only replay
python research/scripts/train_vision.py --data data/vision/train --stage joint --init ckpt/align --out ckpt/joint

# accuracy, ECE, auto-label coverage by patch budget, and text before/after
python research/scripts/eval_vision.py --ckpt ckpt/joint --data data/vision/eval
```

`eval_vision.py` also reports **coverage at a fixed precision**: the share of rows an
auto-labelling pipeline could accept without review at, by default, 98% precision. For your own
labels, build a few hundred rows in the same JSONL format and check that number before accepting
answers automatically.
