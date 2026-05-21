# SegGPT Model Architecture

## Overview

SegGPT is a **dual-stream masked image modeling** architecture built on ViT-L. The core idea: it runs an image stream and a mask stream through the same transformer in parallel, merges them early, then reconstructs the masked portion of the mask stream using a lightweight decoder. At inference, the "masked portion" is the target mask to predict.

---

## Data flow — what happens inside `forward()`

Before the encoder even sees anything, two stitching operations happen in `SegGptModel.forward()`:

```python
# Stack context image ON TOP OF target image (along height, dim=2)
pixel_values = torch.cat((prompt_pixel_values, pixel_values), dim=2)
# → (B, 3, 896, 448)

# Stack context mask ON TOP OF label (or mask_token at inference)
prompt_pixel_values = torch.cat((prompt_masks, labels), dim=2)
# → (B, 3, 896, 448)
```

So from this point on, there is **no separate concept of "context" and "target"** — both are concatenated into a single 896×448 image per stream.

---

## Architecture diagram

```mermaid
flowchart TD
    subgraph IN["Inputs to forward()"]
        TV["pixel_values\ntarget image · 448×448×3"]
        PV["prompt_pixel_values\ncontext image · 448×448×3"]
        PM["prompt_masks\ncontext mask · 448×448×3"]
        LB["labels\ntarget mask · 448×448×3\n(mask_token at inference)"]
    end

    subgraph STITCH["1 · Stitching  (SegGptModel.forward)"]
        IS["image stream\ncat(context_img ↕ target_img)\n896×448×3"]
        MS["mask stream\ncat(context_mask ↕ label)\n896×448×3"]
    end

    PV & TV --> IS
    PM & LB --> MS

    subgraph EMB["2 · Embeddings  (SegGptEmbeddings)"]
        PE1["PatchEmbed — Conv2d 16×16\n→ B×56×28×1024"]
        PE2["PatchEmbed — Conv2d 16×16\n→ B×56×28×1024"]
        MKT["bool_masked_pos replaces\ntarget-half patches → mask_token\n(only in mask stream)"]
        TK["Add per-stream learned tokens:\n• segment_token_input  (image stream)\n• segment_token_prompt (mask stream)\n• position_embed (bicubic-interpolated)\n• type_token_semantic OR type_token_instance"]
        BCAT["cat(img_emb, mask_emb) along batch\n→ 2B×56×28×1024"]
    end

    IS --> PE1 --> TK
    MS --> PE2 --> MKT --> TK
    TK --> BCAT

    subgraph ENC["3 · Encoder  (24 × SegGptLayer)"]
        direction TB
        E01["Layers 0–1\nTwo separate streams · 2B×56×28×1024\nNo cross-stream attention"]
        MRG["Layer 2 — MERGE\navg(img_stream, mask_stream)\n→ B×56×28×1024"]
        E323["Layers 3–23\nSingle fused stream · B×56×28×1024"]
        IHS["Tap intermediate states at layers 7, 11, 15, 19\n→ 4 × (B×56×28×1024)"]
    end

    BCAT --> E01 --> MRG --> E323 --> IHS

    subgraph DEC["4 · Decoder  (SegGptDecoder)"]
        DE["decoder_embed · Linear\n4×1024=4096 → 16²×512\nreshape → B×512×56×28"]
        DP["decoder_pred · SegGptDecoderHead\nConv3×3 → LayerNorm → GELU → Conv1×1\n→ B×3×896×448"]
    end

    IHS -- "concat channels" --> DE --> DP

    subgraph LOSS["5 · Loss  (training only)"]
        GT["ground_truth\ncat(prompt_masks, labels) · B×3×896×448"]
        L["SegGptLoss\nsmooth-L1(pred, GT)\nweighted by bool_masked_pos\n(only masked patches contribute)"]
    end

    DP --> L
    GT --> L
```

---

## SegGptLayer internals (×24)

Each layer is a standard pre-norm ViT block with **decomposed relative position bias** (from MViTv2) instead of absolute position bias in attention.

```mermaid
flowchart LR
    X(["hidden_states\nB×H×W×1024"]) --> LN1["LayerNorm"]
    LN1 --> ATT["Multi-head Attention\n16 heads · scale head_dim⁻⁰·⁵\n+ decomposed rel_pos_h, rel_pos_w bias"]
    ATT --> FE{"feature_ensemble?\n(multi-prompt inference)"}
    FE -- yes --> AVG["avg input-query features\nacross prompts"]
    FE -- no --> DP1["DropPath"]
    AVG --> DP1
    DP1 --> R1(("+")) --> LN2["LayerNorm"] --> MLP["MLP\nLinear 1024→4096\nGELU\nLinear 4096→1024"] --> DP2["DropPath"] --> R2(("+"))
    X --> R1
    R1 --> R2
```

---

## Key design decisions

| Decision | Where | Why |
| -------- | ------ | --- |
| **Two stitched streams** (image + mask), not a single stream | `SegGptModel.forward()` | Lets the model learn to copy mask patterns from the context half to the target half |
| **Early merge at layer 2** instead of processing both streams all the way | `SegGptEncoder` | Saves ~50% compute for the first 3 layers; image and mask features are semantically aligned early anyway |
| **Five learnable tokens** per patch: `mask_token`, `segment_token_input`, `segment_token_prompt`, `type_token_semantic`, `type_token_instance` | `SegGptEmbeddings` | `mask_token` = what the model has to reconstruct; the segment/type tokens differentiate streams without changing the attention architecture |
| **`bool_masked_pos` only replaces patches in the mask stream**, not the image stream | `SegGptEmbeddings.forward()` | The model always sees the full target image — it only has to predict the mask, not the image content |
| **Intermediate hidden states tapped at 4 layers** (7, 11, 15, 19), concatenated → decoder | `SegGptEncoder` + `SegGptDecoder` | Multi-scale features capture both coarse semantics (deep) and fine spatial detail (shallow) |
| **Decoder is tiny** (Linear + Conv3×3 + Conv1×1) | `SegGptDecoder` | The heavy lifting is done by the encoder; the decoder just un-patchifies |
| **Loss only on masked patches** | `SegGptLoss` | Same principle as MAE — model gets no gradient for the visible context patches |
| **Decomposed relative position bias** (separate H and W 1D embeddings) | `SegGptAttention` | Works at arbitrary resolutions; absolute position embeddings are bicubic-interpolated for non-pretrain sizes |

---

## Tensor shapes at a glance (ViT-L, 448×448 input)

| Stage | Shape |
| ----- | ----- |
| Raw input (target or context image) | `(B, 3, 448, 448)` |
| After stitching (image or mask stream) | `(B, 3, 896, 448)` |
| After patch embedding | `(B, 56, 28, 1024)` |
| After batch-cat of two streams | `(2B, 56, 28, 1024)` |
| After merge at layer 2 | `(B, 56, 28, 1024)` |
| Intermediate states (×4, concatenated) | `(B, 56, 28, 4096)` |
| Decoder embed output | `(B, 512, 56, 28)` |
| Decoder pred output (pred_masks) | `(B, 3, 896, 448)` |
| Target half of pred_masks | `(B, 3, 448, 448)` |

---

## Training vs. inference difference

| Parameter | Training | Inference |
| --------- | -------- | --------- |
| `labels` | ground-truth target mask | `None` — context mask is repeated |
| `bool_masked_pos` | block-wise 75% masking of target patches | full target-half masking (all 784 patches) |
| Loss | computed on masked patches | not computed |
| Target half of `pred_masks` | reconstruction of masked GT | **the predicted segmentation mask** |
