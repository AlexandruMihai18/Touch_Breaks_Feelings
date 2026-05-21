# Data Annotation Overview — Hand-Object Touch Segmentation

## Task definition

Produce per-frame binary touch masks for videos showing **a hand touching an object**.
A touch mask is the region that straddles the boundary between the hand and the object:
it covers a narrow zone of both surfaces where contact occurs, as defined by the
dilation-intersection algorithm already implemented in `src/touch.py`.

Ground-truth labels are therefore three-layer composites:

1. **Hand mask** — binary, covering the visible hand
2. **Object mask** — binary, covering the contacted object
3. **Touch mask** — derived automatically from `compute_touch_region(hand_mask, object_mask)`

Only the touch mask is the training target. The individual hand/object masks are
intermediate artefacts used to generate it.

---

## Core design principle: keep PoC and automated pipelines structurally identical

The existing Gradio app (`my_inference_script/app_gradio.py`) already implements the full
inference pipeline:

```text
user scribble → SAM2 (point prompts) → mask → touch region algorithm → touch mask
```

The key insight is that **both the manual annotation phase and the eventual automated
phase should feed into exactly this pipeline**. The only thing that changes between
phases is where the initial prompts come from:

| Phase | Hand prompt source | Object prompt source |
| ----- | ------------------ | -------------------- |
| Manual (PoC) | Human scribble in Gradio | Human scribble in Gradio |
| Semi-automated | MediaPipe Hands keypoints | Human box on first frame, SAM2 video propagates |
| Automated | MediaPipe Hands keypoints | 100DOH offset vector → SAM2 point |

Because SAM2 is always the segmentor and `compute_touch_region` always produces the mask,
the annotations from all three phases are in the same format and can be mixed in training.
The leap from PoC to automation is therefore small: you are just replacing the human
who provides the point/scribble prompt.

---

## Phase 1 — Manual annotation (PoC)

### Workflow

1. Extract frames from a video clip using `preprocess_frames.py` (~20 frames per clip).
2. Open the **Video Touch Detection** tab in the Gradio app.
3. Upload the prompt frame into the `ImageEditor`.
4. Draw a scribble on the **hand** in layer 1, and on the **object** in layer 2.
5. Sample/resample frames until you have a representative set.
6. Click **Run Inference** — the app runs SAM2 on the prompt frame and propagates masks
   to all sampled frames via SegGPT.
7. Save outputs: for each frame, save the binary hand mask, the object mask, and the
   touch mask as PNGs.

The Gradio app currently displays but does not export. A small addition — an **Export**
button that saves `{frame_stem}_hand.png`, `{frame_stem}_object.png`,
`{frame_stem}_touch.png` — is the only code change needed to make it an annotation tool.

### What to annotate first

**Recommend sourcing: first-person kitchen or desk videos.**
Reasons:

- Hands are large and centred in frame → easier for SAM2 to segment accurately
- Objects are well-defined (cup, bottle, keyboard) → object mask is unambiguous
- Lots of public data already exists in this domain (EPIC-Kitchens, HO-3D) if you want
  to cross-validate your annotations later

Aim for at least:

- 5–10 distinct videos
- 3–5 different objects per video
- Both "touching" and "not touching" frames (~50/50 split)
- Variation in hand pose, lighting, and viewpoint

### Quality bar

Because you will fine-tune SegGPT, which is an in-context learner tolerant of noisy
labels, the masks do not need to be pixel-perfect. Aiming for "roughly correct at the
boundary" is sufficient for a first PoC. Fix systematic errors (e.g. SAM2 consistently
including the wrist), not random boundary imprecision.

---

## Phase 2 — Semi-automated annotation

The bottleneck in Phase 1 is drawing two scribbles per clip. Phase 2 removes the
hand scribble by replacing it with MediaPipe Hands keypoints.

### Tool: MediaPipe Hands

- **Cost**: free, CPU-only, ~30 fps on a laptop
- **Output**: 21 3D landmarks per hand per frame, handedness (left/right)
- **SAM2 compatibility**: the wrist landmark (index 0) and middle-finger MCP
  (index 9) make reliable foreground-point prompts for SAM2; the palm bounding box
  derived from landmarks makes a reliable box prompt

**Integration sketch:**

```python
import mediapipe as mp
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
result = hands.process(rgb_frame)
if result.multi_hand_landmarks:
    lms = result.multi_hand_landmarks[0].landmark
    wrist_xy = (int(lms[0].x * W), int(lms[0].y * H))
    mid_mcp_xy = (int(lms[9].x * W), int(lms[9].y * H))
    # feed both as positive SAM2 point prompts → hand mask
```

The object scribble can still be drawn manually in the first frame; SAM2 video mode
then propagates both masks forward through the clip.

### Output quality check

After automated mask generation, run a fast visual QC step:

- Flag frames where the hand mask covers less than 500 px (likely a failed detection)
- Flag frames where the touch mask is empty despite both masks being non-empty
  (likely a false-negative: dilation radius may need to be increased for this clip)

---

## Phase 3 — Automated annotation

### Tool: 100DOH (Hand-Object Detector)

**Reference**: Shan et al., "Understanding Human Hands in Contact at Internet Scale",
CVPR 2020. Code: `https://github.com/ddshan/hand_object_detector`

**What it gives you:**

- Hand bounding box
- **Contact state**: no-contact / self-contact / other-person / portable-object / stationary-object
- An offset vector pointing from the hand centre to the contacted object
- The object bounding box (estimated)

**Why this matters:** the contact-state classification is a direct "is the hand touching
an object?" signal. You can filter to `portable-object` and `stationary-object` frames
and skip the rest entirely, giving you a clean positive-only set to hand to SAM2.

**Integration into the pipeline:**

```text
for each frame:
    detections = hod.detect(frame)
    for hand in detections:
        if hand.contact_state in ("portable", "stationary"):
            hand_box → SAM2 box prompt → hand_mask
            hand.offset_vector → object point → SAM2 point prompt → object_mask
            touch_mask = compute_touch_region(hand_mask, object_mask)
            save(frame, touch_mask)
```

**Compute cost:** Faster R-CNN backbone, runs on GPU at ~15 fps; on CPU ~2–3 fps.
For offline annotation of video clips this is acceptable.

### Alternative: EgoHOS

Segments both the hand and the contacted object jointly in a single forward pass.
Trained on EPIC-Kitchens. Produces cleaner boundaries around the contact zone than
the two-step approach above. Requires a GPU. Use this if you have GPU access and want
higher-quality automated masks.

---

## Known problems and mitigations

### 1. Hand-near-but-not-touching (false positives)

**Problem:** The dilation-intersection touch algorithm fires whenever hand and object
masks are within `dilation_radius` pixels of each other, even if there is a visible gap.

**Mitigations:**

- Use 100DOH contact state as a hard gate: only run the touch algorithm on frames
  classified as "contact"
- Reduce `dilation_radius` for annotation (e.g. 10–15 px rather than 25 px)
- For Phase 1 manual annotation: only annotate frames where you can see the hand
  physically pressing against the object

### 2. Object segmentation is underspecified

**Problem:** Without knowing the object category, SAM2 needs a good prompt to pick
the right region. A point prompt may segment part of a complex background instead of
the object.

**Mitigations:**

- Use a bounding box prompt (not just a point) for objects; box prompts are more
  stable for SAM2 on ambiguous scenes
- For Phase 1: draw a dense scribble covering most of the object surface, not just
  one point
- In automated mode: 100DOH's offset vector gives an approximate object centre;
  this is usually good enough to land inside the object

### 3. Occlusion at the contact boundary

**Problem:** The exact contact zone is often partially hidden by the hand itself.
SAM2 may segment the visible part of the object but miss the occluded part that is
actually being touched.

**Mitigations:**

- The touch mask is inherently at the boundary — it does not need to cover the full
  occluded region, only the visible contact zone
- For quality checks, consider the touch mask valid if it overlaps with both the
  hand mask boundary and the object mask boundary, even if small

### 4. Temporal flickering of touch masks

**Problem:** SAM2 + touch region processing frame-independently produces masks that
flicker: present in frame N, absent in frame N+1, present again in frame N+2.

**Mitigations:**

- Apply temporal smoothing: mark a frame as "touch" only if touch is present in
  at least K of the surrounding 2K+1 frames (e.g. K=2)
- Use SAM2's **video object segmentation** mode (tracking across frames) rather than
  per-frame inference — this gives temporally consistent masks by design

### 5. Class imbalance (most frames have no touch)

**Problem:** If you extract 20 frames uniformly per video, the majority will show the
hand not touching anything, producing many empty touch masks. Training on this will
bias the model toward predicting "no touch".

**Mitigations:**

- Use 100DOH to identify contact frames and sample disproportionately from them
- Explicitly include "no-touch" frames as hard negatives in the training set with
  an empty-mask label (this may actually improve specificity)
- Aim for at most 2:1 no-touch:touch ratio in the final annotation set

### 6. Distribution mismatch between Phase 1 and Phase 2/3 masks

**Problem:** Manual scribbles → SAM2 and MediaPipe keypoints → SAM2 might produce
slightly different mask boundaries for the same scene, because the prompts differ.

**Mitigations:**

- This is less critical than it sounds: both use SAM2 as the segmentor, so the
  mask style is structurally the same
- The touch region smooths over small boundary differences because of the dilation step
- Annotate at least 5–10 clips manually and compare them to the automated output on
  the same clips; if boundary overlap (IoU) is > 0.7 the annotations are compatible

---

## Suggested public datasets for bootstrapping

If manual annotation throughput is too low for a first PoC, these datasets have
hand-object interactions with annotations that can be adapted:

| Dataset | Size | What you get | Adaptation needed |
| ------- | ---- | ------------ | ----------------- |
| **EPIC-Kitchens** | 700 hrs video | Action labels, object tracks | Run 100DOH + SAM2 to generate masks |
| **HO-3D** | ~66K frames | 3D hand + object mesh | Project meshes to 2D masks |
| **ContactPose** | 2.9K grasps | Contact maps on object surface | Re-project to image plane |
| **100DOH** | 100K images | Hand boxes + contact labels | Run SAM2 on hand boxes |
| **MECCANO** | 64 videos | Egocentric assembly + HOI labels | Run full pipeline |

EPIC-Kitchens + 100DOH is the most practical starting point: first-person, high
diversity, and 100DOH was literally trained on it, so detection quality will be high.

---

## Recommended execution order

```text
Week 1 (PoC validation)
  ├── Add Export button to Gradio app
  ├── Manually annotate 3–5 short clips (~50–100 frame/mask pairs)
  └── Verify training pipeline runs end-to-end with these annotations

Week 2 (semi-automation)
  ├── Integrate MediaPipe Hands as automatic hand-prompt source
  ├── Annotate 10–20 clips with semi-automated pipeline
  └── Compare hand mask quality vs. manual baseline (target: IoU > 0.7)

Week 3+ (scaled annotation)
  ├── Integrate 100DOH for contact detection + object prompts
  ├── Process larger clip set; apply temporal smoothing
  └── Build balanced train/val split with generate_annotations.py
```
